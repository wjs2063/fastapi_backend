import operator
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.store.postgres import AsyncPostgresStore
from pydantic import BaseModel, Field

from app.agent.menu_recommend.tools import NaverLocalSearchTool
from app.clients.naver import naver_map_client


class UserPreference(BaseModel):
    """사용자의 개별 취향 정보"""

    category_path: str = Field(
        description="계층적 카테고리 경로. '대분류 > 중분류 > 소분류' 형식 (예: '음식 > 일식 > 초밥')"
    )
    preference_type: Literal["LIKES", "DISLIKES", "ALLERGIC_TO"] = Field(
        description="취향 유형: 좋아함(LIKES), 싫어함(DISLIKES), 알러지(ALLERGIC_TO)"
    )
    domain: Literal["FOOD", "TRAVEL", "LIFESTYLE"] = Field(
        default="FOOD", description="취향의 도메인 분야"
    )
    reason: str | None = Field(default=None, description="취향의 이유나 구체적인 설명")


class PreferenceList(BaseModel):
    """추출된 취향 정보의 리스트 컨테이너"""

    preferences: list[UserPreference] = Field(
        description="사용자 대화에서 추출된 모든 취향 정보 리스트"
    )


class UserInfo(BaseModel):
    lat: float | None = None
    lng: float | None = None


# --- State 정의 ---
class AgentState(TypedDict):
    user_id: str
    request_id: str
    query: str
    history: list[BaseMessage]
    context: str
    search_params: dict[str, int]  # 동적 페이징용
    has_results: bool
    user_info: UserInfo
    answer: AIMessage
    address: str | None  # 변환된 주소 (예: "강동구 천호동")
    is_related: bool  # 가드레일 통과 여부 플래그
    internal_steps: Annotated[list[str | BaseMessage | ToolMessage], operator.add]


class RelevanceCheck(BaseModel):
    """질문의 관련성 판별 결과"""

    is_related: bool = Field(description="음식/맛집 관련 질문 여부")
    reason: str = Field(
        description="이유를 한글로 설명 (예: '음식과 관련 없는 일상 대화입니다')"
    )


async def fetch_chat_history(state: AgentState, config: RunnableConfig):
    store: AsyncPostgresStore = config["configurable"]["store"]

    items = await store.asearch((state["user_id"],))
    items.sort(key=lambda x: x.updated_at, reverse=True)
    history = []

    for item in items:
        print(item)
        history.extend(messages_from_dict(item.value["history"]))
    return {"history": history}


async def guardrail(state: AgentState, config: RunnableConfig):
    """최근 5쌍(10개)의 대화를 분석하여 질문의 관련성을 판별합니다."""

    # 1. 메시지 리스트에서 최근 10개만 슬라이싱 (Human-AI 대화 약 5쌍)
    # messages가 10개보다 적어도 파이썬 슬라이싱은 에러 없이 있는 만큼만 가져옵니다.

    recent_context = state["history"]

    if not recent_context:
        return {"is_related": True}

    # 2. LLM 설정 (Structured Output)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=300,  # 가드레일은 짧은 응답이면 충분하므로 줄임
    ).with_structured_output(RelevanceCheck)

    # 3. 맥락 기반 판별 요청
    try:
        check = await llm.ainvoke(
            [
                SystemMessage(
                    content="""당신은 맛집 추천 서비스의 보안 가드입니다.
            제공된 '최근 대화 맥락'을 바탕으로 사용자의 마지막 질문이 서비스 범위(음식, 맛집, 식당, 요리, 취향)에 속하는지 판별하세요.

            [판단 로직]
            1. 맥락 우선: 질문 자체에 '음식' 단어가 없어도, 이전 대화가 맛집 추천 중이었고 "거기는 어때?", "다른 데는?" 같은 질문이라면 관련 있음(True)입니다.
            2. 인사 및 종료: "안녕", "고마워", "잘 가" 등 기본적인 대화 예절은 관련 있음(True)으로 간주합니다.
            3. 주제 이탈: 갑작스러운 코딩 질문, 정치, 일반 상식 등은 관련 없음(False)입니다.

            반드시 한글로 이유(reason)를 짧게 포함하여 JSON으로 응답하세요."""
                ),
                # 최근 10개의 대화 내용을 모두 전달
                *recent_context,
                HumanMessage(content=state["query"]),
            ]
        )

        if not check.is_related:
            rejection_msg = AIMessage(
                content="🍕 저는 음식과 맛집에 대해서만 이야기할 수 있는 전문가예요! 먹고 싶은 메뉴나 맛집 취향에 대해 물어봐 주시겠어요?"
            )
            # 가드레일에 걸리면 중단 플래그와 거절 메시지 반환
            return {"answer": rejection_msg, "is_related": False}

    except Exception:
        # 에러 발생 시 사용자 경험을 위해 일단 통과시키는 Fallback
        return {"is_related": True}

    return {"is_related": True}


# 2. 조건부 엣지 함수
def route_after_guardrail(state: AgentState, config: RunnableConfig):
    """
    is_related가 False면 바로 종료(END) 시그널을 보냅니다.
    """
    if state.get("is_related") is False:
        # 관련 없는 질문은 뒤도 안 돌아보고 END!
        return "terminate"
    return "continue"


async def resolve_location(state: AgentState, config: RunnableConfig):
    """좌표가 있다면 주소로 변환하여 state에 저장"""
    lat = state["user_info"].lat
    lng = state["user_info"].lng
    if lat and lng:
        map_client = naver_map_client
        address = await map_client.get_address(lat=lat, lng=lng)
        return {"address": address}
    return {"address": None}


# --- 노드 구현 ---
async def load_preference(state: AgentState, config: RunnableConfig):
    db = config["configurable"].get("neo4j_service")
    prefs = await db.get_user_context(state["user_id"])

    formatted = (
        "\n".join(
            [f"- {' > '.join(p['category_path'])}: {p['preference_type']}" for p in prefs]
        )
        if prefs
        else "취향 정보 없음"
    )
    return {
        "context": formatted,
        "search_params": {"start": 1, "display": 5, "retry_count": 0},
    }


async def call_agent(state: AgentState, config: RunnableConfig):
    # 1. 모델과 도구 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # create_react_agent는 도구 호출 -> 실행 -> 결과 확인 -> 답변 루프를 자동으로 수행합니다.
    agent = create_agent(llm, tools=[NaverLocalSearchTool()])

    clean_history = []
    for i, msg in enumerate(state["history"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # 다음 메시지들이 이 도구 호출들에 대한 응답인지 확인
            has_tool_responses = any(
                isinstance(next_msg, ToolMessage)
                for next_msg in state["history"][i + 1 :]
            )
            if not has_tool_responses:
                # 결과가 없다면 tool_calls를 제거하여 일반 메시지로 변환 (OpenAI 에러 방지)
                msg = AIMessage(content=msg.content, tool_calls=[])
        clean_history.append(msg)

    # 2. 시스템 지침 구성 (히스토리는 여기서 제외)
    loc_context = f"\n[현재 위치] {state['address']}" if state.get("address") else ""
    system_content = f"""당신은 맛집 추천 전문가입니다.
    [사용자 취향 정보]
    {state["context"]}
    {loc_context}

    지침:
    1. 사용자의 현재 위치를 기반으로 맛집을 검색하세요.
    2. 'naver_local_search' 도구를 호출할 때, 쿼리에 반드시 지역명과 메뉴를 포함하세요.
       예: "{state.get("address", "")} 초밥 맛집"
    3. 결과가 만족스럽지 않으면 검색 키워드를 바꿔서 재시도하세요.
    4. 모든 검색 결과가 나오면 최종적으로 사용자에게 친절하게 맛집을 추천하세요."""
    input_messages = (
        [SystemMessage(content=system_content)]
        + clean_history
        + [
            SystemMessage(
                content="사용자와의 대화기록을 기반으로 맥락을 파악하여, 다른 식당을 추천해야할지, "
                "다른 음식카테고리를 추천해야할지 등을 파악하여 자율적으로 대답하세요"
            )
        ]
        + [HumanMessage(content=state["query"])]
    )
    # 4. 에이전트 실행 (자율 루프 시작)
    # ainvoke는 모든 도구 호출 단계가 끝날 때까지 내부적으로 반복 실행됩니다.
    result = await agent.ainvoke({"messages": input_messages}, config)

    # 5. 결과 반환
    # result["messages"]의 마지막 요소가 모든 루프를 마친 LLM의 최종 답변입니다.
    final_answer = result["messages"][-1]

    return {
        "answer": final_answer,
        "internal_steps": result[
            "messages"
        ],  # 도구 호출 과정 전체를 보관하고 싶을 때 사용
    }


def adjust_params(state: AgentState, config: RunnableConfig):
    curr = state["search_params"]
    return {
        "search_params": {
            "start": curr["start"] + 5,
            "display": 5,
            "retry_count": curr["retry_count"] + 1,
        }
    }


async def sync_db(state: AgentState, config: RunnableConfig):
    """사용자의 마지막 메시지에서만 취향을 추출하여 Neo4j에 저장"""
    db = config["configurable"].get("neo4j_service")
    store: AsyncPostgresStore = config["configurable"]["store"]
    # 1. 메시지 기록 중 사용자가 보낸 것만 필터링
    user_messages = [HumanMessage(content=state["query"])] + [
        m for m in state["internal_steps"] if isinstance(m, HumanMessage)
    ]

    # 사용자의 메시지가 없으면(그럴 리 없겠지만 방어코드) 종료
    if not user_messages:
        return {}

    # 가장 최근의 사용자 메시지 선택
    last_user_query = user_messages[-1].content

    # 2. Structured Output 설정
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=500,
    ).with_structured_output(PreferenceList)

    # 3. 추출 수행 (사용자의 발화 내용만 전달)
    extracted = await llm.ainvoke(
        [
            SystemMessage(
                content="사용자의 발화에서 음식/여행 취향을 '대분류 > 중분류 > 소분류' 경로로 추출하세요. 추천 결과가 아닌 사용자의 실제 선호도만 추출해야 합니다."
            ),
            HumanMessage(content=last_user_query),
        ]
    )

    # 4. DB 저장
    if extracted and extracted.preferences:
        for p in extracted.preferences:
            await db.upsert_hierarchical_preference(
                state["user_id"], p.category_path, p.preference_type, p.domain
            )
    await store.aput(
        (state["user_id"], state["request_id"]),
        key="chat",
        value={
            "history": [
                message_to_dict(HumanMessage(content=state["query"])),
                message_to_dict(state["answer"]),
            ]
        },
    )
    return {}


# --- 그래프 구축 ---
workflow = StateGraph(AgentState)
workflow.add_node("fetch_chat_history", fetch_chat_history)
workflow.add_node("resolve_location", resolve_location)
workflow.add_node("load_memories", load_preference)
workflow.add_node("agent", call_agent)
workflow.add_node("sync_db", sync_db)
workflow.add_node("guardrail", guardrail)


workflow.add_edge(START, "fetch_chat_history")
workflow.add_edge("fetch_chat_history", "guardrail")
# 가드레일 결과에 따른 분기 로직 업데이트
workflow.add_conditional_edges(
    "guardrail",
    route_after_guardrail,
    {
        "continue": "resolve_location",  # 정상 진행
        "terminate": END,  # 즉시 종료
    },
)
workflow.add_edge("resolve_location", "load_memories")
workflow.add_edge("load_memories", "agent")
workflow.add_edge("agent", "sync_db")
workflow.add_edge("sync_db", END)
