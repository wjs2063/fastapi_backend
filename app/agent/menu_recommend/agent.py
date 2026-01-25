from pydantic import BaseModel, Field
from yaml import serialize

from app.agent.menu_recommend.neo4j_db import Neo4jService
from app.agent.menu_recommend.tools import NaverLocalSearchTool
import operator
import os
from typing import Annotated, TypedDict, Dict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Optional, List
from langchain.messages import HumanMessage, AIMessage

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
        default="FOOD",
        description="취향의 도메인 분야"
    )
    reason: Optional[str] = Field(
        default=None,
        description="취향의 이유나 구체적인 설명"
    )


class PreferenceList(BaseModel):
    """추출된 취향 정보의 리스트 컨테이너"""
    preferences: List[UserPreference] = Field(
        description="사용자 대화에서 추출된 모든 취향 정보 리스트"
    )


class UserInfo(BaseModel):
    lat: float | None = None
    lng: float | None = None


# --- State 정의 ---
class AgentState(TypedDict):
    user_id: str
    messages: Annotated[list, operator.add]
    context: str
    search_params: Dict[str, int]  # 동적 페이징용
    has_results: bool
    user_info: UserInfo
    address: Optional[str]  # 변환된 주소 (예: "강동구 천호동")
    is_related: bool  # 가드레일 통과 여부 플래그


class RelevanceCheck(BaseModel):
    """질문의 관련성 판별 결과"""
    is_related: bool = Field(description="음식/맛집 관련 질문 여부")
    reason: str = Field(description="이유를 한글로 설명 (예: '음식과 관련 없는 일상 대화입니다')")


async def guardrail(state: AgentState):
    """사용자의 마지막 질문 하나만 추출하여 관련성을 엄격히 판별합니다."""

    # 1. 메시지 리스트에서 마지막 HumanMessage 객체만 안전하게 추출
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"is_related": True}  # 메시지가 없으면 일단 통과

    last_query = user_messages[-1].content

    # 2. LLM 설정 (토큰 제한을 적절히 두어 Truncation 에러 방지)
    # max_tokens를 너무 크게 잡으면(5000 등) 모델이 방황할 확률이 높아집니다.
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=500
    ).with_structured_output(RelevanceCheck)

    # 3. 리스트 형식이 아닌 '단일 질문' 관점으로 프롬프트 재구성
    try:
        check = await llm.ainvoke([
            SystemMessage(content="""당신은 맛집 추천 서비스의 보안 가드입니다.
            사용자의 입력이 '음식, 메뉴, 식당, 맛집, 요리, 취향' 중 하나라도 관련이 있는지 판별하세요.

            [판단 기준]
            - 관련 있음(True): "강남역 맛집 추천해줘", "초밥 좋아해", "매운 건 못 먹어", "안녕(인사)"
            - 관련 없음(False): "오늘 날씨 어때?", "비트코인 시세 알려줘", "파이썬 코드 짜줘" 등

            반드시 한글로 이유(reason)를 짧게 포함하여 JSON으로 응답하세요."""),
            HumanMessage(content=f"사용자 질문: {last_query}")  # 문자열로 감싸서 전달
        ])

        # 디버깅 출력 (객체 형태 확인)
        print(f"--- Guardrail Check ---\nQuery: {last_query}\nResult: {check}\n------------------------")

        if not check.is_related:
            rejection_msg = AIMessage(content="🍕 저는 음식과 맛집에 대해서만 이야기할 수 있는 전문가예요! 음식 취향이나 먹고 싶은 메뉴에 대해 물어봐 주시겠어요?")
            # 에러 방지: 리스트에 새 메시지를 추가하고 플래그 설정
            return {"messages": [rejection_msg], "is_related": False}

        return {"is_related": True}

    except Exception as e:
        # LLM 파싱 에러 발생 시 안전하게 통과시키는 Fallback 로직
        print(f"Guardrail Error: {e}")
        return {"is_related": True}


# 2. 조건부 엣지 함수
def route_after_guardrail(state: AgentState):
    """
    is_related가 False면 바로 종료(END) 시그널을 보냅니다.
    """
    if state.get("is_related") is False:
        # 관련 없는 질문은 뒤도 안 돌아보고 END!
        return "terminate"
    return "continue"


async def resolve_location(state: AgentState):
    """좌표가 있다면 주소로 변환하여 state에 저장"""
    lat = state["user_info"].lat
    lng = state["user_info"].lng
    if lat and lng:
        map_client = naver_map_client
        address = await map_client.get_address(lat=lat, lng=lng)
        print(address)
        return {"address": address}
    return {"address": None}


# --- 노드 구현 ---
async def load_memories(state: AgentState):
    db = Neo4jService(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    prefs = await db.get_user_context(state["user_id"])

    formatted = "\n".join([
        f"- {' > '.join(p['category_path'])}: {p['preference_type']}" for p in prefs
    ]) if prefs else "취향 정보 없음"

    print("취향 결과 : ", formatted)
    return {"context": formatted, "search_params": {"start": 1, "display": 5, "retry_count": 0}}


async def call_agent(state: AgentState):
    # 기존 툴 그대로 사용
    llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([NaverLocalSearchTool()])

    # 위치 컨텍스트 생성
    loc_context = f"\n[현재 위치] {state['address']}" if state.get("address") else ""

    system_msg = SystemMessage(content=f"""맛집 추천 전문가입니다.
    [사용자 취향 정보]
    {state['context']}
    {loc_context}

    지침:
    1. 사용자의 현재 위치({state.get('address', '알 수 없음')})를 기반으로 맛집을 검색하세요.
    2. 'naver_local_search' 도구를 호출할 때, 쿼리에 반드시 지역명과 메뉴를 포함하세요.
       예: "{state.get('address', '')} 초밥 맛집"
    3. 결과가 만족스럽지 않으면 검색 키워드를 바꿔서 재시도하세요.""")

    response = await llm.ainvoke([system_msg] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    """검색 결과 유무에 따른 루프 결정"""
    last_msg = state["messages"][-1]

    # 도구 응답 메시지 중 검색 결과가 비어있는지 확인
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage) and m.name == "naver_local_search"]
    if tool_msgs:
        last_result = tool_msgs[-1].content
        if "[]" in last_result or len(last_result) < 10:
            if state["search_params"]["retry_count"] < 2:
                return "adjust_params"

    return tools_condition(state)


async def adjust_params(state: AgentState):
    curr = state["search_params"]
    return {
        "search_params": {
            "start": curr["start"] + 5,
            "display": 5,
            "retry_count": curr["retry_count"] + 1
        }
    }


async def sync_db(state: AgentState):
    """사용자의 마지막 메시지에서만 취향을 추출하여 Neo4j에 저장"""

    # 1. 메시지 기록 중 사용자가 보낸 것만 필터링
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]

    # 사용자의 메시지가 없으면(그럴 리 없겠지만 방어코드) 종료
    if not user_messages:
        return {"messages": []}

    # 가장 최근의 사용자 메시지 선택
    last_user_query = user_messages[-1].content

    # 2. Structured Output 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=500, ).with_structured_output(PreferenceList)

    # 3. 추출 수행 (사용자의 발화 내용만 전달)
    extracted = await llm.ainvoke([
        SystemMessage(content="사용자의 발화에서 음식/여행 취향을 '대분류 > 중분류 > 소분류' 경로로 추출하세요. 추천 결과가 아닌 사용자의 실제 선호도만 추출해야 합니다."),
        HumanMessage(content=last_user_query)
    ])

    # 4. DB 저장
    if extracted and extracted.preferences:
        db = Neo4jService(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USER"),
            os.getenv("NEO4J_PASSWORD")
        )

        for p in extracted.preferences:
            await db.upsert_hierarchical_preference(
                state["user_id"],
                p.category_path,
                p.preference_type,
                p.domain
            )

    return {"messages": []}


# --- 그래프 구축 ---
workflow = StateGraph(AgentState)
workflow.add_node("resolve_location", resolve_location)
workflow.add_node("load_memories", load_memories)
workflow.add_node("agent", call_agent)
workflow.add_node("tools", ToolNode([NaverLocalSearchTool()]))
workflow.add_node("adjust_params", adjust_params)
workflow.add_node("sync_db", sync_db)
workflow.add_node("guardrail", guardrail)

workflow.add_edge(START, "guardrail")
# 가드레일 결과에 따른 분기 로직 업데이트
workflow.add_conditional_edges(
    "guardrail",
    route_after_guardrail,
    {
        "continue": "resolve_location",  # 정상 진행
        "terminate": END  # 즉시 종료
    }
)
workflow.add_edge("resolve_location", "load_memories")
workflow.add_edge("load_memories", "agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "adjust_params": "adjust_params",
    END: "sync_db"
})
workflow.add_edge("tools", "agent")
workflow.add_edge("adjust_params", "agent")
workflow.add_edge("sync_db", END)

graph = workflow.compile()
