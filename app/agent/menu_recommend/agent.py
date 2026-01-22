from langgraph.graph import StateGraph, START, END
from app.agent.menu_recommend.state import State
from app.agent.menu_recommend.node import fetch_user_info
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from app.agent.menu_recommend.neo4j_db import Neo4jService
from typing import Annotated
import operator
import uuid
import os


class UserPreference(BaseModel):
    """사용자의 음식 취향 정보"""
    item: str = Field(description="음식명, 식재료명 또는 메뉴 카테고리 (예: 오이, 김치찌개, 일식)")
    preference_type: Literal["LIKES", "DISLIKES", "ALLERGIC_TO"] = Field(
        description="취향 유형: 좋아함(LIKES), 싫어함(DISLIKES), 알러지(ALLERGIC_TO)"
    )
    reason: Optional[str] = Field(description="취향의 이유 (선택 사항)")


class PreferenceList(BaseModel):
    """추출된 취향 리스트"""
    preferences: List[UserPreference]


class AgentState(TypedDict):
    user_id: str
    messages: Annotated[list, operator.add]
    result: str
    context: List[dict]  # Neo4j에서 가져온 데이터
    extracted_prefs: List[UserPreference]  # 이번 대화에서 추출된 취향


# 2. 컨텍스트 로드 노드
async def load_memories(state: AgentState):
    # 실제 운영 시에는 인스턴스를 외부에서 주입(DI)받는 것을 권장합니다.
    db_service = Neo4jService(
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    )

    raw_context = await db_service.get_user_context(state["user_id"])

    # LLM이 읽기 좋은 텍스트 형태로 변환
    formatted_context = "\n".join([
        f"- {c['name']} ({c['relationship']})" for c in raw_context
    ]) if raw_context else "기존 취향 정보가 없습니다."
    print(formatted_context)
    return {"context": formatted_context}


# 3. 모델 실행 노드: 취향을 반영하여 답변 생성
async def call_agent(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 개인 맞춤형 음식 추천 전문가입니다. 
        아래의 사용자 취향 정보를 바탕으로 메뉴를 추천하거나 질문에 답하세요.

        [사용자 기존 취향]
        {user_context}

        대화 시 지침:
        1. 싫어하거나 알러지가 있는 식재료가 포함된 메뉴는 절대 추천하지 마세요.
        2. 좋아하는 메뉴와 유사한 스타일을 우선적으로 고려하세요.
        3. 정중하고 친절하게 답변하세요."""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm
    response = await chain.ainvoke({
        "user_context": state["context"],
        "messages": state["messages"]
    })

    return {"messages": [response]}


# 4. 취향 추출 및 DB 저장 노드: 대화에서 새로운 정보가 있는지 확인
async def sync_preferences(state: AgentState):
    # Structured Output 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(PreferenceList)

    db_service = Neo4jService(
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    )

    # 최근 대화 내용을 분석하여 새로운 취향이 언급되었는지 확인
    # 시스템 메시지를 통해 명시적으로 추출 지시
    extract_prompt = [
        SystemMessage(content="사용자의 마지막 메시지에서 음식 취향(좋아함, 싫어함, 알러지)이 있다면 추출하세요. 없다면 빈 리스트를 반환하세요."),
        state["messages"][0]  # 마지막 HumanMessage 또는 AI의 확인 멘트
    ]

    extracted = await llm.ainvoke(extract_prompt)
    print(extracted)
    if extracted and extracted.preferences:
        for pref in extracted.preferences:
            await db_service.upsert_preference(
                user_id=state["user_id"],
                item_name=pref.item,
                pref_type=pref.preference_type
            )
        return {"extracted_prefs": extracted.preferences}

    return {"extracted_prefs": []}


# --- 그래프 구축 ---
workflow = StateGraph(AgentState)

workflow.add_node("load_memories", load_memories)
workflow.add_node("agent", call_agent)
workflow.add_node("sync_db", sync_preferences)

workflow.add_edge(START, "load_memories")
workflow.add_edge("load_memories", "agent")
workflow.add_edge("agent", "sync_db")
workflow.add_edge("sync_db", END)

graph = workflow.compile()
