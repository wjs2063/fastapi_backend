import json
import logging
import os
import uuid
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect, status
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from fastapi import FastAPI,Request,Depends,status
from app.agent.menu_recommend.agent import UserInfo, workflow
from app.agent.menu_recommend.neo4j_db import Neo4jManager, Neo4jService
from app.agent.menu_recommend.state import init_agent_state
from app.api.deps import SessionDep  # 템플릿의 의존성 활용
from app.clients.naver import find_my_office, naver_search_client
from app.core import security
from app.core.config import settings
from app.models import TokenPayload, User

agent_executor = ChatOpenAI(model="gpt-4o")
logger = logging.getLogger(__name__)

# ... (LangChain 관련 import 생략) ...

router = APIRouter(prefix="/chat", tags=["chat"])


# WebSocket용 인증 함수
def get_current_user_ws(session: SessionDep, token: str = Query(...)) -> User | None:
    """
    WebSocket 연결 시 쿼리 파라미터로 토큰을 받아 유효성을 검증하고 유저를 반환합니다.
    실패 시 None을 반환하거나 예외를 발생시킬 수 있습니다.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[security.ALGORITHM])
        token_data = TokenPayload(**payload)
    except (jwt.InvalidTokenError, ValidationError):
        return None

    user = session.get(User, token_data.sub)
    if not user or not user.is_active:
        return None
    return user


def update_state(msg: str, state: dict) -> dict:
    user_msg = json.loads(msg)

    query = user_msg["text"]
    if user_msg.get("location", {}):
        lat = user_msg["location"]["lat"]
        lng = user_msg["location"]["lng"]
        state["user_info"] = UserInfo(lat=lat, lng=lng)
    state["query"] = query
    return state


from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from app.core.db import connection_pool


async def get_langgraph_storage():
    async with connection_pool.connection() as conn:
        # 이미 테이블은 lifespan에서 생성되었으므로 setup 없이 즉시 반환
        yield AsyncPostgresSaver(conn), AsyncPostgresStore(conn)


@router.websocket("/menu-recommend/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    # WebSocket 연결 시 Query Parameter로 token을 받습니다.
    # 예: ws://localhost:8000/api/v1/chat/ws?token=eyJhbG...
    session: SessionDep,
    user: Annotated[User | None, Depends(get_current_user_ws)],
    storage: Annotated[
        tuple[AsyncPostgresSaver, AsyncPostgresStore], Depends(get_langgraph_storage)
    ],
):
    # 1. 인증 실패 시 즉시 연결 종료 (Policy Violation)
    if user is None:
        # accept() 전에 거부하는 것이 보안상 좋지만,
        # 디버깅 편의를 위해 accept 후 메시지를 보내고 닫기도 합니다.
        # 여기서는 accept 없이 바로 거부하거나, 1008 코드로 닫습니다.
        await websocket.accept()
        await websocket.send_text("Authentication failed.")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    if agent_executor is None:
        await websocket.send_text("System Error: Agent not initialized.")
        await websocket.close()
        return

    # 연결된 유저 로깅 (선택)
    logger.info(f"User connected: {user.email}")

    thread_id = str(user.id)  # 현재는 1:1 채팅이라 동일하게 설정

    saver, store = storage
    state = {
        "user_id": str(user.id),
        "request_id": str(uuid.uuid4()),
        "user_info": UserInfo(),
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": str(user.id),
            "neo4j_service": Neo4jManager.get_service(),
            "store": store,
        }
    }
    config = RunnableConfig(**config)

    graph = workflow.compile(checkpointer=saver, store=store)

    try:
        while True:
            data = await websocket.receive_text()

            # 여기서 user 객체를 활용해 권한 체크나 컨텍스트 주입 가능
            # 예: response = await agent_executor.ainvoke({"input": data, "user_id": user.id})

            try:
                state = update_state(msg=data, state=state)

                response = await graph.ainvoke(state, config=config)
                await websocket.send_text(response["answer"].content)
            except Exception as e:
                logger.error(f"Agent Execution Error: {e}")
                await websocket.send_text(f"Error: {str(e)}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {user.email}")


@router.get("/test")
async def conversation_handler(
    session: SessionDep,
    user: Annotated[User | None, Depends(get_current_user_ws)],
):
    initial_state = init_agent_state(user, message="안녕")
    config = RunnableConfig(
        run_name="test",
        configurable={
            "rdb_session": session,
        },
    )
    graph = workflow.compile()
    print("그래프 결과", await graph.ainvoke(initial_state, config=config))


@router.get("/naver-map")
async def naver_map_handler():
    return await find_my_office()


@router.get("/naver-local-search")
async def naver_local_search_handler(query: str):
    response = await naver_search_client.search_local(query=query, display=10)

    return response


@router.get("/menu-chat")
async def menu_chat(
    query: str = Query(...),
):
    graph = workflow.compile()
    response = await graph.ainvoke(
        {"user_id": "string", "messages": [HumanMessage(content=query)]}
    )
    print(response)
    return response


@router.get("/reset-db")
async def reset_db_handler():
    db = Neo4jService(
        os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")
    )
    await db.reset_db()
    return {"msg": "success"}


async def get_data():
    for i in range(3):
        yield i


@router.get("/async")
async def async_handler():
    async for data in get_data():
        print(data)


