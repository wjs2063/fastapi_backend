import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler as langfuse_handler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from app.agent.menu_recommend.agent import UserInfo, workflow
from app.agent.menu_recommend.neo4j_db import Neo4jManager
from app.core.db import connection_pool

llm = ChatOpenAI()
langfuse = get_client()

# ... (기타 필요한 라이브러리 import)

router = APIRouter(prefix="/langfuse", tags=["with_langfuse"])


async def get_langgraph_storage():
    async with connection_pool.connection() as conn:
        # 이미 테이블은 lifespan에서 생성되었으므로 setup 없이 즉시 반환
        yield AsyncPostgresSaver(conn), AsyncPostgresStore(conn)


def update_state(query: str, state: dict) -> dict:
    state["query"] = query
    return state


@router.post("/chat")
async def custom_chat(
    query: str,
    user_id: str,
    storage: Annotated[
        tuple[AsyncPostgresSaver, AsyncPostgresStore], Depends(get_langgraph_storage)
    ],
):
    # 트레이스 및 생성(generation) 기록
    session_id = ":".join([user_id, str(uuid.uuid4())])

    saver, store = storage
    state = {
        "user_id": user_id,
        "request_id": str(uuid.uuid4()),
        "user_info": UserInfo(),
    }
    config = RunnableConfig(
        configurable={
            "thread_id": user_id,
            "user_id": user_id,
            "neo4j_service": Neo4jManager.get_service(),
            "store": store,
        },
        callbacks=[langfuse_handler()],
    )

    graph = workflow.compile(checkpointer=saver, store=store)
    state = update_state(query=query, state=state)
    with langfuse.start_as_current_observation(as_type="span", name="llm_trace") as span:
        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            metadata={"env": "test"},
            tags=["template"],
        ):
            response = await graph.ainvoke(state, config=config)
            return response
