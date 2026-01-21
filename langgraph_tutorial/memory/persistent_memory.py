import asyncio
import psycopg
from contextlib import asynccontextmanager
from typing import Optional, Annotated

from fastapi import FastAPI, Depends, Request, HTTPException
from pydantic import BaseModel
from psycopg_pool import AsyncConnectionPool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import interrupt, Command

# --- [1. Configuration & DB Setup] ---
DB_URI = "postgresql://postgres:changethis@localhost:5432/app"
pool = AsyncConnectionPool(conninfo=DB_URI, max_size=10, open=False)


class GraphState(MessagesState):
    user_id: str
    summary: str = ""


# --- [2. Nodes & Graph Logic] ---
async def assistant_node(state: GraphState, *, store):
    """단기 기억(Messages)과 장기 기억(Store)을 조합하는 노드"""
    user_id = state["user_id"]

    # 1. 장기 기억(Store)에서 사용자 정보 조회
    namespace = (user_id, "memories")
    existing_memories = await store.search(namespace)
    user_info = existing_memories[0].value.get("text") if existing_memories else "정보 없음"

    # 2. 메시지 생성
    last_msg = state["messages"][-1].content
    response = f"유저 정보('{user_info}')를 바탕으로 '{last_msg}'에 대한 초안을 작성했습니다. 승인하시겠습니까?"

    # 3. 인터럽트 발생 (여기서 실행 중단 및 상태 DB 저장)
    feedback = interrupt(response)

    return {
        "messages": [{"role": "assistant", "content": response}],
        "summary": f"사용자 피드백: {feedback}"
    }


def create_graph(checkpointer, store):
    workflow = StateGraph(GraphState)
    workflow.add_node("assistant", assistant_node)
    workflow.add_edge(START, "assistant")
    workflow.add_edge("assistant", END)

    # 문서 가이드에 따라 checkpointer와 store를 모두 등록
    return workflow.compile(checkpointer=checkpointer, store=store)


# --- [3. Dependencies] ---
def get_graph(request: Request):
    return request.app.state.graph


def get_store(request: Request):
    return request.app.state.store


# --- [4. FastAPI & Lifespan] ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB 연결 및 테이블 자동 생성 (autocommit=True로 인덱스 에러 방지)
    await pool.open()
    async with await psycopg.AsyncConnection.connect(DB_URI, autocommit=True) as conn:
        cp = AsyncPostgresSaver(conn)
        st = AsyncPostgresStore(conn)
        await cp.setup()
        await st.setup()

    # 실제 앱용 컴파일
    checkpointer = AsyncPostgresSaver(pool)
    store = AsyncPostgresStore(pool)
    app.state.graph = create_graph(checkpointer, store)
    app.state.store = store

    yield
    await pool.close()


app = FastAPI(lifespan=lifespan)


# --- [5. API Routers] ---

class StartRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    feedback: str


@app.post("/workflow/start")
async def start_workflow(req: StartRequest, graph=Depends(get_graph)):
    """새로운 대화 세션 시작 (단기 기억 생성 시작)"""
    config = {"configurable": {"thread_id": req.thread_id}}
    initial_input = {
        "messages": [{"role": "user", "content": req.message}],
        "user_id": req.id
    }

    # interrupt 지점까지 실행 후 중단된 상태 반환
    result = await graph.ainvoke(initial_input, config)
    return {"status": "paused", "snapshot": result}


@app.post("/workflow/resume")
async def resume_workflow(req: ResumeRequest, graph=Depends(get_graph)):
    """중단된 워크플로우를 유저 피드백과 함께 재개"""
    config = {"configurable": {"thread_id": req.thread_id}}

    # Command(resume=...)을 통해 interrupt()의 반환값을 주입
    result = await graph.ainvoke(Command(resume=req.feedback), config)
    return {"status": "completed", "final_result": result}


@app.post("/memory/long-term")
async def save_long_term(user_id: str, text: str, store=Depends(get_store)):
    """쓰레드와 관계없이 유지되는 장기 기억 저장"""
    namespace = (user_id, "memories")
    await store.aput(namespace, "profile", {"text": text})
    return {"message": "Long-term memory saved"}


@app.get("/workflow/history/{thread_id}")
async def get_history(thread_id: str, graph=Depends(get_graph)):
    """특정 쓰레드의 전체 상태 히스토리 조회"""
    config = {"configurable": {"thread_id": thread_id}}
    history = []
    async for state in graph.aget_state_history(config):
        history.append({
            "step": state.metadata.get("step"),
            "values": state.values,
            "next": state.next
        })
    return {"history": history}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)