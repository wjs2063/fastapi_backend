from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

import asyncio
from contextlib import asynccontextmanager
from typing import TypedDict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg_pool import AsyncConnectionPool
import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import interrupt, Command


# --- 1. LangGraph 상태 및 노드 정의 ---
class GraphState(TypedDict):
    content: str
    review_result: Optional[str]


async def draft_node(state: GraphState):
    print("--- 초안 작성 중 ---")
    return {"content": "FastAPI와 LangGraph를 활용한 자동화 초안입니다."}


async def human_review_node(state: GraphState):
    print("--- 사용자 승인 대기 (Interrupt) ---")
    # 여기서 실행이 중단되고 DB에 저장됨
    # resume 시 전달되는 값이 feedback 변수에 담깁니다.
    feedback = interrupt("Review the content and provide 'approve' or 'reject'")
    return {"review_result": feedback}


async def finalize_node(state: GraphState):
    print(f"--- 작업 완료: {state['review_result']} ---")
    return state


# --- 2. 그래프 빌드 함수 ---
def create_app_graph(checkpointer):
    workflow = StateGraph(GraphState)
    workflow.add_node("drafter", draft_node)
    workflow.add_node("reviewer", human_review_node)
    workflow.add_node("finalizer", finalize_node)

    workflow.add_edge(START, "drafter")
    workflow.add_edge("drafter", "reviewer")
    workflow.add_edge("reviewer", "finalizer")
    workflow.add_edge("finalizer", END)

    return workflow.compile(checkpointer=checkpointer)


# --- 3. FastAPI Lifespan 및 설정 ---
DB_URI = "postgresql://postgres:changethis@localhost:5432/app"
pool = AsyncConnectionPool(conninfo=DB_URI, max_size=10, open=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Setup을 위한 임시 autocommit 커넥션 (인덱스 생성 에러 방지)
    async with await psycopg.AsyncConnection.connect(DB_URI, autocommit=True) as conn:
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()

    # 2. 실제 앱에서 사용할 커넥션 풀 기반 Saver
    # Pool을 사용해야 멀티태스킹 환경에서 효율적입니다.
    async with AsyncConnectionPool(conninfo=DB_URI, max_size=10) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        # 그래프 생성 및 state에 저장
        app.state.graph = create_app_graph(checkpointer)
        yield
        # yield 이후(앱 종료 시)에는 pool이 자동으로 닫힙니다.
    # 3. 실제 애플리케이션에서 사용할 Checkpointer (Pool 사용)
    # 실제 노드 실행 시에는 효율을 위해 커넥션 풀을 사용합니다.



app = FastAPI(lifespan=lifespan)


# --- 4. API 스키마 및 엔드포인트 ---
class WorkflowStartRequest(BaseModel):
    thread_id: str


class WorkflowResumeRequest(BaseModel):
    thread_id: str
    feedback: str


@app.post("/start")
async def start_workflow(request: WorkflowStartRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    initial_input = {"content": ""}

    # 실행 시작 (첫 번째 중단점인 reviewer 노드까지 진행)
    # astream 혹은 ainvoke 사용
    result = await app.state.graph.ainvoke(initial_input, config)

    # interrupt가 발생하면 해당 노드에서 멈춘 상태의 결과를 반환
    return {"status": "paused", "current_state": result}


@app.post("/resume")
async def resume_workflow(request: WorkflowResumeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    # 중단된 지점부터 다시 시작 (Command 객체로 resume 값 전달)
    # 이 값은 interrupt()의 반환값으로 들어감
    try:
        result = await app.state.graph.ainvoke(
            Command(resume=request.feedback),
            config
        )
        return {"status": "success", "final_result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/status/{thread_id}")
async def get_status(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = await app.state.graph.aget_state(config)
    return {"next_step": state.next, "values": state.values}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)