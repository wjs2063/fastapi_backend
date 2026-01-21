from typing import Annotated, Optional
from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel

# LangGraph InMemory 관련 임포트
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt, Command


# 1. 상태 정의
class GraphState(MessagesState):
    user_id: str
    task_description: str = ""


# 2. 노드 정의
async def assistant_node(state: GraphState, *, store):
    user_id = state["user_id"]

    # [장기 기억 조회] InMemoryStore에서 유저 프로필 검색
    namespace = (user_id, "profile")
    memories = store.search(namespace)  # InMemory는 동기식으로도 동작함
    user_context = memories[0].value.get("info") if memories else "신규 유저"

    last_message = state["messages"][-1].content
    response = f"[{user_context}]님, 요청하신 '{last_message}' 작업을 시작할까요?"

    # [단기 기억/인터럽트] 여기서 실행 중지 및 메모리에 상태 저장
    feedback = interrupt(response)

    return {
        "messages": [{"role": "assistant", "content": response}],
        "task_description": f"유저 피드백: {feedback}"
    }


# 3. 그래프 구성 및 컴파일
# 전역 변수로 관리 (InMemory는 DB 연결이 필요 없으므로 단순하게 구성 가능)
memory_checkpointer = InMemorySaver()
long_term_store = InMemoryStore()

builder = StateGraph(GraphState)
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

# 체크포인터와 스토어를 함께 등록
app_graph = builder.compile(checkpointer=memory_checkpointer, store=long_term_store)

# 4. FastAPI 설정
app = FastAPI()


# --- API 모델 ---
class StartRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    feedback: str


# --- API 엔드포인트 ---

@app.post("/workflow/start")
async def start_workflow(req: StartRequest):
    """새로운 쓰레드에서 대화 시작"""
    config = {"configurable": {"thread_id": req.thread_id}}
    initial_input = {
        "messages": [{"role": "user", "content": req.message}],
        "user_id": req.user_id
    }

    # 첫 번째 interrupt 지점까지 실행
    result = await app_graph.ainvoke(initial_input, config)
    return {"status": "paused", "output": result["messages"][-1].content}


@app.post("/workflow/resume")
async def resume_workflow(req: ResumeRequest):
    """중단된 쓰레드를 재개"""
    config = {"configurable": {"thread_id": req.thread_id}}

    # 유저의 입력을 Command로 전달하여 다시 실행
    result = await app_graph.ainvoke(Command(resume=req.feedback), config)
    return {"status": "completed", "final_state": result}


@app.post("/memory/long-term")
async def add_long_term_memory(user_id: str, info: str):
    """쓰레드와 무관한 영구(세션 내) 정보 저장"""
    namespace = (user_id, "profile")
    long_term_store.put(namespace, "user_info", {"info": info})
    return {"message": "Long-term memory added to RAM"}


@app.get("/workflow/inspect/{thread_id}")
async def inspect_thread(thread_id: str):
    """특정 쓰레드의 단기 기억(체크포인트) 들여다보기"""
    config = {"configurable": {"thread_id": thread_id}}
    state = await app_graph.aget_state(config)
    return {"values": state.values, "next_node": state.next}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)