import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph

app = FastAPI()


async def event_generator(input_data: dict):
    """
    LangGraph의 이벤트를 캡처하여 SSE 포맷으로 변환하는 제너레이터
    """
    # graph는 미리 컴파일된 CompiledGraph 객체라고 가정합니다.
    async for event in graph.astream_events(input_data, version="v2"):
        kind = event["event"]

        # 1. 특정 노드 실행 시작 시 상태 전송
        if kind == "on_chain_start" and event["name"] == "LangGraph":
            yield f"data: {json.dumps({'status': 'started', 'node': 'root'})}\n\n"

        elif kind == "on_chain_start":
            # 노드 이름(node name)을 추출하여 현재 어떤 단계인지 UI에 알림
            node_name = event.get("metadata", {}).get("langgraph_node", "unknown")
            if node_name != "unknown":
                yield f"data: {json.dumps({'status': 'executing', 'node': node_name})}\n\n"

        # 2. Tool 실행 상태 추적
        elif kind == "on_tool_start":
            yield f"data: {json.dumps({'status': 'tool_calling', 'tool': event['name']})}\n\n"

        # 3. 최종 결과 또는 중간 출력 스트리밍
        elif kind == "on_chat_model_stream":
            content = event["data"].get("chunk").content
            if content:
                yield f"data: {json.dumps({'status': 'streaming', 'content': content})}\n\n"

        # 예외 처리: 에러 발생 시 UI에 즉시 전달
        elif kind == "on_chain_error":
            yield f"data: {json.dumps({'status': 'error', 'message': str(event['data'])})}\n\n"

    yield "data: {\"status\": \"completed\"}\n\n"


@app.get("/stream")
async def stream_agent_status(query: str):
    return StreamingResponse(
        event_generator({"messages": [("user", query)]}),
        media_type="text/event-stream"
    )