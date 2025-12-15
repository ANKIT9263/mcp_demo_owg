"""
Basic API for MCP Agent
-----------------------
Run with:
    python api.py
"""

import asyncio
import json
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from client import run_client

app = FastAPI(title="MCP Agent API")


class QueryRequest(BaseModel):
    query: str


def format_sse(event: str, data: dict) -> str:
    """Format Server-Sent Event"""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def event_stream(query: str):
    """Generate SSE stream from agent execution"""
    queue = asyncio.Queue()

    async def stream_callback(event_type: str, data: dict):
        await queue.put((event_type, data))

    async def run_agent_task():
        try:
            await run_client(query, stream_callback)
        except Exception as e:
            await queue.put(("error", {"message": str(e)}))
        finally:
            await queue.put(("done", {}))

    # Start agent task
    asyncio.create_task(run_agent_task())

    # Stream events
    while True:
        event_type, data = await queue.get()
        yield format_sse(event_type, data)

        if event_type == "done":
            break


@app.post("/run_agent")
async def run_agent(request: QueryRequest):
    """
    Execute agent with streaming response

    Events:
    - plan: Execution plan
    - step: Tool execution started
    - step_result: Tool execution result
    - final: Final output
    - error: Error occurred
    - done: Stream complete
    """
    return StreamingResponse(
        event_stream(request.query),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
