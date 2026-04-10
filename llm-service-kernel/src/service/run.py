########## not using ##############

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import VLLM_BASE_URL, SERVED_MODEL_NAME
from .clients.vllm_client import VLLMClient

app = FastAPI()
client = VLLMClient(VLLM_BASE_URL)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()

    payload = {
        "model": body.get("model", SERVED_MODEL_NAME),
        "messages": body["messages"],
        "max_tokens": body.get("max_tokens", 128),
        "temperature": body.get("temperature", 0.7),
        "stream": bool(body.get("stream", False)),
    }

    if not payload["stream"]:
        data = client.chat(payload)
        return JSONResponse(data)

    def gen():
        for line in client.chat_stream(payload):
            yield line + "\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
