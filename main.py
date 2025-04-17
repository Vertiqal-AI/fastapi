# ==========================================================
#  main.py  – FastAPI proxy with streaming (SSE) for GPT & Grok
# ==========================================================
import os, asyncio
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.requests import Request

# ---------- CONFIG -------------------------------------------------
DEFAULT_API_KEY = "vertiqalKey1"
API_KEY        = os.getenv("API_KEY", DEFAULT_API_KEY)
API_KEY_NAME   = "X-API-Key"

# ---------- FASTAPI APP & CORS -------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ralph.vertiqal.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- BEARER → X‑API‑KEY MIDDLEWARE --------------------------
@app.middleware("http")
async def accept_bearer_middleware(request: Request, call_next):
    if "authorization" in request.headers:
        auth = request.headers["authorization"]
        if auth.startswith("Bearer "):
            token = auth[len("Bearer "):].strip()
            request.headers.__dict__["_list"].append((b"x-api-key", token.encode()))
    return await call_next(request)

async def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(403, "Invalid API Key")
    return x_api_key

# ---------- DATA MODELS -------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool | None = False            # enables live tokens

# ---------- PING ---------------------------------------------------
@app.get("/")
async def root(api_key: str = Depends(get_api_key)):
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

# ---------- OPENAI (single‑shot) -----------------------------------
async def call_openai_chat_completion(chat_request: ChatRequest) -> Dict[str, Any]:
    key  = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
    if not key:
        raise HTTPException(500, "OPENAI_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model":    chat_request.model,
        "messages": [m.dict() for m in chat_request.messages],
        "stream":   False,
    }
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{base}/v1/chat/completions", json=payload, headers=headers)
    r.raise_for_status()
    return r.json()

# ---------- OPENAI STREAM ------------------------------------------
async def openai_event_stream(chat_request: ChatRequest):
    key  = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model":    chat_request.model,
        "messages": [m.dict() for m in chat_request.messages],
        "stream":   True,
    }
    url = f"{base}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", url, headers=headers, json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                yield f"{line}\n\n"
                if line.strip() == "data: [DONE]":
                    break

async def fetch_openai_models() -> List[Dict[str, Any]]:
    key  = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
    if not key:
        return []
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{base}/v1/models", headers=headers)
    r.raise_for_status()
    return [{"id": m["id"], "object": "model", "owned_by": "openai"} for m in r.json().get("data", [])]

# ---------- GROK (single‑shot) -------------------------------------
async def call_grok_chat_completion(chat_request: ChatRequest) -> Dict[str, Any]:
    key  = os.getenv("GROK_API_KEY")
    base = os.getenv("GROK_BASE_URL")
    if not key or not base:
        raise HTTPException(500, "GROK keys not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model":       chat_request.model,
        "messages":    [m.dict() for m in chat_request.messages],
        "stream":      False,
        "temperature": 0,
    }
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{base}/chat/completions", json=payload, headers=headers)
    r.raise_for_status()
    return r.json()

# ---------- GROK STREAM --------------------------------------------
async def grok_event_stream(chat_request: ChatRequest):
    key  = os.getenv("GROK_API_KEY")
    base = os.getenv("GROK_BASE_URL")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model":       chat_request.model,
        "messages":    [m.dict() for m in chat_request.messages],
        "stream":      True,
        "temperature": 0,
    }
    url = f"{base}/chat/completions"
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", url, headers=headers, json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                yield f"{line}\n\n"
                if line.strip() == "data: [DONE]":
                    break

async def fetch_grok_models() -> List[Dict[str, Any]]:
    key  = os.getenv("GROK_API_KEY")
    base = os.getenv("GROK_BASE_URL")
    if not key or not base:
        return []
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{base}/models", headers=headers)
    r.raise_for_status()
    return [{"id": m["id"], "object": "model", "owned_by": "grok"} for m in r.json().get("data", [])]

# ---------- MODELS ENDPOINT ----------------------------------------
@app.get("/v1/models")
async def get_models(api_key: str = Depends(get_api_key)):
    return {
        "object": "list",
        "data": (await fetch_openai_models()) + (await fetch_grok_models())
    }

@app.get("/models")
async def get_models_alias(api_key: str = Depends(get_api_key)):
    return await get_models(api_key)

# ---------- CHAT COMPLETIONS (with streaming) ----------------------
@app.post("/v1/chat/completions")
async def create_chat_completion(chat_request: ChatRequest,
                                 api_key: str = Depends(get_api_key)):
    model_lower = chat_request.model.lower()

    # -------- GROK --------
    if model_lower.startswith("grok"):
        if chat_request.stream:
            return StreamingResponse(grok_event_stream(chat_request),
                                     media_type="text/event-stream")
        return await call_grok_chat_completion(chat_request)

    # -------- OPENAI -------
    if chat_request.stream:
        return StreamingResponse(openai_event_stream(chat_request),
                                 media_type="text/event-stream")
    return await call_openai_chat_completion(chat_request)

# legacy alias
@app.post("/chat/completions")
async def create_chat_completion_alias(chat_request: ChatRequest,
                                       api_key: str = Depends(get_api_key)):
    return await create_chat_completion(chat_request, api_key)
