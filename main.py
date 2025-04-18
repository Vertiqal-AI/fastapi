# ==========================================================
#  main.py – FastAPI proxy with streaming & auto‑vision switch
# ==========================================================
import os
from typing import List, Dict, Any, Union

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

OPENAI_BASE    = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
GROK_BASE      = os.getenv("GROK_BASE_URL")

# auto‑vision fallback
VISION_FALLBACK_MODEL = os.getenv("VISION_FALLBACK_MODEL", "gpt-4o")
VISION_OK = {
    "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-audio-preview",
    "gpt-4o-mini", "gpt-4o-mini-audio-preview",
    "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"
}

# ---------- FASTAPI & CORS ----------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ralph.vertiqal.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- BEARER → X‑API‑KEY ------------------------------------
@app.middleware("http")
async def accept_bearer_middleware(request: Request, call_next):
    if "authorization" in request.headers:
        auth = request.headers["authorization"]
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            request.headers.__dict__["_list"].append((b"x-api-key", token.encode()))
    return await call_next(request)

async def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(403, "Invalid API Key")
    return x_api_key

# ---------- DATA MODELS -------------------------------------------
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]   # vision or text

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool | None = False

# ---------- HELPERS ------------------------------------------------
def contains_image(msgs: List[Message]) -> bool:
    for m in msgs:
        if isinstance(m.content, list):
            if any(part.get("type") == "input_image" for part in m.content):
                return True
    return False

def supports_vision(model_id: str) -> bool:
    ml = model_id.lower()
    return ml in VISION_OK or ml.startswith("gpt-4o")

def messages_to_dicts(msgs: List[Message]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs:
        out.append({"role": m.role, "content": m.content})
    return out

# ---------- ROOT ---------------------------------------------------
@app.get("/")
async def root(api_key: str = Depends(get_api_key)):
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

# ---------- OPENAI single‑shot ------------------------------------
async def call_openai_chat_completion(cr: ChatRequest) -> Dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(500, "OPENAI_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": cr.model, "messages": messages_to_dicts(cr.messages), "stream": False}
    async with httpx.AsyncClient(timeout=None) as c:
        r = await c.post(f"{OPENAI_BASE}/v1/chat/completions", json=payload, headers=headers)
    r.raise_for_status()
    return r.json()

# ---------- OPENAI stream -----------------------------------------
async def openai_event_stream(cr: ChatRequest):
    key = os.getenv("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": cr.model, "messages": messages_to_dicts(cr.messages), "stream": True}
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", f"{OPENAI_BASE}/v1/chat/completions",
                            headers=headers, json=payload) as r:
            async for line in r.aiter_lines():
                if line:
                    yield f"{line}\n\n"
                if line.strip() == "data: [DONE]":
                    break

async def fetch_openai_models() -> List[Dict[str, Any]]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return []
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{OPENAI_BASE}/v1/models", headers=headers)
    r.raise_for_status()
    return [{"id": m["id"], "object": "model", "owned_by": "openai"} for m in r.json().get("data", [])]

# ---------- GROK single‑shot --------------------------------------
async def call_grok_chat_completion(cr: ChatRequest) -> Dict[str, Any]:
    key = os.getenv("GROK_API_KEY")
    if not key or not GROK_BASE:
        raise HTTPException(500, "GROK keys not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": cr.model, "messages": messages_to_dicts(cr.messages),
               "stream": False, "temperature": 0}
    async with httpx.AsyncClient(timeout=None) as c:
        r = await c.post(f"{GROK_BASE}/chat/completions", json=payload, headers=headers)
    r.raise_for_status()
    return r.json()

# ---------- GROK stream -------------------------------------------
async def grok_event_stream(cr: ChatRequest):
    key = os.getenv("GROK_API_KEY")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": cr.model, "messages": messages_to_dicts(cr.messages),
               "stream": True, "temperature": 0}
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", f"{GROK_BASE}/chat/completions",
                            headers=headers, json=payload) as r:
            async for line in r.aiter_lines():
                if line:
                    yield f"{line}\n\n"
                if line.strip() == "data: [DONE]":
                    break

async def fetch_grok_models() -> List[Dict[str, Any]]:
    key = os.getenv("GROK_API_KEY")
    if not key or not GROK_BASE:
        return []
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{GROK_BASE}/models", headers=headers)
    r.raise_for_status()
    return [{"id": m["id"], "object": "model", "owned_by": "grok"} for m in r.json().get("data", [])]

# ---------- MODELS endpoint ---------------------------------------
@app.get("/v1/models")
async def get_models(api_key: str = Depends(get_api_key)):
    return {"object": "list",
            "data": (await fetch_openai_models()) + (await fetch_grok_models())}

@app.get("/models")
async def get_models_alias(api_key: str = Depends(get_api_key)):
    return await get_models(api_key)

# ---------- CHAT endpoint -----------------------------------------
@app.post("/v1/chat/completions")
async def create_chat_completion(cr: ChatRequest,
                                 request: Request,
                                 api_key: str = Depends(get_api_key)):

    has_img     = contains_image(cr.messages)
    model_lower = cr.model.lower()

    # ---- auto‑switch to vision model if needed --------------------
    if has_img and not supports_vision(model_lower):
        cr.model = VISION_FALLBACK_MODEL
        model_lower = cr.model.lower()

    # ---- GROK (text‑only) ----------------------------------------
    if model_lower.startswith("grok"):
        if has_img:
            raise HTTPException(400, "Grok models do not support image input.")
        if cr.stream:
            return StreamingResponse(grok_event_stream(cr), media_type="text/event-stream")
        return await call_grok_chat_completion(cr)

    # ---- OPENAI ---------------------------------------------------
    if cr.stream:
        return StreamingResponse(openai_event_stream(cr), media_type="text/event-stream")
    return await call_openai_chat_completion(cr)

# legacy alias
@app.post("/chat/completions")
async def create_chat_completion_alias(cr: ChatRequest,
                                       api_key: str = Depends(get_api_key)):
    return await create_chat_completion(cr, api_key)
