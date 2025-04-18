# ==========================================================
#  main.py – FastAPI proxy (GPT‑4o vision + DALL·E + Grok)
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
API_KEY = os.getenv("API_KEY", DEFAULT_API_KEY)

OPENAI_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
GROK_BASE   = os.getenv("GROK_BASE_URL")

VISION_FALLBACK_MODEL = "gpt-4o"          # always use gpt‑4o for vision
VISION_OK = {"gpt-4o"}                    # extend later if you add more IDs

# ---------- FASTAPI & CORS -----------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ralph.vertiqal.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- BEARER -> X‑API‑KEY ------------------------------------
@app.middleware("http")
async def accept_bearer(request: Request, call_next):
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

# ---------- DATA MODELS -------------------------------------------
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool | None = False

class ImageGenerationRequest(BaseModel):
    model: str = "dall-e-3"
    prompt: str
    n: int | None = 1
    size: str | None = "1024x1024"
    style: str | None = None
    quality: str | None = None
    response_format: str | None = "url"

# ---------- HELPERS ------------------------------------------------
def contains_image(msgs: List[Message]) -> bool:
    for m in msgs:
        if isinstance(m.content, list):
            if any(p.get("type") == "input_image" for p in m.content):
                return True
    return False

def supports_vision(model: str) -> bool:
    ml = model.lower()
    return ml in VISION_OK or ml.startswith("gpt-4o")

def to_dicts(msgs: List[Message]) -> List[Dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in msgs]

# ---------- ROOT ---------------------------------------------------
@app.get("/")
async def root(_: str = Depends(get_api_key)):
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

# ---------- OPENAI CHAT (sync & stream) ---------------------------
async def openai_chat(cr: ChatRequest, stream: bool):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(500, "OPENAI_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": cr.model, "messages": to_dicts(cr.messages), "stream": stream}
    url = f"{OPENAI_BASE}/v1/chat/completions"

    if not stream:
        async with httpx.AsyncClient() as c:
            r = await c.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", url, headers=headers, json=payload) as r:
                async for line in r.aiter_lines():
                    if line:
                        yield f"{line}\n\n"
                    if line.strip() == "data: [DONE]":
                        break
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------- GROK CHAT (sync & stream) ------------------------------
async def grok_chat(cr: ChatRequest, stream: bool):
    key = os.getenv("GROK_API_KEY")
    if not key or not GROK_BASE:
        raise HTTPException(500, "GROK keys not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": cr.model, "messages": to_dicts(cr.messages),
               "stream": stream, "temperature": 0}
    url = f"{GROK_BASE}/chat/completions"

    if not stream:
        async with httpx.AsyncClient() as c:
            r = await c.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", url, headers=headers, json=payload) as r:
                async for line in r.aiter_lines():
                    if line:
                        yield f"{line}\n\n"
                    if line.strip() == "data: [DONE]":
                        break
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------- IMAGE GENERATION (OpenAI DALL·E) ------------------------
@app.post("/v1/images/generations")
async def image_generation(req: ImageGenerationRequest,
                           _: str = Depends(get_api_key)):

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(500, "OPENAI_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=None) as c:
        r = await c.post(f"{OPENAI_BASE}/v1/images/generations",
                         json=req.dict(exclude_none=True),
                         headers=headers)
    r.raise_for_status()
    return r.json()

# ---------- MODEL LISTS -------------------------------------------
async def openai_models() -> List[Dict[str, Any]]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return []
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{OPENAI_BASE}/v1/models", headers=headers)
    r.raise_for_status()
    return [{"id": m["id"], "object": "model", "owned_by": "openai"}
            for m in r.json().get("data", [])]

async def grok_models() -> List[Dict[str, Any]]:
    key = os.getenv("GROK_API_KEY")
    if not key or not GROK_BASE:
        return []
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{GROK_BASE}/models", headers=headers)
    r.raise_for_status()
    return [{"id": m["id"], "object": "model", "owned_by": "grok"}
            for m in r.json().get("data", [])]

@app.get("/v1/models")
async def list_models(_: str = Depends(get_api_key)):
    return {"object": "list",
            "data": (await openai_models()) + (await grok_models())}

@app.get("/models")
async def list_models_alias(_: str = Depends(get_api_key)):
    return await list_models()

# ---------- CHAT COMPLETIONS ENDPOINT ------------------------------
@app.post("/v1/chat/completions")
async def chat(cr: ChatRequest,
               _: str = Depends(get_api_key),
               request: Request = None):

    has_img = contains_image(cr.messages)
    model_l = cr.model.lower()

    # ----- auto‑switch to gpt‑4o for any vision prompt -------------
    if has_img and not supports_vision(model_l):
        cr.model = VISION_FALLBACK_MODEL
        model_l  = cr.model

    # ----- GROK (text‑only) ---------------------------------------
    if model_l.startswith("grok"):
        if has_img:
            raise HTTPException(400, "Grok models do not support image input.")
        return await grok_chat(cr, cr.stream or False)

    # ----- OPENAI --------------------------------------------------
    return await openai_chat(cr, cr.stream or False)

# ---------- legacy alias ------------------------------------------
@app.post("/chat/completions")
async def chat_alias(cr: ChatRequest, api_key: str = Depends(get_api_key)):
    return await chat(cr, api_key)
