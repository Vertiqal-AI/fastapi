import os
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import httpx
from starlette.requests import Request

# --- Configuration ---
DEFAULT_API_KEY = "vertiqalKey1"  # fallback if 'API_KEY' not in env
API_KEY = os.getenv("API_KEY", DEFAULT_API_KEY)
API_KEY_NAME = "X-API-Key"

app = FastAPI()

# --- CORS Setup (adjust domain as needed) ---
origins = [
    "https://ralph.vertiqal.ai",  
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MIDDLEWARE to accept Bearer token & treat as X-API-Key ---
@app.middleware("http")
async def accept_bearer_middleware(request: Request, call_next):
    """If 'Authorization: Bearer <token>', treat it like 'X-API-Key: <token>'."""
    if "authorization" in request.headers:
        auth = request.headers["authorization"]
        if auth.startswith("Bearer "):
            token = auth[len("Bearer "):].strip()
            # Dynamically inject X-API-Key
            request.headers.__dict__["_list"].append((b"x-api-key", token.encode()))
    return await call_next(request)

# --- API Key Dependency ---
async def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# ==================
#    ROOT ENDPOINT
# ==================
@app.get("/")
async def root(api_key: str = Depends(get_api_key)):
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

# ==================
#    DATA MODELS
# ==================
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

# ==================
#   OPENAI CALLS
# ==================
async def call_openai_chat_completion(chat_request: ChatRequest) -> Dict[str, Any]:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = {
        "model": chat_request.model,
        "messages": [{"role": m.role, "content": m.content} for m in chat_request.messages],
        "stream": False
    }
    url = f"{openai_api_base}/v1/chat/completions"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

async def fetch_openai_models() -> List[Dict[str, Any]]:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
    if not openai_api_key:
        print("OPENAI_API_KEY not set.")
        return []
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    url = f"{openai_api_base}/v1/models"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    response.raise_for_status()
    data = response.json().get("data", [])
    return [
        {"id": m["id"], "object": "model", "owned_by": "openai"}
        for m in data
    ]

# ==================
#    GROK CALLS
# ==================
async def call_grok_chat_completion(chat_request: ChatRequest) -> Dict[str, Any]:
    grok_api_key = os.getenv("GROK_API_KEY")
    grok_base_url = os.getenv("GROK_BASE_URL")
    if not grok_api_key or not grok_base_url:
        raise HTTPException(status_code=500, detail="GROK_API_KEY or GROK_BASE_URL not configured")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grok_api_key}"
    }
    payload = {
        "model": chat_request.model,
        "messages": [{"role": m.role, "content": m.content} for m in chat_request.messages],
        "stream": False,
        "temperature": 0
    }
    url = f"{grok_base_url}/chat/completions"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

async def fetch_grok_models() -> List[Dict[str, Any]]:
    grok_api_key = os.getenv("GROK_API_KEY")
    grok_base_url = os.getenv("GROK_BASE_URL")
    if not grok_api_key or not grok_base_url:
        print("GROK_API_KEY or GROK_BASE_URL not set.")
        return []
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grok_api_key}"
    }
    url = f"{grok_base_url}/models"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    response.raise_for_status()
    data = response.json().get("data", [])
    return [
        {"id": m["id"], "object": "model", "owned_by": "grok"}
        for m in data
    ]

# ==================
#   MODELS ENDPOINT
# ==================
@app.get("/v1/models")
async def get_models(api_key: str = Depends(get_api_key)):
    openai_models = await fetch_openai_models()
    grok_models = await fetch_grok_models()
    return {"object": "list", "data": openai_models + grok_models}

@app.get("/models")
async def get_models_alias(api_key: str = Depends(get_api_key)):
    return await get_models(api_key)

# ==================
#  CHAT COMPLETIONS
# ==================
@app.post("/v1/chat/completions")
async def create_chat_completion(chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    # If model starts with "grok", route to Grok, otherwise route to OpenAI
    model = chat_request.model.strip().lower()
    if model.startswith("grok"):
        try:
            return await call_grok_chat_completion(chat_request)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Default everything else to OpenAI
        try:
            return await call_openai_chat_completion(chat_request)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/completions")
async def create_chat_completion_alias(chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    return await create_chat_completion(chat_request, api_key)
