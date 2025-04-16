import os
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import httpx

# --- Configuration ---
DEFAULT_API_KEY = "vertiqalKey1"
API_KEY = os.getenv("API_KEY", DEFAULT_API_KEY)
API_KEY_NAME = "X-API-Key"

app = FastAPI()

# --- CORS Setup ---
origins = [
    "https://ralph.vertiqal.ai",  # Your OpenWebUI public domain
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key Dependency ---
async def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# --- Root Endpoint ---
@app.get("/")
async def root(api_key: str = Depends(get_api_key)):
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

# --- Data Models ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

# --- OpenAI API Functions Using httpx ---
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
    url = f"{openai_api_base}/chat/completions"
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
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{openai_api_base}/models", headers=headers)
    response.raise_for_status()
    models_response = response.json()
    return [
        {"id": model["id"], "object": "model", "owned_by": "openai"}
        for model in models_response.get("data", [])
    ]

# --- Grok API Functions Using httpx ---
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
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{grok_base_url}/models", headers=headers)
    response.raise_for_status()
    models = response.json().get("data", [])
    return [
        {"id": model["id"], "object": "model", "owned_by": "grok"}
        for model in models
    ]

# --- Models Endpoint ---
@app.get("/v1/models")
async def get_models(api_key: str = Depends(get_api_key)):
    openai_models = await fetch_openai_models()
    grok_models = await fetch_grok_models()
    combined_models = openai_models + grok_models
    return {"object": "list", "data": combined_models}

# --- Chat Completions Endpoint ---
@app.post("/v1/chat/completions")
async def create_chat_completion(chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    model = chat_request.model.strip().lower()

    if model in ["gpt-4", "gpt-3.5-turbo"] or model.startswith("openai"):
        try:
            result = await call_openai_chat_completion(chat_request)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    elif model.startswith("grok"):
        try:
            result = await call_grok_chat_completion(chat_request)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported model")
