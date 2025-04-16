import os
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
import openai

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

# --- Function to Fetch OpenAI Models ---
def fetch_openai_models() -> List[Dict[str, Any]]:
    """
    Fetch the list of models from OpenAI.
    Uses OPENAI_API_KEY and OPENAI_API_BASE from environment variables.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
    if not openai.api_key:
        print("OPENAI_API_KEY not set.")
        return []
    try:
        models_response = openai.Model.list()
        return [
            {"id": model["id"], "object": "model", "owned_by": "openai"}
            for model in models_response["data"]
        ]
    except Exception as e:
        print("Error fetching OpenAI models:", e)
        return []

# --- Function to Fetch Grok (xAI) Models ---
def fetch_grok_models() -> List[Dict[str, Any]]:
    """
    Fetch the list of models from Grok (xAI) using their API.
    Ensure GROK_API_KEY and GROK_BASE_URL are set in environment variables.
    Expected GROK_BASE_URL: "https://api.x.ai/v1"
    """
    grok_api_key = os.getenv("GROK_API_KEY")
    grok_base_url = os.getenv("GROK_BASE_URL")
    if not grok_api_key or not grok_base_url:
        print("GROK_API_KEY or GROK_BASE_URL not set.")
        return []
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(f"{grok_base_url}/models", headers=headers)
        response.raise_for_status()
        models = response.json().get("data", [])
        return [
            {"id": model["id"], "object": "model", "owned_by": "grok"}
            for model in models
        ]
    except Exception as e:
        print("Error fetching Grok models:", e)
        return []

# --- Models Endpoint ---
@app.get("/v1/models")
async def get_models(api_key: str = Depends(get_api_key)):
    openai_models = fetch_openai_models()
    grok_models = fetch_grok_models()
    combined_models = openai_models + grok_models
    return {"object": "list", "data": combined_models}

# --- Data Models for Chat Completion ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

# --- Function to Call Grok Chat Completions ---
def grok_chat_completion(chat_request: ChatRequest) -> Dict[str, Any]:
    """
    Calls the Grok (xAI) chat completions endpoint.
    Uses GROK_API_KEY and GROK_BASE_URL from environment variables.
    """
    grok_api_key = os.getenv("GROK_API_KEY")
    grok_base_url = os.getenv("GROK_BASE_URL")
    if not grok_api_key or not grok_base_url:
        raise Exception("GROK_API_KEY or GROK_BASE_URL not configured")
    
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": chat_request.model,
        "messages": [{"role": m.role, "content": m.content} for m in chat_request.messages],
        "stream": False,
        "temperature": 0
    }
    
    try:
        response = requests.post(f"{grok_base_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Grok API request failed: {e}")

# --- Chat Completions Endpoint ---
@app.post("/v1/chat/completions")
async def create_chat_completion(chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    model = chat_request.model.strip().lower()

    # For OpenAI models:
    if model in ["gpt-4", "gpt-3.5-turbo"] or model.startswith("openai"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
        if not openai.api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        try:
            # Use the new asynchronous interface with ChatCompletion
            completion = await openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in chat_request.messages]
            )
            return completion
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # For Grok models:
    elif model.startswith("grok"):
        try:
            result = grok_chat_completion(chat_request)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported model")
