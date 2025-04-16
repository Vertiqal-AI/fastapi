import os
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import openai
import requests

# --- Configuration ---
# Default API Key for authentication (can be overridden with environment variable "API_KEY")
DEFAULT_API_KEY = "vertiqalKey1"
API_KEY = os.getenv("API_KEY", DEFAULT_API_KEY)
API_KEY_NAME = "X-API-Key"

app = FastAPI()

# --- CORS Configuration ---
# Allow requests from your OpenWebUI public domain.
origins = [
    "https://ralph.vertiqal.ai",  # Adjust if needed.
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

# --- Model Listing Functions ---
def fetch_openai_models() -> List[Dict[str, Any]]:
    """
    Fetch models from OpenAI using the provided API key and base URL.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # Use the configurable OpenAI base URL if set.
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

def fetch_grok_models() -> List[Dict[str, Any]]:
    """
    Fetch Grok (xAI) models using the provided API key and base URL.
    """
    grok_api_key = os.getenv("GROK_API_KEY")
    grok_base_url = os.getenv("GROK_BASE_URL")  # e.g., "https://api.grok.xai/v1"
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

@app.get("/v1/models")
async def get_models(api_key: str = Depends(get_api_key)):
    """
    Expose endpoint that returns a combined list of models from OpenAI and Grok.
    """
    openai_models = fetch_openai_models()
    grok_models = fetch_grok_models()
    combined_models = openai_models + grok_models
    return {"object": "list", "data": combined_models}

# --- Chat Completion Endpoint and Data Models ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

def grok_chat_completion(chat_request: ChatRequest) -> Dict[str, Any]:
    """
    Call the Grok chat completions endpoint.
    Adjust this function as per Grok's real API documentation.
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
        "messages": [{"role": m.role, "content": m.content} for m in chat_request.messages]
    }
    
    try:
        response = requests.post(f"{grok_base_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Grok API request failed: {e}")

@app.post("/v1/chat/completions")
async def create_chat_completion(chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    """
    Accepts a chat request and routes to OpenAI or Grok depending on the model selected.
    """
    model = chat_request.model.strip().lower()

    # For OpenAI models:
    if model in ["gpt-4", "gpt-3.5-turbo"] or model.startswith("openai"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
        if not openai.api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in chat_request.messages]
            )
            return completion
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # For Grok models (e.g., model IDs starting with "grok"):
    elif model.startswith("grok"):
        try:
            result = grok_chat_completion(chat_request)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported model")
