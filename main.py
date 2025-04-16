from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Configure CORS to allow requests from your OpenWebUI domain
origins = [
    "https://ralph.vertiqal.ai",  # your UI's domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

class Prompt(BaseModel):
    text: str

@app.post("/ask")
async def ask(prompt: Prompt):
    response_text = f"Ralph received: {prompt.text}"
    return {"response": response_text}
