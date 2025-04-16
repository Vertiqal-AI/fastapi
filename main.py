# main.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

# New endpoint for testing communication from OpenWebUI
class Prompt(BaseModel):
    text: str

@app.post("/ask")
async def ask(prompt: Prompt):
    response_text = f"Ralph received: {prompt.text}"
    return {"response": response_text}
