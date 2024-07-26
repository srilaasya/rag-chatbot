from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from langchain_handler import initialize_langchain, process_user_message

app = FastAPI()

# Serve static files (like CSS) from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the request model


class UserInput(BaseModel):
    message: str


# Initialize LangChain and context
initialize_langchain()
conversation_context = []


@app.get("/", response_class=HTMLResponse)
async def get():
    with open("index.html") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/chat/")
async def chat(user_input: UserInput):
    global conversation_context

    # Process the user message and get the response
    try:
        response, conversation_context = process_user_message(
            user_input.message, conversation_context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
