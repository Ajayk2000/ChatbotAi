from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os
from dotenv import load_dotenv


load_dotenv()

#FastAPI Setup 
app = FastAPI()

# Allow React frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  LangChain Setup
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY"),  
    model="deepseek/deepseek-chat-v3.1:free"
)


# state type
class ChatState(TypedDict):
    messages: list

# graph builder
graph = StateGraph(ChatState)

# node: call llm
def call_model(state: ChatState):
    result = llm.invoke(state["messages"])
    state["messages"].append({"role": "assistant", "content": result.content})
    return state

# add node + edges
graph.add_node("model", call_model)
graph.set_entry_point("model")
graph.add_edge("model", END)

# compile graph
chat_app = graph.compile()


state = {"messages": [{"role": "system", "content": "You are a helpful AI assistant."}]}

#  FastAPI Endpoints
class Message(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(msg: Message):
    global state
    user_message = msg.message

    # add user message to state
    state["messages"].append({"role": "user", "content": user_message})

    # invoke chatbot
    state = chat_app.invoke(state)

    # get latest assistant reply
    bot_reply = state["messages"][-1]["content"]

    return {"reply": bot_reply}
