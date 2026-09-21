from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from typing import Annotated
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv(find_dotenv())
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


class ChatState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


def chat(state: ChatState):
    response = llm.invoke(state.messages)
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpointer)
