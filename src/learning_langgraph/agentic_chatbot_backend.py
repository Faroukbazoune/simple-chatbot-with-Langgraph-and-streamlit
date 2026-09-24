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
from pprint import pprint
from langgraph.prebuilt import ToolNode, tools_condition
from learning_langgraph.tools import send_gmail_mesage, get_current_weather, search_tool

load_dotenv(find_dotenv())
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

tools = [send_gmail_mesage, get_current_weather, search_tool]

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash").bind(tools=tools)


class ChatState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


def chat(state: ChatState):
    response = llm.invoke(state.messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)
graph = StateGraph(ChatState)
graph.add_node("chat", chat)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")


chatbot = graph.compile(checkpointer=checkpointer)

# pprint(
#     chatbot.get_state(
#         config={"configurable": {"thread_id": "e2afacdb-b677-41ad-a94b-48ec4dbb9839"}}
#     ).values
# )


def gettig_all_threads():
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(threads)
