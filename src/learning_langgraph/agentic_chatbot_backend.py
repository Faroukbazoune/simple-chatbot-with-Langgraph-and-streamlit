from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from typing import Annotated
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pprint import pprint
from langgraph.prebuilt import ToolNode, tools_condition
from learning_langgraph.tools import (
    send_gmail_mesage,
    get_current_weather,
    search_tool,
    rag_tool,
)

load_dotenv(find_dotenv())
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


tools = [send_gmail_mesage, get_current_weather, search_tool, rag_tool]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
llm_withtools = llm.bind(tools=tools)


class ChatState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


def chat(state: ChatState):
    system_prompt = SystemMessage(
        content="""You are a helpful AI assistant with access to these tools:

* `send_gmail_mesage`: send emails when explicitly requested.
* `get_current_weather`: get current weather.
* `search_tool`: search the web for current or external information.
* `rag_tool`: search the connected knowledge base/documents.

Use the appropriate tool when needed. Do not guess when a tool can provide the answer. For multiple-step requests, use tools in the necessary order. Never claim to have performed an action unless you actually did.
"""
    )
    messages = [system_prompt, *state.messages]
    response = llm_withtools.invoke(messages)
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
