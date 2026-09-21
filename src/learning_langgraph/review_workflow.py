from langgraph.graph import StateGraph, START, END
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, Optional

load_dotenv(find_dotenv())


class sentiment(BaseModel):
    sentiment: Literal["positive", "negative"]


class diagnosishema(BaseModel):
    issue_type: Literal["UX", "Performance", "other", "bug", "support"]
    tone: Literal["frustrated", "Harsh", "Direct", "Professional", "angry", "calm"]
    urgency: Literal["low", "medium", "high"]


class review(BaseModel):
    review: str
    sentiment: Literal["negative", "positive"] | None = None
    diagnosis: diagnosishema | None = None
    reply: str = None


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
diagnois_llm = llm.with_structured_output(diagnosishema)
sentiment_llm = llm.with_structured_output(sentiment)


def find_sent(state: review):
    result = sentiment_llm.invoke(f"find the sentiment on this review : {state.review}")

    return {"sentiment": result.sentiment}


def find_diagnosis(state: review):
    result = diagnois_llm.invoke(f"diagnosis this review:{state.review}")

    return {"diagnosis": result}


def positive_response(state: review):
    result = llm.invoke(f"write a warm thank you reply to this review: {state.review}")
    return {"reply": result.content}


def negative_response(state: review):
    result = llm.invoke(
        f""" you are a support assistant , the user had  a {state.diagnosis.issue_type} issue , sounded {state.diagnosis.tone} and marked urgency as {state.diagnosis.urgency} , write an empathetic helpful resolution  """
    )

    return {"reply": result.content}


def check_sent(state: review) -> Literal["find_diagnosis", "positive_response"]:
    return "find_diagnosis" if state.sentiment == "negative" else "positive_response"


graph = StateGraph(review)

graph.add_node("find_sentiment", find_sent)
graph.add_node("find_diagnosis", find_diagnosis)
graph.add_node("negative_response", negative_response)
graph.add_node("positive_response", positive_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sent)
graph.add_edge("positive_response", END)
graph.add_edge("find_diagnosis", "negative_response")
graph.add_edge("negative_response", END)


workflow = graph.compile()

result = workflow.invoke(
    {
        "review": """The user experience is frustrating and poorly designed. The interface feels confusing, with unclear navigation and too many unnecessary steps to complete simple tasks. Important features are difficult to find, and the layout does not feel intuitive. Overall, the UX needs significant improvement to make the product easier, faster
"""
    }
)
result["diagnosis"] = result["diagnosis"].model_dump()
print(result)
