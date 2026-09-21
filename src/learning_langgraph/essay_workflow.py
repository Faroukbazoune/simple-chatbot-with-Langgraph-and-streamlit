from langgraph.graph import START, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, computed_field, Field
from typing import Annotated
from operator import add

load_dotenv(find_dotenv())


class EssayState(BaseModel):
    essay: str
    analysis_feedback: str | None = None
    thought_feedback: str | None = None
    language_feedback: str | None = None
    overall_feedback: str | None = None
    individual_scores: Annotated[list[int], add] = Field(default_factory=list)

    @computed_field
    @property
    def avg_score(self) -> float:
        return round(sum(self.individual_scores) / 3)


class evaluation_response(BaseModel):
    score: int = Field(description="score out of 10", le=10, ge=0)
    feedback: str


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
llm_structured = llm.with_structured_output(evaluation_response)


def analysis_node(state: EssayState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert essay and critical-thinking evaluator. Focus only on the quality, depth, logic, and strength of the essay's analysis and arguments. Be objective and constructive. Always provide a score out of 10 and concise feedback with specific improvements.
""",
            ),
            (
                "human",
                """Evaluate the essay's **analysis and arguments**. Focus on depth, logic, evidence, reasoning, and quality of arguments.

Give:

* **Score: /10**
* **Feedback:** strengths, weaknesses, and how to improve.

Essay:
{essay}
""",
            ),
        ]
    ).format_messages(essay=state.essay)

    res = llm_structured.invoke(prompt)

    return {
        "analysis_feedback": res.feedback,
        "individual_scores": [res.score],
    }


def language_node(state: EssayState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert English language evaluator. Focus only on grammar, vocabulary, spelling, sentence structure, and fluency. Be objective and constructive. Always provide a score out of 10 and concise feedback with specific improvements.

""",
            ),
            (
                "human",
                """Evaluate the essay's **language**. Focus on grammar, vocabulary, spelling, sentence structure, and fluency.

Give:

* **Score: /10**
* **Feedback:** strengths, mistakes, and how to improve.

Essay:
{essay}

""",
            ),
        ]
    ).format_messages(essay=state.essay)

    res = llm_structured.invoke(prompt)

    return {
        "language_feedback": res.feedback,
        "individual_scores": [res.score],
    }


def thought_node(state: EssayState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert evaluator of written communication. Focus only on the clarity, coherence, organization, and logical flow of the writer's thoughts. Be objective and constructive. Always provide a score out of 10 and concise feedback with specific improvements.

""",
            ),
            (
                "human",
                """Evaluate the **clarity and organization of the writer's thoughts**. Focus on coherence, logical flow, explanation of ideas, and overall readability.

Give:

* **Score: /10**
* **Feedback:** what is clear, what is confusing, and how to improve.

Essay:
{essay}

""",
            ),
        ]
    ).format_messages(essay=state.essay)

    res = llm_structured.invoke(prompt)

    return {
        "thought_feedback": res.feedback,
        "individual_scores": [res.score],
    }


def final_node(state: EssayState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the final essay evaluator. Use the three provided evaluations—language, analysis, and thoughts/clarity—to produce an overall assessment of the essay.

Provide only concise, objective, and constructive overall feedback. Summarize the essay's main strengths, weaknesses, and the most important areas for improvement. Do not provide a score.


""",
            ),
            (
                "human",
                """Based on the following evaluations, provide overall feedback for the essay.

LANGUAGE:
{language_feedback}

ANALYSIS:
{analysis_feedback}

THOUGHTS & CLARITY:
{thought_feedback}

OVERALL SCORE :{overall_score}

Give a concise overall assessment covering the main strengths, weaknesses, and most important improvements. Do not provide a score.

""",
            ),
        ]
    ).format_messages(
        language_feedback=state.language_feedback,
        analysis_feedback=state.analysis_feedback,
        thought_feedback=state.thought_feedback,
        overall_score=state.avg_score,
    )

    res = llm.invoke(prompt)

    return {
        "overall_feedback": res.feedback,
    }


graph = StateGraph(EssayState)

graph.add_node("analysis", analysis_node)
graph.add_node("thought", thought_node)
graph.add_node("language", language_node)
graph.add_node("final", final_node)


graph.add_edge(START, "analysis")
graph.add_edge(START, "thought")
graph.add_edge(START, "language")

graph.add_edge("analysis", "final")
graph.add_edge("thought", "final")
graph.add_edge("language", "final")

graph.add_edge("final", END)

workflow = graph.compile()

result = workflow.invoke(
    {
        "essay": """Technology has become an important part of our daily lives. We use it for communication, education, work, and entertainment. It has made many tasks faster and easier than they were in the past.

One of the biggest advantages of technology is that it gives people access to information. Students can use the internet to learn about almost any subject, while workers can communicate with colleagues from different parts of the world. Technology can also help businesses become more efficient and create new opportunities.

However, technology also has some disadvantages. People can become too dependent on their devices, and spending too much time online can reduce real-life social interaction. In addition, not all information on the internet is reliable, so people need to learn how to evaluate what they read.

In conclusion, technology has both benefits and disadvantages. If people use it responsibly, it can be a powerful tool for learning, communication, and improving everyday life.
""",
        "individual_scores": [],
    }
)

state = EssayState(**result)
print(state.model_dump())
