from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Annotated, Literal
from operator import add

load_dotenv(find_dotenv())


class PostState(BaseModel):
    topic: str
    post: str | None = None
    evaluation: Literal["approved", "not_approved"] | None = None
    feedback: str | None = None
    iteration: int | None = 0
    max_iteration: int | None = None

    post_history: Annotated[list[str], add]
    feedback_history: Annotated[list[str], add]


class EvaluationResponse(BaseModel):
    evaluation: Literal["approved", "not_approved"] | None = None
    feedback: str


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
evaluation_llm = llm.with_structured_output(EvaluationResponse)


def generating_post(state: PostState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a social media post generator. Create engaging, clear, and natural posts based on the user's topic or instructions.

Adapt the tone to the requested audience and platform. Keep posts concise, interesting, and easy to read. Do not invent facts. Return only the final post unless the user asks for an explanation.
""",
            ),
            ("human", "generate me a post based on this topic {topic}"),
        ]
    ).format(topic=state.topic)

    result = llm.invoke(prompt).content
    return {
        "post": result,
        "post_history": [result],
    }


def evaluating_post(state: PostState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a  strict post evaluator. Evaluate the given post for quality, relevance, clarity, and appropriateness.

Provide concise feedback explaining what is good or what should be improved. Then decide whether the post is approved
at least dont approved it for the first time.""",
            ),
            ("human", "evaluate this post :\n {post}"),
        ]
    ).format(post=state.post)
    result = evaluation_llm.invoke(prompt)

    return {
        "feedback": result.feedback,
        "evaluation": result.evaluation,
    }


def optimizing_post(state: PostState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a post optimization assistant. Improve the given post based on the provided feedback.

Preserve the original message and intent while improving its clarity, engagement, tone, and overall quality. Apply the feedback carefully without adding unsupported information.

Return only the optimized post.
""",
            ),
            ("human", "topic:{topic}\n\n post :{post} \n\n feedback: {feedback}"),
        ]
    ).format(post=state.post, feedback=state.feedback, topic=state.topic)

    result = llm.invoke(prompt).content
    print(
        f"result :\n{result}\n\n\niteration number:{state.iteration + 1}\n\nfeedback used:{state.feedback}"
    )
    return {
        "post": result,
        "post_history": [result],
        "iteration": state.iteration + 1,
    }


def check_evaluation(state: PostState) -> Literal["end", "optimizing_post"]:
    if state.evaluation == "approved" or state.max_iteration == state.iteration:
        return "end"
    else:
        return "optimizing_post"


graph = StateGraph(PostState)

graph.add_node("generate_post", generating_post)
graph.add_node("evaluating_post", evaluating_post)
graph.add_node("optimizing_post", optimizing_post)

graph.add_edge(START, "generate_post")
graph.add_edge("generate_post", "evaluating_post")
graph.add_conditional_edges(
    "evaluating_post",
    check_evaluation,
    {
        "end": END,
        "optimizing_post": "optimizing_post",
    },
)
graph.add_edge("optimizing_post", "evaluating_post")


workflow = graph.compile()

res = workflow.invoke(
    {
        "topic": "The impact of AI on the future of software development.",
        "max_iteration": 5,
    }
)

print(res)
