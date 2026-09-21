from dotenv import find_dotenv, load_dotenv
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel
from typing import TypedDict


class graphstate(TypedDict):
    employee_name: str
    monthly_salary: int
    working_days: int
    completed_projects: int

    yearly_salary: int
    bonus_amount: int
    project_status: str
    summary: str


graph = StateGraph(graphstate)


def calculate_yearly_salary(state: graphstate):
    return {"yearly_salary": state["monthly_salary"] * 12}


def calculate_bonus(state: graphstate):
    return {"bonus_amount": state["monthly_salary"] * 2}


def project_evaluation(state: graphstate):
    return {
        "project_status": (
            "excellent" if state["completed_projects"] > 5 else "average"
        )
    }


def summary(state: graphstate):
    return {
        "summary": (
            f"{state['employee_name']} has {state['completed_projects']}completed projects and {state['yearly_salary']} yearly_salary and bonus of {state['bonus_amount']} bonus"
        )
    }


graph.add_node("calculate_yearly_salary", calculate_yearly_salary)
graph.add_node("calculate_bonus", calculate_bonus)
graph.add_node("project_evaluation", project_evaluation)
graph.add_node("summary", summary)

graph.add_edge(START, "calculate_yearly_salary")
graph.add_edge(START, "calculate_bonus")
graph.add_edge(START, "project_evaluation")

graph.add_edge("calculate_yearly_salary", "summary")
graph.add_edge("calculate_bonus", "summary")
graph.add_edge("project_evaluation", "summary")

graph.add_edge("summary", END)


workflow = graph.compile()

res = workflow.invoke(
    {
        "employee_name": "farouk",
        "monthly_salary": 50000,
        "working_days": 10,
        "completed_projects": 10,
    }
)

print(res)
