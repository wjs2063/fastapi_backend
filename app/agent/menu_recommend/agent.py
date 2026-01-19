from langgraph.graph import StateGraph, START, END
from app.agent.menu_recommend.state import State
from app.agent.menu_recommend.node import fetch_user_info

workflow = StateGraph(State)

workflow.add_node("fetch_user_info", fetch_user_info)

workflow.add_edge(START, "fetch_user_info")
workflow.add_edge("fetch_user_info", END)

graph = workflow.compile()
