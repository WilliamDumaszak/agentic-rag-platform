"""
LangGraph StateGraph definition for the Agentic RAG pipeline.

Graph:
  START → decision → retrieve → generate → rank → END
                   ↘ web_search → END
                   ↘ generate → rank → END
"""

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    decision_node,
    generate_node,
    rank_node,
    retrieve_node,
    web_search_node,
)
from agent.state import AgentState


def _route(state: AgentState) -> str:
    return state.next_step


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("decision", decision_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("rank", rank_node)

    graph.add_edge(START, "decision")

    graph.add_conditional_edges(
        "decision",
        _route,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "generate": "generate",
        },
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "rank")
    graph.add_edge("rank", END)
    graph.add_edge("web_search", END)

    return graph.compile()


# module-level compiled graph (imported by API and tests)
agent_graph = build_graph()
