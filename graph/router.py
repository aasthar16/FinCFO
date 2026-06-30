"""
Router logic for LangGraph.
"""

from graph.state import GlobalState


def route_from_supervisor(state: GlobalState) -> str:
    """
    Conditional edge routing from supervisor to the appropriate agent.
    """
    next_action = state.get("next_action", "end")
    
    # If next_action is "end", return "end" to trigger END node
    if next_action == "end":
        return "end"
    
    # Map to node names
    routing_map = {
        "scenario": "scenario",
        "burn": "burn",
        "forecast": "forecast",
        "recommendation": "recommendation",
    }
    
    return routing_map.get(next_action, "end")


def after_spoke(state: GlobalState) -> str:
    """
    Spoke agents always route back to supervisor.
    """
    return "supervisor"