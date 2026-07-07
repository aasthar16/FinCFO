"""
LangGraph graph builder - Updated with V2 Supervisor.
"""

from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GlobalState
from graph.nodes.parser import parser_node, route_after_parser
from graph.nodes.supervisor_v2 import supervisor_v2_node, route_from_supervisor_v2
from graph.nodes.scenario import scenario_node
from graph.nodes.burn_expense import burn_expense_node
from graph.nodes.forecast import forecast_node
from graph.nodes.recommendation import recommendation_node
# from graph.config.database import postgres_config




def build_ai_cfo_graph(checkpointer: Optional[MemorySaver] = None, use_v2: bool = True):

    """
    Build the LangGraph workflow.
    
    Args:
        checkpointer: Optional memory saver for persistence
        use_v2: If True, use LLM-powered supervisor (V2). 
                If False, use original regex supervisor (V1).
    """
    builder = StateGraph(GlobalState)
    
    if use_v2:
        # === V2: Agentic Architecture ===
        # Single supervisor node that handles everything via tools
        builder.add_node("parser", parser_node)
        builder.add_node("supervisor", supervisor_v2_node)
        
       
        
        builder.set_entry_point("parser")

        builder.add_conditional_edges(
            "parser",
            route_after_parser,
            {
                "supervisor": "supervisor",
                "end": END,
            }
        )
        
        # Supervisor → end
        builder.add_conditional_edges(
            "supervisor",
            route_from_supervisor_v2,
            {
                "end": END,
            }
        )
    else:
        # === V1: Hub-and-Spoke Architecture (original) ===
        from graph.nodes.supervisor import supervisor_node
        from graph.router import route_from_supervisor
        
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("scenario", scenario_node)
        builder.add_node("burn", burn_expense_node)
        builder.add_node("forecast", forecast_node)
        builder.add_node("recommendation", recommendation_node)
        
        builder.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {
                "scenario": "scenario",
                "burn": "burn",
                "forecast": "forecast",
                "recommendation": "recommendation",
                "end": END,
            }
        )
        
        builder.add_edge("scenario", "supervisor")
        builder.add_edge("burn", "supervisor")
        builder.add_edge("forecast", "supervisor")
        builder.add_edge("recommendation", "supervisor")
        
        builder.set_entry_point("supervisor")
    
    # Compile with checkpointer if provided
    if checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
    else:
        graph = builder.compile()
    
    return graph