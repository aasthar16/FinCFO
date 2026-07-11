"""
LangGraph graph builder - Agentic Architecture.
"""

from typing import Optional
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from graph.state import GlobalState
from graph.nodes.parser import parser_node, route_after_parser
from graph.nodes.supervisor_agent import supervisor_agent_node, route_after_agent, tool_node
from graph.nodes.burn_calculator import burn_calculator_node
# from graph.nodes.tools import tool_node

def build_ai_cfo_graph(checkpointer: Optional[PostgresSaver] = None):
    """
    Agentic architecture:
        parser → agent → burn/tools/end
                    ↑       ↓
                    └───────┘ (loop back)
    """
    builder = StateGraph(GlobalState)
    
    builder.add_node("parser", parser_node)
    builder.add_node("agent", supervisor_agent_node)
    builder.add_node("burn", burn_calculator_node)
    builder.add_node("tools", tool_node)
    
    builder.set_entry_point("parser")
    
    builder.add_edge("parser", "agent")
    
    builder.add_conditional_edges("agent", route_after_agent, {
        "burn": "burn",
        "tools": "tools",
        "end": END,
    })
    
    builder.add_edge("burn", "agent")
    builder.add_edge("tools", "agent")
    
    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()