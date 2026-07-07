"""
Global state definition for the LangGraph.
"""

from typing import Dict, Any, List, Optional, Annotated, Literal, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class StartupProfile(TypedDict):
    """Startup profile - user configurable in UI."""
    name: str
    stage: Literal["Pre-seed", "Seed", "Series A", "Series B", "Series C+"]
    currency: str
    industry: Optional[str]
    country: Optional[str]
    founded_date: Optional[str]


class GlobalState(TypedDict):
    """
    Global state for the AI CFO system.
    """
    # Chat history - handles both dict and BaseMessage types
    messages: Annotated[List[Union[Dict[str, Any], BaseMessage]], add_messages]
    
    # Startup profile (user-configurable in UI)
    startup_profile: StartupProfile
    
    # Financial data
    cash_balance: float
    monthly_revenue: float
    computed_metrics: Optional[Dict[str, Any]]
    
    # Scenarios
    scenario_overrides: Dict[str, Any]
    active_scenario: Optional[str]
    scenario_history: List[Dict[str, Any]]
    
    # Forecasting
    forecast_results: Optional[Dict[str, Any]]
    runway_forecast: Optional[Dict[str, Any]]
    
    # Recommendations
    recommendations: List[Dict[str, Any]]
    
    # Audit trail
    assumptions_ledger: List[Dict[str, Any]]
    
    # Routing
    next_action: Literal["scenario", "burn", "forecast", "recommendation", "end"]
    requires_recompute: bool
    current_agent: str
    error_state: Optional[str]
    
    # Data (serializable - NO DataFrames)
    transactions_data: List[Dict[str, Any]]
     # Data (serializable - NO DataFrames)
    transactions_data: List[Dict[str, Any]]
    
    # ===== NEW FIELDS FOR PARSER =====
    raw_files: List[Dict[str, Any]]  # Uploaded files: [{"filename": str, "content": str/bytes, "type": str}]
    parsing_status: Literal["pending", "done", "failed", "no_files"]  # Status of parsing