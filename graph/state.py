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
    messages: Annotated[List[BaseMessage],add_messages]
    
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
    
    requires_recompute: bool
    current_agent: str
    error_state: Optional[str]
    
    # Data (serializable - NO DataFrames)
    transactions_data: List[Dict[str, Any]]
    
    
    # ===== NEW FIELDS FOR PARSER =====
    raw_files: List[Dict[str, Any]]  # Uploaded files: [{"filename": str, "content": str/bytes, "type": str}]
    parsing_status: Literal["pending", "done", "failed", "no_files"]  # Status of parsing

    # ===== NEW FIELDS FOR TOOLS =====
    financial_snapshot: Optional[Dict[str, Any]]

    forecast_results: Optional[Dict[str, Any]]

    recommendations: List[Dict[str, Any]]


# Thread-local storage for current state (used by tools)
import threading

_state_store = threading.local()

def set_current_state(state: dict):
    """Store current state for tool access."""
    _state_store.state = state

def get_current_state() -> dict:
    """Get current state from tools."""
    return getattr(_state_store, 'state', {})