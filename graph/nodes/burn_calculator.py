"""
Burn Calculator Node - Deterministic computation.
Runs when agent requests burn metrics. Has full state access.
"""

import json
import logging
import pandas as pd
from langchain_core.messages import ToolMessage

from graph.state import GlobalState
from services.compute_burn import compute_burn, convert_to_serializable
from utils.helpers import generate_mock_transactions

logger = logging.getLogger(__name__)


def burn_calculator_node(state: GlobalState) -> dict:
    """
    Compute burn metrics from transaction data.
    Called when agent requests calculate_burn_metrics.
    Has full access to state.
    """
    
    # Get transactions from state
    transactions_data = state.get("transactions_data", [])
    
    if not transactions_data:
        logger.warning("No transactions found, using mock data")
        transactions_df = generate_mock_transactions(months=6)
    else:
        transactions_df = pd.DataFrame(transactions_data)
        if 'date' in transactions_df.columns:
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    
    cash_balance = float(state.get("cash_balance", 1200000))
    monthly_revenue = float(state.get("monthly_revenue", 85000))
    scenario_overrides = state.get("scenario_overrides") or None
    
    logger.info(f"💰 Computing burn: cash=${cash_balance:,.0f}, revenue=${monthly_revenue:,.0f}, overrides={scenario_overrides}")
    
    result = compute_burn(
        transactions_df=transactions_df,
        cash_balance=cash_balance,
        monthly_revenue=monthly_revenue,
        scenario_overrides=scenario_overrides,
    )
    
    # Store in state
    state["computed_metrics"] = result.get("metrics", result)
    
    # Return as ToolMessage so agent sees the result
    tool_result = convert_to_serializable(result)
    
    # Find the tool_call_id from the agent's message
    tool_call_id = "burn_calc_001"
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                name = tc.get('name') if isinstance(tc, dict) else tc.name
                if name == "calculate_burn_metrics":
                    tool_call_id = tc.get('id') if isinstance(tc, dict) else tc.id
                    break
    
    logger.info(f"✅ Burn computed: gross={result.get('metrics').gross_burn if hasattr(result.get('metrics'), 'gross_burn') else 'N/A'}")
    
    return {
        "messages": [
            ToolMessage(
                content=json.dumps(tool_result, default=str),
                tool_call_id=tool_call_id,
                name="calculate_burn_metrics"
            )
        ]
    }