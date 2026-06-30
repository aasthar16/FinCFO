"""
Burn & Expense node.
"""

from typing import Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from langchain_core.messages import AIMessage

from graph.state import GlobalState
from services.compute_burn import compute_burn, convert_to_serializable
from config.langsmith import traced
from utils.helpers import generate_mock_transactions


@traced("burn_expense_node", tags=["burn", "expense"])
def burn_expense_node(state: GlobalState) -> Dict[str, Any]:
    """
    Burn & Expense Node: Computes burn metrics using Python.
    """
    # Get data from state
    cash_balance = float(state.get("cash_balance", 1000000))
    monthly_revenue = float(state.get("monthly_revenue", 100000))
    scenario_overrides = state.get("scenario_overrides", {})
    transactions_data = state.get("transactions_data")
    
    # Convert transactions_data to DataFrame
    if transactions_data:
        transactions_df = pd.DataFrame(transactions_data)
        if 'date' in transactions_df.columns:
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    else:
        transactions_df = generate_mock_transactions(months=6)
    
    # Compute burn metrics
    burn_result = compute_burn(
        transactions_df=transactions_df,
        cash_balance=cash_balance,
        monthly_revenue=monthly_revenue,
        scenario_overrides=scenario_overrides,
    )
    
    metrics = burn_result["metrics"]
    
    # Convert metrics to dict and ensure all values are Python native types
    metrics_dict = {
        "gross_burn": float(metrics.gross_burn),
        "net_burn": float(metrics.net_burn),
        "net_burn_3m_avg": float(metrics.net_burn_3m_avg),
        "one_time_expenses": float(metrics.one_time_expenses),
        "recurring_expenses": float(metrics.recurring_expenses),
        "fully_loaded_ratio": float(metrics.fully_loaded_ratio),
        "cash_runway_months": float(metrics.cash_runway_months) if metrics.cash_runway_months is not None else 0.0,
        "burn_multiple": float(metrics.burn_multiple) if metrics.burn_multiple is not None else 0.0,
        "cash_balance": float(metrics.cash_balance),
        "monthly_revenue": float(metrics.monthly_revenue),
    }
    
    # Generate summary
    runway_text = f"{metrics.cash_runway_months:.1f}" if metrics.cash_runway_months is not None and metrics.cash_runway_months > 0 else "∞"
    burn_multiple_text = f"{metrics.burn_multiple:.1f}" if metrics.burn_multiple is not None and metrics.burn_multiple > 0 else "N/A"
    
    summary = f"""
💰 **Burn Analysis**

• Gross Burn: ${metrics.gross_burn:,.0f}/month
• Net Burn: ${metrics.net_burn:,.0f}/month
• 3-Month Avg Net Burn: ${metrics.net_burn_3m_avg:,.0f}/month
• Runway: {runway_text} months
• Burn Multiple: {burn_multiple_text}x
• One-Time Expenses: ${metrics.one_time_expenses:,.0f}
"""
    
    if scenario_overrides:
        summary += f"\n📊 **Scenario Applied:** {state.get('active_scenario', 'custom')}"
    
    # Ensure monthly_breakdown is serializable
    monthly_breakdown = burn_result.get("monthly_breakdown", [])
    if isinstance(monthly_breakdown, pd.DataFrame):
        monthly_breakdown = monthly_breakdown.to_dict('records')
    
    # Convert any numpy types in monthly_breakdown
    monthly_breakdown = convert_to_serializable(monthly_breakdown)
    
    return {
        "computed_metrics": convert_to_serializable(metrics_dict),
        "monthly_breakdown": monthly_breakdown,
        "next_action": "forecast",
        "current_agent": "burn",
        "requires_recompute": False,
        "messages": [AIMessage(content=summary)],
        "transactions_data": convert_to_serializable(transactions_df.to_dict('records')) if isinstance(transactions_df, pd.DataFrame) else [],
    }