"""
Burn Calculator Node.
Builds the Financial Snapshot from parsed transactions.
"""

import json
import logging
# from turtle import st
import streamlit as st
import pandas as pd
from langchain_core.messages import ToolMessage

from graph.state import GlobalState
from services.compute_burn import (
    
    convert_to_serializable,
)
from services.financial_snapshot import build_financial_snapshot
from utils.helpers import generate_mock_transactions

logger = logging.getLogger(__name__)


"""
Burn Calculator Node.
Builds the Financial Snapshot from parsed transactions.
"""

import json
import logging
import pandas as pd

from langchain_core.messages import ToolMessage

from graph.state import GlobalState
# from services.compute_burn import build_financial_snapshot
from utils.helpers import generate_mock_transactions,convert_to_serializable

logger = logging.getLogger(__name__)


def burn_calculator_node(state: GlobalState) -> dict:
    """
    Creates the Financial Snapshot.

    Responsibilities
    ----------------
    ✓ Load transaction history
    ✓ Apply scenario overrides
    ✓ Compute current financial metrics
    ✓ Build financial snapshot

    Does NOT:
        - Forecast
        - Generate recommendations
        - Call any LLM
    """

    # ---------------------------------------------------------
    # Load transaction history
    # ---------------------------------------------------------
    logger.info(
    f"Supervisor financial_snapshot exists = {state.get('financial_snapshot') is not None}"
)
    transactions_data = state.get("transactions_data", [])

    if not transactions_data:
        logger.warning("No transactions found. Using mock data.")
        transactions_df = generate_mock_transactions(months=12)
    else:
        transactions_df = pd.DataFrame(transactions_data)

        if "date" in transactions_df.columns:
            transactions_df["date"] = pd.to_datetime(
                transactions_df["date"]
            )

    cash_balance = float(state.get("cash_balance") or 0)
    monthly_revenue = float(state.get("monthly_revenue" ) or 0)
    scenario_overrides = state.get("scenario_overrides") or {}

    logger.info(
        f"Building Financial Snapshot "
        f"(cash={cash_balance:,.2f}, "
        f"revenue={monthly_revenue:,.2f})"
    )

    # ---------------------------------------------------------
    # Build Financial Snapshot
    # ---------------------------------------------------------

    snapshot = build_financial_snapshot(
        transactions_df=transactions_df,
        cash_balance=cash_balance,
        monthly_revenue=monthly_revenue,
        scenario_overrides=scenario_overrides,
    )
    # 🚨 ADD THESE DIAGNOSTIC LOGS HERE 🚨
    logger.info("--- DIAGNOSTIC LOGS: BURN CALCULATOR ---")
    logger.info(f"1. Applied Scenario Overrides: {scenario_overrides}")
    
    # Safely extract metrics from the Pydantic object
    metrics = snapshot.get("metrics")
    if metrics:
        # Use dot notation or getattr for Pydantic objects
        logger.info(f"2. Output Gross Burn: {getattr(metrics, 'gross_burn', 'N/A')}")
        logger.info(f"3. Output Net Burn: {getattr(metrics, 'net_burn', 'N/A')}")
    
    logger.info(f"4. Timeseries Keys: {list(snapshot.get('financial_timeseries', {}).keys())}")
    logger.info("----------------------------------------")
    # ---------------------------------------------------------
    # Find originating tool call id
    # ---------------------------------------------------------

    tool_call_id = "burn_calc_001"

    messages = state.get("messages", [])

    if messages:
        last_msg = messages[-1]

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:

                name = (
                    tc.get("name")
                    if isinstance(tc, dict)
                    else tc.name
                )

                if name == "calculate_burn_metrics":
                    tool_call_id = (
                        tc.get("id")
                        if isinstance(tc, dict)
                        else tc.id
                    )
                    break

    logger.info("Financial Snapshot built successfully.")

    # ---------------------------------------------------------
    # Return graph updates
    # ---------------------------------------------------------

    snapshot = convert_to_serializable(snapshot)
    if "state" not in st.session_state:
        st.session_state["state"] = {}

    st.session_state["state"]["financial_snapshot"] = snapshot
    return {
        "financial_snapshot": snapshot,
        "financial_timeseries": snapshot["financial_timeseries"],
        "computed_metrics": snapshot["metrics"],
        "messages": [
            ToolMessage(
                content=json.dumps(snapshot),
                tool_call_id=tool_call_id,
                name="calculate_burn_metrics",
            )
        ],
    }
