from typing import Dict, Any
import pandas as pd

from config.langsmith import traced

from graph import state
from services.compute_burn import compute_burn
from services.timeseries import build_financial_timeseries
from services.breakdown import build_monthly_breakdown
from services.assumptions import build_assumptions



@traced("financial_snapshot", tags=["snapshot", "finance"])
def build_financial_snapshot(
    transactions_df: pd.DataFrame,
    cash_balance: float,
    monthly_revenue: float,  # <-- This is your fallback from Streamlit
    scenario_overrides: Dict[str, Any] | None = None,
    startup_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build the complete current financial snapshot.
    """

    # -------------------------------------------------------
    # 0. Dynamically calculate real revenue from CSV FIRST
    # -------------------------------------------------------
    # Make sure your columns are exactly 'Amount' and 'date' in the df
    # 0. Dynamically calculate real revenue from CSV FIRST
    # Use lowercase 'amount' to match the parser's output
    if 'amount' in transactions_df.columns:
        amt_col = 'amount'
    elif 'Amount' in transactions_df.columns:
        amt_col = 'Amount'
    else:
        # Fallback if the parser completely changes the name
        amt_col = transactions_df.columns[2] # Assuming it's the 3rd column

    csv_revenue = transactions_df[transactions_df[amt_col] > 0][amt_col].sum()
    # csv_revenue = transactions_df[transactions_df['Amount'] > 0]['Amount'].sum()

    if csv_revenue > 0:
        # Count the number of unique months in the dataset
        num_months = transactions_df['date'].dt.to_period('M').nunique()
        num_months = max(1, num_months) # Prevent division by zero
        
        # Override the Streamlit argument with the true CSV average
        monthly_revenue = csv_revenue / num_months
    else:
        # If CSV has no revenue, just keep the 'monthly_revenue' argument passed into the function
        pass 

    # -------------------------------------------------------
    # 1. Compute current financial metrics
    # -------------------------------------------------------
    # Now this uses the correct, dynamically calculated revenue!
    burn_result = compute_burn(
        transactions_df=transactions_df,
        cash_balance=cash_balance,
        monthly_revenue=monthly_revenue,
        scenario_overrides=scenario_overrides,
    )
    metrics = burn_result["metrics"]

    # -------------------------------------------------------
    # 2. Historical series for forecasting
    # -------------------------------------------------------
    financial_timeseries = build_financial_timeseries(
        transactions_df=transactions_df,
        monthly_revenue=monthly_revenue,
        scenario_overrides=scenario_overrides,
    )

    # -------------------------------------------------------
    # 3. Monthly breakdown (UI / reporting)
    # -------------------------------------------------------
    monthly_breakdown = build_monthly_breakdown(
        transactions_df=transactions_df,
    )

    # -------------------------------------------------------
    # 4. Assumptions ledger
    # -------------------------------------------------------
    assumptions = build_assumptions(
        startup_profile=startup_profile,
        scenario_overrides=scenario_overrides,
    )

    # -------------------------------------------------------
    # 5. Financial Snapshot
    # -------------------------------------------------------
    return {
        "metrics": metrics,
        "financial_timeseries": financial_timeseries,
        "monthly_breakdown": monthly_breakdown,
        "assumptions": assumptions,
    }