"""
Build historical financial time series for forecasting.
"""

from typing import Dict, Any

import pandas as pd


def build_financial_timeseries(
    transactions_df: pd.DataFrame,
    monthly_revenue: float,
    scenario_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build monthly financial history.

    Returns a feature table suitable for forecasting models like
    Darts, Prophet, StatsForecast, etc.
    """

    if transactions_df.empty:
        return {"monthly": []}

    df = transactions_df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    if "is_one_time" not in df.columns:
        df["is_one_time"] = False

    monthly_records = []

    months = sorted(df["month"].unique())

    for month in months:

        month_df = df[df["month"] == month]

        # -----------------------------
        # Revenue
        # -----------------------------

        revenue = month_df.loc[
            month_df["amount"] > 0,
            "amount",
        ].sum()

        # fallback if parser never extracted revenue
        if revenue == 0:
            revenue = monthly_revenue

        # -----------------------------
        # Expenses
        # -----------------------------

        expense_df = month_df[
            month_df["amount"] < 0
        ]

        recurring_expenses = abs(
            expense_df.loc[
                ~expense_df["is_one_time"],
                "amount",
            ].sum()
        )

        one_time_expenses = abs(
            expense_df.loc[
                expense_df["is_one_time"],
                "amount",
            ].sum()
        )

        gross_burn = (
            recurring_expenses +
            one_time_expenses
        )

        net_burn = max(
            gross_burn - revenue,
            0,
        )

        monthly_records.append(
            {
                "month": str(month),

                "revenue": float(revenue),

                "gross_burn": float(gross_burn),

                "net_burn": float(net_burn),

                "recurring_expenses": float(
                    recurring_expenses
                ),

                "one_time_expenses": float(
                    one_time_expenses
                ),

                "transaction_count": int(
                    len(month_df)
                ),
            }
        )

    return {
        "monthly": monthly_records,
    }