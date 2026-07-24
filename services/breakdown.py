"""
Monthly expense breakdown.
"""

from typing import List, Dict, Any
import pandas as pd


def build_monthly_breakdown(
    transactions_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Monthly expense summary for dashboard display.
    """

    if transactions_df.empty:
        return []

    df = transactions_df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    expenses = df[df["amount"] < 0]

    if expenses.empty:
        return []

    monthly = (
        expenses
        .groupby("month")
        .agg(
            total_expense=("amount", lambda x: abs(x.sum())),
            transaction_count=("amount", "count"),
        )
        .reset_index()
    )

    result = []

    for _, row in monthly.iterrows():

        result.append(
            {
                "month": str(row["month"]),
                "total_expense": float(row["total_expense"]),
                "transaction_count": int(row["transaction_count"]),
            }
        )

    return result