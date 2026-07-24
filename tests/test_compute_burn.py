"""
Quick integration test for Financial Snapshot.
Run:
    python tests/test_financial_snapshot.py
"""

from datetime import datetime, timedelta

import pandas as pd

from services.financial_snapshot import build_financial_snapshot


def sample_transactions():
    today = datetime.today()

    return pd.DataFrame(
        [
            {
                "date": today - timedelta(days=85),
                "amount": -5000,
                "category": "salary",
                "is_one_time": False,
            },
            {
                "date": today - timedelta(days=70),
                "amount": -3000,
                "category": "rent",
                "is_one_time": False,
            },
            {
                "date": today - timedelta(days=45),
                "amount": -1200,
                "category": "software",
                "is_one_time": False,
            },
            {
                "date": today - timedelta(days=20),
                "amount": -25000,
                "category": "legal",
                "is_one_time": True,
            },
            {
                "date": today - timedelta(days=10),
                "amount": 40000,
                "category": "revenue",
                "is_one_time": False,
            },
        ]
    )


def main():

    snapshot = build_financial_snapshot(
        transactions_df=sample_transactions(),
        cash_balance=500000,
        monthly_revenue=40000,
        startup_profile={
            "country": "India",
            "stage": "Seed",
            "industry": "SaaS",
        },
        scenario_overrides={},
    )

    print("\n========== SNAPSHOT ==========\n")

    print(snapshot.keys())

    print("\nMetrics\n")
    print(snapshot["metrics"])

    print("\nTime Series\n")
    print(snapshot["financial_timeseries"])

    print("\nBreakdown\n")
    print(snapshot["monthly_breakdown"])

    print("\nAssumptions\n")
    print(snapshot["assumptions"])

    # -------------------------
    # Assertions
    # -------------------------

    assert "metrics" in snapshot
    assert "financial_timeseries" in snapshot
    assert "monthly_breakdown" in snapshot
    assert "assumptions" in snapshot

    metrics = snapshot["metrics"]

    assert metrics["gross_burn"] >= 0
    assert metrics["net_burn"] >= 0
    assert metrics["cash_balance"] == 500000
    assert metrics["monthly_revenue"] == 40000

    assumptions = snapshot["assumptions"]

    assert assumptions["fully_loaded_ratio"] > 1
    assert assumptions["burn_window_months"] == 3

    print("\n✅ ALL TESTS PASSED")


if __name__ == "__main__":
    main()