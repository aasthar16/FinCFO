"""
End-to-end Financial Pipeline Test

Run:
    python tests/test_financial_pipeline.py
"""

from datetime import datetime, timedelta

import pandas as pd

from services.financial_snapshot import build_financial_snapshot
from services.forecasting import forecast_financials
from services.recommendations import generate_agentic_recommendations


def sample_transactions():

    today = datetime.today()

    rows = []

    # 12 months of realistic history

    for i in range(12):

        month = today - timedelta(days=30 * (11 - i))

        rows.extend(
            [
                {
                    "date": month,
                    "amount": -24000,
                    "category": "salary",
                    "is_one_time": False,
                },
                {
                    "date": month + timedelta(days=2),
                    "amount": -7000,
                    "category": "rent",
                    "is_one_time": False,
                },
                {
                    "date": month + timedelta(days=5),
                    "amount": -3500,
                    "category": "software",
                    "is_one_time": False,
                },
                {
                    "date": month + timedelta(days=8),
                    "amount": -4500,
                    "category": "marketing",
                    "is_one_time": False,
                },
                {
                    "date": month + timedelta(days=10),
                    "amount": 55000,
                    "category": "revenue",
                    "is_one_time": False,
                },
            ]
        )

    # One-time legal expense

    rows.append(
        {
            "date": today - timedelta(days=35),
            "amount": -45000,
            "category": "legal",
            "is_one_time": True,
        }
    )

    return pd.DataFrame(rows)


def main():

    print("\n==============================")
    print("BUILDING FINANCIAL SNAPSHOT")
    print("==============================")

    snapshot = build_financial_snapshot(
        transactions_df=sample_transactions(),
        cash_balance=900000,
        monthly_revenue=55000,
        scenario_overrides={},
    )

    print("\nSnapshot Keys")
    print(snapshot.keys())

    print("\nMetrics")
    print(snapshot["metrics"])

    print("\nMonthly Time Series")
    print(snapshot["financial_timeseries"]["monthly"][:3], "...")

    print("\n==============================")
    print("RUNNING FORECAST")
    print("==============================")

    forecast = forecast_financials(
        financial_snapshot=snapshot,
        horizon=12,
    )

    print("\nForecast Keys")
    print(forecast.keys())

    print("\nModel Used")
    print(forecast["model"])

    print("\nFirst Forecast Row")
    print(forecast["forecast"][0])

    print("\nRunway")
    print(forecast["runway"])

    print("\nRisk")
    print(forecast["risk"])

    print("\nInsights")
    print(forecast["insights"])

    print("\n==============================")
    print("LLM RECOMMENDATIONS")
    print("==============================")

    startup_profile = {
        "name": "FinCFO Demo",
        "stage": "Seed",
        "industry": "SaaS",
        "country": "India",
        "currency": "USD",
    }

    recommendations = generate_agentic_recommendations(
        financial_snapshot=snapshot,
        forecast_results=forecast,
        startup_profile=startup_profile,
        scenario_overrides={},
    )

    print()

    for i, rec in enumerate(recommendations, 1):

        print(f"{i}. {rec['priority']}")

        print("Title:", rec["title"])
        print("Category:", rec["category"])
        print("Reason:", rec["reason"])
        print("Action:", rec["recommended_action"])
        print("Impact:", rec["expected_impact"])
        print("Confidence:", rec["confidence"])
        print()

    # ----------------------------
    # Assertions
    # ----------------------------

    assert "metrics" in snapshot
    assert "financial_timeseries" in snapshot
    assert "monthly_breakdown" in snapshot
    assert "assumptions" in snapshot

    metrics = snapshot["metrics"]

    assert metrics.gross_burn >= 0
    assert metrics.net_burn >= 0
    assert metrics.cash_balance == 900000
    assert metrics.monthly_revenue == 55000

    assert "forecast" in forecast
    assert "runway" in forecast
    assert "risk" in forecast
    assert "insights" in forecast

    assert len(forecast["forecast"]) == 12

    assert len(recommendations) > 0

    print("\n==============================")
    print("ALL TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()