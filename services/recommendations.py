"""
Agentic Recommendation Engine.

Uses an LLM to act as an AI CFO and generate strategic
recommendations from the financial snapshot and forecast.
"""

from typing import List, Dict, Any, Literal

import json

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from settings import settings


# ============================================================
# Pydantic Schemas
# ============================================================

class Recommendation(BaseModel):
    priority: Literal[
        "Critical",
        "High",
        "Medium",
        "Low",
    ]

    category: str = Field(
        description="Area affected such as Cash Flow, Burn, Hiring, Revenue, Runway."
    )

    title: str

    reason: str

    recommended_action: str

    expected_impact: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    temperature=0.2,
)

recommendation_llm = llm.with_structured_output(
    RecommendationResponse
)


# ============================================================
# Recommendation Engine
# ============================================================

def generate_agentic_recommendations(
    financial_snapshot: Dict[str, Any],
    forecast_results: Dict[str, Any] | None,
    startup_profile: Dict[str, Any],
    scenario_overrides: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    AI CFO recommendation engine.

    Performs reasoning over financial state and
    returns prioritized recommendations.
    """

    system_prompt = """
You are the Chief Financial Officer (CFO) of a startup.

You are responsible for evaluating the company's current financial
health and future outlook.

You will receive:

1. Startup Profile
2. Current Financial Snapshot
3. Forecast Results
4. Active Scenario Overrides

Your responsibilities are:

• Identify the most important financial problems.

• Rank them according to business impact.

• Assign one priority:

    - Critical
    - High
    - Medium
    - Low

• Explain WHY each issue matters.

• Recommend practical actions.

• Estimate the expected business impact.

• Give a confidence score between 0 and 1.

Rules:

- Never invent financial numbers.
- Use ONLY the supplied data.
- Recommendations should be actionable.
- Prioritize survival before optimization.
- If finances look healthy, recommend growth opportunities instead.
"""

    payload = {
        "startup_profile": startup_profile,
        "financial_snapshot": financial_snapshot,
        "forecast_results": forecast_results,
        "scenario_overrides": scenario_overrides or {},
    }

    response: RecommendationResponse = recommendation_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=json.dumps(
                    payload,
                    indent=2,
                    default=str,
                )
            ),
        ]
    )

    return [
        rec.model_dump()
        for rec in response.recommendations
    ]