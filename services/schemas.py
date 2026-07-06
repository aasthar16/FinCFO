# services/schemas.py (NEW FILE)

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class EnrichedRecommendation(BaseModel):
    """Schema for a single enriched recommendation."""
    priority: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Priority level based on urgency and impact"
    )
    category: Literal[
        "cash_management", 
        "efficiency", 
        "expense_management", 
        "revenue_growth",
        "financial_health",
        "general"
    ] = Field(
        description="Category of the recommendation"
    )
    title: str = Field(
        description="Short, actionable title"
    )
    description: str = Field(
        description="Detailed description of the recommendation"
    )
    suggested_actions: List[str] = Field(
        description="List of specific, actionable steps"
    )
    impact_estimate: str = Field(
        description="Estimated impact if implemented"
    )
    source: Literal["deterministic", "llm_enriched"] = Field(
        default="deterministic",
        description="Source of the recommendation"
    )
    
    # LLM-enriched fields (optional)
    contextual_insight: Optional[str] = Field(
        default=None,
        description="Context-specific insight based on startup stage"
    )
    priority_justification: Optional[str] = Field(
        default=None,
        description="Why this matters NOW given current situation"
    )
    industry_best_practice: Optional[str] = Field(
        default=None,
        description="Relevant industry benchmark or best practice"
    )
    expected_roi: Optional[str] = Field(
        default=None,
        description="Estimated ROI or timeline for results"
    )


class EnrichedRecommendationsResponse(BaseModel):
    """Schema for LLM enrichment response."""
    recommendations: List[EnrichedRecommendation] = Field(
        description="List of enriched recommendations"
    )
    overall_assessment: str = Field(
        default="",
        description="Overall financial health assessment"
    )
    key_priority: str = Field(
        default="",
        description="The single most important action to take"
    )

    # services/schemas.py (ADD THIS)


class ScenarioExtractionResult(BaseModel):
    """Schema for scenario extraction from user query."""
    
    action: Literal["hire", "fire", "replace", "revenue_change", "expense_change", "none"] = Field(
        default="none",
        description="Type of scenario action detected"
    )
    
    count: int = Field(
        default=0,
        description="Number of people to hire/fire"
    )
    
    role: str = Field(
        default="employee",
        description="Job role mentioned (e.g., engineer, designer, manager)"
    )
    
    salary: Optional[float] = Field(
        default=None,
        description="Monthly salary per person (in dollars)"
    )
    
    headcount_change: int = Field(
        default=0,
        description="Net change in headcount (positive = hire, negative = fire)"
    )
    
    revenue_change: Optional[float] = Field(
        default=None,
        description="Change in monthly revenue (in dollars)"
    )
    
    one_time_expenses: Optional[float] = Field(
        default=None,
        description="One-time expense amount (in dollars)"
    )
    
    is_addition: bool = Field(
        default=False,
        description="True if adding to previous scenario, False if new scenario"
    )
    
    explanation: str = Field(
        default="",
        description="Human-readable explanation of what was parsed"
    )