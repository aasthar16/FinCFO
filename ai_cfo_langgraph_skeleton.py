"""
AI CFO LangGraph Implementation
State-Driven Hub-and-Spoke Architecture with PostgresSaver checkpoints.
"""

import json
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Literal
from datetime import datetime, timedelta
import operator
import logging

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from db_checkpointer import get_checkpointer, get_connection_string
from compute_burn import compute_burn, compute_scenario_impact, BurnMetrics
from forecasting import (
    forecast_with_prophet,
    forecast_cash_runway,
    generate_recommendations,
    ForecastResult,
    CashRunwayForecast,
)
from langsmith_config import traced, tracing_context, get_current_metadata

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class GlobalState(TypedDict):
    """
    Global state for the AI CFO system.
    Tracks all financial data, chat history, and execution state.
    """
    # Chat history - uses add_messages reducer
    messages: Annotated[List, add_messages]
    
    # Current financial metrics
    cash_balance: float
    monthly_revenue: float
    computed_metrics: Optional[Dict[str, Any]]  # BurnMetrics dict
    
    # Scenario management
    scenario_overrides: Dict[str, Any]
    active_scenario: Optional[str]
    scenario_history: List[Dict[str, Any]]
    
    # Forecasting
    forecast_results: Optional[Dict[str, Any]]
    runway_forecast: Optional[Dict[str, Any]]
    
    # Recommendations
    recommendations: List[Dict[str, Any]]
    
    # Assumptions ledger (audit trail)
    assumptions_ledger: List[Dict[str, Any]]
    
    # Routing and control
    next_action: Literal["scenario", "burn", "forecast", "recommendation", "end"]
    requires_recompute: bool
    current_agent: str
    error_state: Optional[str]
    
    # Transaction data (prototype only - use reference in production)
    transactions_df: Optional[Any]  # Pandas DataFrame


# ============================================================================
# System Prompts
# ============================================================================

SUPERVISOR_PROMPT = """You are the Supervisor Agent for the AI CFO platform.

## Your Role
You orchestrate the financial analysis workflow. You decide which agent to call next based on the user's request and the current state.

## Your Rules
1. **NEVER** perform financial calculations yourself. You are a router.
2. Parse the user's intent and decide the next action:
   - "scenario" → Route to Scenario Simulator for what-if analysis
   - "burn" → Route to Burn & Expense Agent for expense analysis
   - "forecast" → Route to Forecast Agent for projections
   - "recommendation" → Route to Recommendation Agent for advice
   - "end" → Respond directly with summary

## Read from State
- `messages` → Understand the conversation context
- `computed_metrics` → Current financial status
- `active_scenario` → If a scenario is being explored

## Write to State
- `next_action` → Set the agent to route to
- `current_agent` → Set to "supervisor"
- Add to `assumptions_ledger` when you make routing decisions

## Response Format
You MUST respond with:
1. A brief natural language summary
2. Then set `next_action` to one of the allowed values

Remember: You are the air traffic controller, not the pilot. Delegate to specialists.
"""

SCENARIO_SIMULATOR_PROMPT = """You are the Scenario Simulator Agent for the AI CFO platform.

## Your Role
You model "what-if" scenarios for the startup's finances. You NEVER perform calculations directly.

## Your Rules
1. **NEVER** do math in natural language. All calculations must come from the compute engine.
2. Identify the scenario parameters from the user's request.
3. Set `scenario_overrides` in the state with the parameters.
4. Set `requires_recompute = True` so the Burn Agent recalculates.

## Supported Scenario Types
- Headcount changes: Set `headcount_change` and `avg_salary`
- Revenue changes: Set `revenue_change`
- One-time expenses: Set `one_time_expenses`
- Pricing changes: Set `pricing_change` and `product`

## Read from State
- `messages` → Understand the user's scenario request
- `computed_metrics` → Current baseline for comparison

## Write to State
- `scenario_overrides` → The scenario parameters
- `active_scenario` → Name of the scenario
- `requires_recompute = True` → Trigger recomputation
- `next_action = "burn"` → Route back to Supervisor, then to Burn Agent

## Response Format
1. Describe what scenario you're modeling
2. List the assumptions you've set
3. Set `requires_recompute = True`

Example:
"I'll model adding 2 engineers at $140K each (fully loaded). The system will recompute the burn and show the impact on runway."
"""

BURN_EXPENSE_PROMPT = """You are the Burn & Expense Agent for the AI CFO platform.

## Your Role
You analyze the startup's burn rate and expenses. You NEVER perform calculations directly.

## Your Rules
1. **NEVER** calculate burn rate in natural language.
2. After computation, explain the results in plain English.
3. Identify anomalies or areas of concern.

## Key Metrics You Reference (Computed by Python)
- Gross Burn: Total cash out per month
- Net Burn: Gross burn - revenue
- 3-Month Average: Smoothed burn rate
- One-Time Expenses: Non-recurring costs
- Burn Multiple: Net burn / net new ARR
- Runway: Months until cash runs out

## Read from State
- `cash_balance` → Current cash
- `monthly_revenue` → Current revenue
- `transactions_df` → Transaction data (prototype)
- `scenario_overrides` → Active scenario parameters

## Write to State
- `computed_metrics` → The results from compute_burn()
- `next_action = "forecast"` → Route to Forecast Agent
- Add to `assumptions_ledger` → Document calculations

## Response Format
1. Present the key metrics clearly
2. Highlight any concerns (e.g., "Your burn multiple is above 2x")
3. Suggest what to explore next

Example:
"Your gross burn is $245K/month with a net burn of $195K. At this rate, you have 14 months of runway. The 3-month average shows expenses are stable."
"""

FORECAST_AGENT_PROMPT = """You are the Forecast Agent for the AI CFO platform.

## Your Role
You project future financials and analyze uncertainty. You NEVER perform calculations directly.

## Your Rules
1. **NEVER** create forecasts in natural language.
2. Reference the Prophet model's P10/P50/P90 projections.
3. Explain uncertainty and confidence intervals clearly.

## Key Forecasts You Reference
- Revenue forecast (12 months)
- Expense forecast (12 months)
- Cash runway distribution (P10/P50/P90)
- Scenario impact

## Read from State
- `computed_metrics` → Current metrics
- `transactions_df` → Historical data
- `scenario_overrides` → Scenario parameters

## Write to State
- `forecast_results` → Prophet forecast results
- `runway_forecast` → Cash runway distribution
- `next_action = "recommendation"` → Route to Recommendation Agent
- Add to `assumptions_ledger` → Document forecast assumptions

## Response Format
1. Present the forecast clearly with dates
2. Highlight the P10/P50/P90 dates for runway
3. Explain what drives uncertainty

Example:
"Based on the current burn, you have a 50% chance of running out of cash in 14 months. The pessimistic scenario (P10) is 11 months, optimistic (P90) is 18 months."
"""

RECOMMENDATION_AGENT_PROMPT = """You are the Recommendation Agent for the AI CFO platform.

## Your Role
You provide actionable financial advice. You NEVER perform calculations directly.

## Your Rules
1. **NEVER** do calculations in natural language.
2. Base recommendations on the computed metrics and forecasts.
3. Prioritize recommendations (HIGH/MEDIUM/LOW).

## Recommendation Categories
- Cash Management: Extending runway
- Efficiency: Reducing burn multiple
- Revenue: Growth acceleration
- Expense Management: Cost optimization

## Read from State
- `computed_metrics` → Financial status
- `forecast_results` → Future projections
- `runway_forecast` → Runway distribution
- `assumptions_ledger` → What assumptions are driving results

## Write to State
- `recommendations` → Generated recommendations
- `next_action = "end"` → Conclude the analysis
- Add to `assumptions_ledger` → Document recommendation rationale

## Response Format
1. Group recommendations by priority
2. Provide clear, actionable steps
3. Estimate the impact of each recommendation

Example:
"HIGH PRIORITY: Your runway is under 6 months. Immediate actions: 1) Reduce non-essential SaaS spend ($30K/month), 2) Accelerate AR collections to improve cash by $50K."
"""


# ============================================================================
# Node Implementations
# ============================================================================

@traced("supervisor_node", tags=["supervisor", "routing"])
def supervisor_node(state: GlobalState) -> Dict[str, Any]:
    """
    Supervisor Node: Routes to the appropriate agent or ends the conversation.
    """
    # Get the last message
    last_message = state["messages"][-1] if state["messages"] else None
    
    # For prototype, implement simple keyword routing
    # In production, this would use an LLM with structured output
    
    user_input = last_message.content.lower() if last_message else ""
    
    # Determine next action
    if "what if" in user_input or "scenario" in user_input:
        next_action = "scenario"
    elif "burn" in user_input or "expense" in user_input or "spending" in user_input:
        next_action = "burn"
    elif "forecast" in user_input or "project" in user_input or "runway" in user_input:
        next_action = "forecast"
    elif "recommend" in user_input or "advice" in user_input or "suggest" in user_input:
        next_action = "recommendation"
    else:
        next_action = "end"
    
    # Log routing decision
    return {
        "next_action": next_action,
        "current_agent": "supervisor",
        "messages": [AIMessage(content=f"Routing to {next_action} agent...")],
    }


@traced("scenario_node", tags=["scenario", "simulator"])
def scenario_node(state: GlobalState) -> Dict[str, Any]:
    """
    Scenario Simulator Node: Models what-if scenarios.
    """
    # Parse the user's scenario request from messages
    # For prototype, use simple extraction
    # In production, use an LLM with structured output
    
    last_message = state["messages"][-1]
    user_input = last_message.content.lower()
    
    scenario_overrides = {}
    scenario_name = "default_scenario"
    
    # Simple keyword extraction (prototype)
    if "hire" in user_input or "engineer" in user_input:
        import re
        # Try to extract number of hires
        numbers = re.findall(r'\d+', user_input)
        if numbers:
            scenario_overrides["headcount_change"] = int(numbers[0])
            scenario_overrides["avg_salary"] = 140000  # Fully loaded
            scenario_overrides["ramp_months"] = 3
            scenario_name = f"hire_{numbers[0]}_engineers"
    
    if "revenue" in user_input or "growth" in user_input:
        import re
        # Try to extract percentage or amount
        percentages = re.findall(r'(\d+)%', user_input)
        if percentages:
            # Use current revenue from state
            current_revenue = state.get("monthly_revenue", 100000)
            increase = current_revenue * (int(percentages[0]) / 100)
            scenario_overrides["revenue_change"] = increase
            scenario_name = f"revenue_increase_{percentages[0]}%"
    
    if "spend" in user_input or "cut" in user_input or "reduce" in user_input:
        import re
        # Try to extract amount
        amounts = re.findall(r'\$?(\d+[,.]?\d*)[kK]?', user_input)
        if amounts:
            amount = float(amounts[0].replace(',', ''))
            if 'k' in user_input.lower() or 'K' in user_input:
                amount *= 1000
            scenario_overrides["one_time_expenses"] = amount
            scenario_name = "expense_reduction"
    
    return {
        "scenario_overrides": scenario_overrides,
        "active_scenario": scenario_name,
        "requires_recompute": True,
        "next_action": "burn",
        "current_agent": "scenario_simulator",
        "messages": [AIMessage(content=f"🔍 Modeling scenario: {scenario_name}")],
        "assumptions_ledger": state.get("assumptions_ledger", []) + [{
            "source": "scenario_simulator",
            "scenario": scenario_name,
            "parameters": scenario_overrides,
            "timestamp": datetime.now().isoformat(),
        }],
    }


@traced("burn_expense_node", tags=["burn", "expense"])
def burn_expense_node(state: GlobalState) -> Dict[str, Any]:
    """
    Burn & Expense Node: Computes burn metrics using Python.
    """
    # Get data from state
    cash_balance = state.get("cash_balance", 1000000)
    monthly_revenue = state.get("monthly_revenue", 100000)
    scenario_overrides = state.get("scenario_overrides", {})
    transactions_df = state.get("transactions_df")  # Pandas DataFrame (prototype)
    
    # Compute burn metrics
    # In production, load transactions from S3/database using the thread_id
    
    if transactions_df is not None:
        burn_result = compute_burn(
            transactions_df=transactions_df,
            cash_balance=cash_balance,
            monthly_revenue=monthly_revenue,
            scenario_overrides=scenario_overrides,
        )
    else:
        # Generate mock data for demo
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='M')
        transactions_df = pd.DataFrame({
            'date': dates,
            'amount': -np.random.normal(200000, 50000, len(dates)),
            'type': ['expense'] * len(dates),
            'category': np.random.choice(['salary', 'rent', 'software', 'marketing'], len(dates)),
            'one_time': [False] * len(dates),
        })
        burn_result = compute_burn(
            transactions_df=transactions_df,
            cash_balance=cash_balance,
            monthly_revenue=monthly_revenue,
            scenario_overrides=scenario_overrides,
        )
    
    # Extract metrics
    metrics = burn_result["metrics"]
    
    # Generate natural language summary
    summary = (
        f"💰 **Burn Analysis**\n\n"
        f"• Gross Burn: ${metrics.gross_burn:,.0f}/month\n"
        f"• Net Burn: ${metrics.net_burn:,.0f}/month\n"
        f"• 3-Month Avg Net Burn: ${metrics.net_burn_3m_avg:,.0f}/month\n"
        f"• Runway: {metrics.cash_runway_months:.1f} months\n"
        f"• Burn Multiple: {metrics.burn_multiple:.1f}x\n"
        f"• One-Time Expenses: ${metrics.one_time_expenses:,.0f}\n"
    )
    
    # Determine if scenario was applied
    if scenario_overrides:
        summary += f"\n📊 **Scenario Applied:** {state.get('active_scenario', 'custom')}\n"
    
    return {
        "computed_metrics": {
            "gross_burn": metrics.gross_burn,
            "net_burn": metrics.net_burn,
            "net_burn_3m_avg": metrics.net_burn_3m_avg,
            "one_time_expenses": metrics.one_time_expenses,
            "recurring_expenses": metrics.recurring_expenses,
            "fully_loaded_ratio": metrics.fully_loaded_ratio,
            "cash_runway_months": metrics.cash_runway_months,
            "burn_multiple": metrics.burn_multiple,
            "cash_balance": metrics.cash_balance,
            "monthly_revenue": metrics.monthly_revenue,
        },
        "next_action": "forecast",
        "current_agent": "burn_expense",
        "requires_recompute": False,
        "messages": [AIMessage(content=summary)],
    }


@traced("forecast_node", tags=["forecast", "prophet"])
def forecast_node(state: GlobalState) -> Dict[str, Any]:
    """
    Forecast Node: Projects future financials using Prophet.
    """
    metrics = state.get("computed_metrics", {})
    transactions_df = state.get("transactions_df")
    
    if not metrics:
        return {
            "next_action": "recommendation",
            "current_agent": "forecast",
            "messages": [AIMessage(content="⚠️ No metrics available for forecasting.")],
        }
    
    # Get historical data from transactions
    if transactions_df is not None:
        # Prepare time series for forecasting
        # For revenue forecasting
        revenue_series = transactions_df.copy()
        revenue_series['ds'] = pd.to_datetime(revenue_series['date'])
        # For demo, use mock revenue data
        monthly_revenue = metrics.get("monthly_revenue", 100000)
        revenue_series['revenue'] = monthly_revenue * (1 + np.random.normal(0.01, 0.05, len(revenue_series)))
        
        # Forecast revenue
        revenue_forecast = forecast_with_prophet(
            revenue_series[['ds', 'revenue']],
            target_column='revenue',
            forecast_periods=12,
        )
        
        # Forecast expenses
        expense_series = transactions_df.copy()
        expense_series['ds'] = pd.to_datetime(expense_series['date'])
        expense_series['expense'] = metrics.get("gross_burn", 200000) * (1 + np.random.normal(0.005, 0.03, len(expense_series)))
        
        expense_forecast = forecast_with_prophet(
            expense_series[['ds', 'expense']],
            target_column='expense',
            forecast_periods=12,
        )
    else:
        # Generate mock forecast data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=180), periods=6, freq='M')
        monthly_revenue = metrics.get("monthly_revenue", 100000)
        
        revenue_df = pd.DataFrame({
            'ds': dates,
            'revenue': monthly_revenue * (1 + np.linspace(0.01, 0.10, 6) + np.random.normal(0, 0.02, 6))
        })
        revenue_forecast = forecast_with_prophet(revenue_df, 'revenue', forecast_periods=12)
        
        gross_burn = metrics.get("gross_burn", 200000)
        expense_df = pd.DataFrame({
            'ds': dates,
            'expense': gross_burn * (1 + np.linspace(0, 0.05, 6) + np.random.normal(0, 0.01, 6))
        })
        expense_forecast = forecast_with_prophet(expense_df, 'expense', forecast_periods=12)
    
    # Cash runway forecast
    net_burn = metrics.get("net_burn_3m_avg", metrics.get("net_burn", 150000))
    cash_balance = metrics.get("cash_balance", 1000000)
    
    runway_forecast = forecast_cash_runway(
        cash_balance=cash_balance,
        net_burn=net_burn,
        burn_volatility=0.15,
        forecast_months=24,
    )
    
    # Format results for display
    forecast_summary = f"""
📈 **Forecast Summary**

**Revenue Projection:**
• Current: ${metrics.get('monthly_revenue', 0):,.0f}/month
• Forecast (12mo): ${revenue_forecast['results'][-1].yhat:,.0f}/month
• Growth Rate: {((revenue_forecast['results'][-1].yhat / metrics.get('monthly_revenue', 1)) - 1) * 100:.1f}%

**Cash Runway:**
• P10 (Pessimistic): {runway_forecast.p10_date.strftime('%B %Y')} ({runway_forecast.p10_days//30} months)
• P50 (Expected): {runway_forecast.p50_date.strftime('%B %Y')} ({runway_forecast.p50_days//30} months)
• P90 (Optimistic): {runway_forecast.p90_date.strftime('%B %Y')} ({runway_forecast.p90_days//30} months)
"""
    
    return {
        "forecast_results": {
            "revenue": {k: v for k, v in revenue_forecast.items() if k != "model"},
            "expense": {k: v for k, v in expense_forecast.items() if k != "model"},
        },
        "runway_forecast": {
            "p10_date": runway_forecast.p10_date.isoformat(),
            "p50_date": runway_forecast.p50_date.isoformat(),
            "p90_date": runway_forecast.p90_date.isoformat(),
            "p10_days": runway_forecast.p10_days,
            "p50_days": runway_forecast.p50_days,
            "p90_days": runway_forecast.p90_days,
        },
        "next_action": "recommendation",
        "current_agent": "forecast",
        "messages": [AIMessage(content=forecast_summary)],
    }


@traced("recommendation_node", tags=["recommendation", "advice"])
def recommendation_node(state: GlobalState) -> Dict[str, Any]:
    """
    Recommendation Node: Generates actionable financial advice.
    """
    metrics = state.get("computed_metrics", {})
    forecast_results = state.get("forecast_results", {})
    runway_forecast_data = state.get("runway_forecast", {})
    
    # Reconstruct runway forecast object
    runway_forecast = CashRunwayForecast(
        p10_date=datetime.fromisoformat(runway_forecast_data.get("p10_date", datetime.now().isoformat())),
        p50_date=datetime.fromisoformat(runway_forecast_data.get("p50_date", datetime.now().isoformat())),
        p90_date=datetime.fromisoformat(runway_forecast_data.get("p90_date", datetime.now().isoformat())),
        p10_days=runway_forecast_data.get("p10_days", 180),
        p50_days=runway_forecast_data.get("p50_days", 365),
        p90_days=runway_forecast_data.get("p90_days", 540),
        model_accuracy=0.8,
        assumptions={},
    )
    
    # Generate recommendations
    recommendations = generate_recommendations(
        burn_metrics={"metrics": BurnMetrics(**metrics)},
        forecast_results=forecast_results,
        runway_forecast=runway_forecast,
    )
    
    # Format recommendations
    rec_summary = "💡 **Recommendations**\n\n"
    
    if not recommendations:
        rec_summary += "✅ No critical issues detected. Continue monitoring.\n"
    else:
        for rec in recommendations:
            rec_summary += f"**{rec['priority']} - {rec['title']}**\n"
            rec_summary += f"• {rec['description']}\n"
            rec_summary += f"• Actions: {', '.join(rec['suggested_actions'][:2])}\n"
            rec_summary += f"• Impact: {rec['impact_estimate']}\n\n"
    
    return {
        "recommendations": recommendations,
        "next_action": "end",
        "current_agent": "recommendation",
        "messages": [AIMessage(content=rec_summary)],
    }


# ============================================================================
# Router Logic
# ============================================================================

def route_from_supervisor(state: GlobalState) -> str:
    """
    Conditional edge routing from supervisor to the appropriate agent.
    """
    next_action = state.get("next_action", "end")
    
    if next_action == "scenario":
        return "scenario"
    elif next_action == "burn":
        return "burn"
    elif next_action == "forecast":
        return "forecast"
    elif next_action == "recommendation":
        return "recommendation"
    else:
        return "end"


def after_spoke(state: GlobalState) -> str:
    """
    Spoke agents always route back to supervisor.
    """
    return "supervisor"


def should_recompute(state: GlobalState) -> bool:
    """
    Check if recomputation is needed.
    """
    return state.get("requires_recompute", False)


# ============================================================================
# Graph Construction
# ============================================================================

def build_ai_cfo_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """
    Build the LangGraph workflow.
    """
    # Initialize graph
    builder = StateGraph(GlobalState)
    
    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("scenario", scenario_node)
    builder.add_node("burn", burn_expense_node)
    builder.add_node("forecast", forecast_node)
    builder.add_node("recommendation", recommendation_node)
    
    # Add edges
    builder.add_edge("supervisor", "scenario")
    builder.add_edge("supervisor", "burn")
    builder.add_edge("supervisor", "forecast")
    builder.add_edge("supervisor", "recommendation")
    builder.add_edge("supervisor", "end")
    
    # Spoke agents always return to supervisor
    builder.add_edge("scenario", "supervisor")
    builder.add_edge("burn", "supervisor")
    builder.add_edge("forecast", "supervisor")
    builder.add_edge("recommendation", "supervisor")
    builder.add_edge("end", END)
    
    # Set entry point
    builder.set_entry_point("supervisor")
    
    # Conditional routing from supervisor
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "scenario": "scenario",
            "burn": "burn",
            "forecast": "forecast",
            "recommendation": "recommendation",
            "end": "end",
        }
    )
    
    # Compile with checkpointer
    if checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
    else:
        graph = builder.compile()
    
    return graph


# ============================================================================
# Graph Execution Helper
# ============================================================================

def run_ai_cfo(
    user_input: str,
    thread_id: str,
    state: Optional[GlobalState] = None,
) -> Dict[str, Any]:
    """
    Run the AI CFO graph for a single user interaction.
    """
    # Get checkpointer
    checkpointer = get_checkpointer()
    
    # Build graph
    graph = build_ai_cfo_graph(checkpointer)
    
    # Prepare initial state
    if state is None:
        state = {
            "messages": [],
            "cash_balance": 1000000,
            "monthly_revenue": 100000,
            "computed_metrics": None,
            "scenario_overrides": {},
            "active_scenario": None,
            "scenario_history": [],
            "forecast_results": None,
            "runway_forecast": None,
            "recommendations": [],
            "assumptions_ledger": [],
            "next_action": "end",
            "requires_recompute": False,
            "current_agent": "",
            "error_state": None,
            "transactions_df": None,
        }
    
    # Add user message
    state["messages"].append(HumanMessage(content=user_input))
    
    # Execute graph
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run with tracing context
    with tracing_context(thread_id=thread_id, metadata={"source": "direct_call"}):
        result = graph.invoke(state, config)
    
    return result


# ============================================================================
# Mock Data Generation for Testing
# ============================================================================

def generate_mock_transactions(
    start_date: datetime = None,
    months: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate mock transaction data for testing.
    """
    import pandas as pd
    import numpy as np
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=months * 30)
    
    np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=months, freq='M')
    categories = ['salary', 'rent', 'software', 'marketing', 'other']
    weights = [0.4, 0.2, 0.15, 0.15, 0.1]
    
    data = []
    for date in dates:
        for _ in range(np.random.randint(10, 20)):
            category = np.random.choice(categories, p=weights)
            amount = -abs(np.random.normal(
                loc={'salary': 5000, 'rent': 3000, 'software': 1000, 'marketing': 2000, 'other': 500}.get(category, 1000),
                scale={'salary': 1000, 'rent': 500, 'software': 300, 'marketing': 800, 'other': 200}.get(category, 300)
            ))
            one_time = np.random.random() < 0.05  # 5% one-time
            
            data.append({
                'date': date + timedelta(days=np.random.randint(0, 28)),
                'amount': amount,
                'type': 'expense',
                'category': category,
                'one_time': one_time,
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)
    return df


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the graph
    import pandas as pd
    import numpy as np
    
    print("🚀 Testing AI CFO LangGraph...")
    
    # Generate mock data
    transactions = generate_mock_transactions()
    cash_balance = 1200000
    monthly_revenue = 85000
    
    # Create initial state
    initial_state = {
        "messages": [],
        "cash_balance": cash_balance,
        "monthly_revenue": monthly_revenue,
        "computed_metrics": None,
        "scenario_overrides": {},
        "active_scenario": None,
        "scenario_history": [],
        "forecast_results": None,
        "runway_forecast": None,
        "recommendations": [],
        "assumptions_ledger": [],
        "next_action": "end",
        "requires_recompute": False,
        "current_agent": "",
        "error_state": None,
        "transactions_df": transactions,
    }
    
    # Test queries
    queries = [
        "What is our current burn rate?",
        "What if we hire 2 more engineers?",
        "Forecast our runway for the next 12 months",
        "What recommendations do you have?",
    ]
    
    for query in queries:
        print(f"\n--- User: {query} ---")
        result = run_ai_cfo(
            user_input=query,
            thread_id="test_thread",
            state=initial_state,
        )
        
        # Get last message
        if result.get("messages"):
            last_msg = result["messages"][-1]
            print(f"AI: {last_msg.content[:500]}...")
        
        # Print metrics if available
        if result.get("computed_metrics"):
            metrics = result["computed_metrics"]
            print(f"Metrics: Burn=${metrics.get('net_burn', 0):,.0f}/mo, Runway={metrics.get('cash_runway_months', 0):.1f}mo")