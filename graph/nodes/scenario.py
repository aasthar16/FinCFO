"""
Scenario Simulator node.
"""

import re
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import AIMessage
from graph.state import GlobalState
from config.langsmith import traced


@traced("scenario_node", tags=["scenario", "simulator"])
def scenario_node(state: GlobalState) -> Dict[str, Any]:
    """
    Scenario Simulator Node: Models what-if scenarios.
    """
    last_message = state["messages"][-1]
    user_input = last_message.content.lower()
    
    # Simple keyword extraction
    scenario_overrides = {}
    scenario_name = "default_scenario"
    
    if "hire" in user_input or "engineer" in user_input or "headcount" in user_input:
        numbers = re.findall(r'\d+', user_input)
        if numbers:
            scenario_overrides["headcount_change"] = int(numbers[0])
            scenario_overrides["avg_salary"] = 140000
            scenario_overrides["ramp_months"] = 3
            scenario_name = f"hire_{numbers[0]}_engineers"
    
    if "revenue" in user_input or "growth" in user_input:
        percentages = re.findall(r'(\d+)%', user_input)
        if percentages:
            current_revenue = state.get("monthly_revenue", 100000)
            increase = current_revenue * (int(percentages[0]) / 100)
            scenario_overrides["revenue_change"] = increase
            scenario_name = f"revenue_increase_{percentages[0]}%"
        else:
            amounts = re.findall(r'\$?(\d+[,.]?\d*)[kK]?', user_input)
            if amounts:
                amount = float(amounts[0].replace(',', ''))
                if 'k' in user_input.lower():
                    amount *= 1000
                scenario_overrides["revenue_change"] = amount
                scenario_name = "revenue_increase"
    
    if "spend" in user_input or "cut" in user_input or "reduce" in user_input or "expense" in user_input:
        amounts = re.findall(r'\$?(\d+[,.]?\d*)[kK]?', user_input)
        if amounts:
            amount = float(amounts[0].replace(',', ''))
            if 'k' in user_input.lower() or 'K' in user_input.lower():
                amount *= 1000
            scenario_overrides["one_time_expenses"] = amount
            scenario_name = "expense_reduction"
    
    # If no scenario detected, use default
    if not scenario_overrides:
        scenario_overrides = {"headcount_change": 2, "avg_salary": 140000, "ramp_months": 3}
        scenario_name = "default_scenario"
    
    return {
        "scenario_overrides": scenario_overrides,
        "active_scenario": scenario_name,
        "requires_recompute": True,
        "next_action": "burn",  # Go to burn after scenario
        "current_agent": "scenario",  # IMPORTANT: Set current_agent
        "messages": [AIMessage(content=f"🧪 Modeling scenario: {scenario_name}")],
        "assumptions_ledger": state.get("assumptions_ledger", []) + [{
            "source": "scenario_simulator",
            "scenario": scenario_name,
            "parameters": scenario_overrides,
            "timestamp": datetime.now().isoformat(),
        }],
    }