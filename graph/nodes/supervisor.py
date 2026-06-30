"""
Supervisor node - routes to appropriate agents.
"""
import logging
logger = logging.getLogger(__name__)
from typing import Dict, Any
from langchain_core.messages import AIMessage
from graph.state import GlobalState
from config.langsmith import traced


@traced("supervisor_node", tags=["supervisor", "routing"])
def supervisor_node(state: GlobalState) -> Dict[str, Any]:
    """
    Supervisor Node: Routes to the appropriate agent based on user intent.
    """
    messages = state.get("messages", [])
    
    # Track loop count - READ from state, increment in return
    loop_count = state.get("_loop_count", 0) + 1
    
    # If we've looped more than 15 times, force end
    if loop_count > 15:
        logger.warning(f"Loop count {loop_count} exceeded, forcing end")
        return {
            "next_action": "end",
            "current_agent": "supervisor",
            "_loop_count": loop_count,
        }
    
    # Get the last user message
    last_user_message = None
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == "human":
            last_user_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break
    
    # If no user message found
    if not last_user_message:
        return {
            "next_action": "end",
            "current_agent": "supervisor",
            "_loop_count": loop_count,
            "messages": [AIMessage(content="How can I help you with your finances today?")],
        }
    
    user_input = last_user_message.lower()
    
    # Check state flags
    has_metrics = state.get("computed_metrics") is not None
    has_forecast = state.get("forecast_results") is not None
    has_recommendations = state.get("recommendations") and len(state.get("recommendations", [])) > 0
    has_scenario = state.get("active_scenario") is not None and state.get("scenario_overrides", {})
    
    # Get the last node that ran
    last_node = state.get("current_agent", "")
    next_action_from_state = state.get("next_action", "")
    
    logger.info(f"Supervisor: loop={loop_count}, last_node={last_node}, has_metrics={has_metrics}, has_forecast={has_forecast}, has_recs={has_recommendations}")
    logger.info(f"Supervisor: user_input='{user_input[:50]}...'")
    
    # ============================================================
    # FLOW CONTROL - Chain nodes in sequence
    # ============================================================
    
    # After scenario → go to burn
    if last_node == "scenario":
        logger.info("Supervisor: scenario done, routing to burn")
        return {
            "next_action": "burn",
            "current_agent": "supervisor",
            "_loop_count": loop_count,
            "messages": [AIMessage(content="🔥 Analyzing scenario impact on burn rate...")],
        }
    
    # After burn → go to forecast
    if last_node == "burn":
        logger.info("Supervisor: burn done, routing to forecast")
        return {
            "next_action": "forecast",
            "current_agent": "supervisor",
            "_loop_count": loop_count,
        }
    
    # After forecast → go to recommendation
    if last_node == "forecast":
        logger.info("Supervisor: forecast done, routing to recommendation")
        return {
            "next_action": "recommendation",
            "current_agent": "supervisor",
            "_loop_count": loop_count,
        }
    
    # After recommendation → end
    if last_node == "recommendation":
        logger.info("Supervisor: recommendation done, ending")
        return {
            "next_action": "end",
            "current_agent": "supervisor",
            "_loop_count": loop_count,
        }
    
    # ============================================================
    # FIRST ENTRY or AFTER SUPERVISOR - Route based on user intent
    # ============================================================
    
    # If everything is done, end
    if has_metrics and has_forecast and has_recommendations:
        logger.info("Supervisor: all done, ending")
        return {
            "next_action": "end",
            "current_agent": "supervisor",
            "_loop_count": loop_count,
            "messages": [AIMessage(content="✅ I've completed the full analysis. What else would you like to know?")],
        }
    
    # First time: no metrics yet
    if not has_metrics:
        # Route based on user's query
        if "scenario" in user_input or "what if" in user_input or "hire" in user_input:
            logger.info("Supervisor: routing to scenario (first time)")
            return {
                "next_action": "scenario",
                "current_agent": "supervisor",
                "_loop_count": loop_count,
                "messages": [AIMessage(content="🧪 Let me model that scenario for you...")],
            }
        elif "recommend" in user_input or "advice" in user_input or "suggest" in user_input:
            logger.info("Supervisor: routing to burn first (asked for recommendations)")
            return {
                "next_action": "burn",
                "current_agent": "supervisor",
                "_loop_count": loop_count,
                "messages": [AIMessage(content="🔍 Let me analyze your finances first, then I'll provide recommendations...")],
            }
        else:
            # Default: start with burn analysis
            logger.info("Supervisor: routing to burn (default first action)")
            return {
                "next_action": "burn",
                "current_agent": "supervisor",
                "_loop_count": loop_count,
                "messages": [AIMessage(content="🔍 Let me analyze your current financial position...")],
            }
    
    # Has metrics but no forecast yet
    if has_metrics and not has_forecast:
        logger.info("Supervisor: has metrics, routing to forecast")
        return {
            "next_action": "forecast",
            "current_agent": "supervisor",
            "_loop_count": loop_count,
            "messages": [AIMessage(content="📈 Now let me generate your financial forecast...")],
        }
    
    # Has metrics and forecast but no recommendations
    if has_metrics and has_forecast and not has_recommendations:
        logger.info("Supervisor: has metrics+forecast, routing to recommendations")
        return {
            "next_action": "recommendation",
            "current_agent": "supervisor",
            "_loop_count": loop_count,
            "messages": [AIMessage(content="💡 Now let me generate recommendations...")],
        }
    
    # User asks for specific things when we already have data
    if has_metrics:
        if "burn" in user_input or "expense" in user_input or "spending" in user_input:
            logger.info("Supervisor: user asked for burn, re-running burn")
            return {
                "next_action": "burn",
                "current_agent": "supervisor",
                "_loop_count": loop_count,
                "messages": [AIMessage(content="🔥 Recalculating your burn rate...")],
            }
        
        if "forecast" in user_input or "runway" in user_input or "projection" in user_input:
            if has_forecast:
                logger.info("Supervisor: forecast already available")
                return {
                    "next_action": "end",
                    "current_agent": "supervisor",
                    "_loop_count": loop_count,
                    "messages": [AIMessage(content="✅ Forecast is already available. Check the dashboard for details.")],
                }
            else:
                logger.info("Supervisor: generating forecast")
                return {
                    "next_action": "forecast",
                    "current_agent": "supervisor",
                    "_loop_count": loop_count,
                    "messages": [AIMessage(content="📈 Generating your forecast...")],
                }
        
        if "recommend" in user_input or "advice" in user_input:
            if has_recommendations:
                logger.info("Supervisor: recommendations already available")
                return {
                    "next_action": "end",
                    "current_agent": "supervisor",
                    "_loop_count": loop_count,
                    "messages": [AIMessage(content="✅ Recommendations are already available. Check the dashboard for details.")],
                }
            else:
                logger.info("Supervisor: generating recommendations")
                return {
                    "next_action": "recommendation",
                    "current_agent": "supervisor",
                    "_loop_count": loop_count,
                    "messages": [AIMessage(content="💡 Generating recommendations...")],
                }
        
        if "scenario" in user_input or "what if" in user_input or "hire" in user_input:
            logger.info("Supervisor: running scenario")
            return {
                "next_action": "scenario",
                "current_agent": "supervisor",
                "_loop_count": loop_count,
                "messages": [AIMessage(content="🧪 Modeling that scenario...")],
            }
    
    # Final fallback
    logger.warning(f"Supervisor: no matching condition, falling back to end")
    return {
        "next_action": "end",
        "current_agent": "supervisor",
        "_loop_count": loop_count,
        "messages": [AIMessage(content="I'm not sure what you'd like me to do. Try asking about your burn rate, runway, or scenarios.")],
    }