"""
FinCFO: Autonomous Financial Intelligence Platform
Main Streamlit Application
"""

import streamlit as st
from datetime import datetime
import logging
import pandas as pd
import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from settings import settings
from config.database import get_checkpointer, verify_tables
from config.langsmith import tracing_context, setup_langsmith
from builder import build_ai_cfo_graph
from ui import render_startup_profile, startup_profile_changed
from ui.chat import render_chat, render_quick_actions, render_chat_history, save_current_chat
from user_auth import render_login_signup, get_user_chats
from utils.helpers import generate_mock_transactions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger('config.database').setLevel(logging.WARNING)
logging.getLogger('config.langsmith').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

st.set_page_config(page_title="FinCFO", page_icon="💰", layout="wide")


def get_default_profile():
    return {"name": "", "stage": "Seed", "currency": "USD", "industry": None, "country": None, "founded_date": None}


def init_session_state():   
    """Initialize Streamlit session state."""
    if "widget_counter" not in st.session_state:
        st.session_state.widget_counter = 0
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        checkpointer = get_checkpointer()
        st.session_state.graph = build_ai_cfo_graph(checkpointer)
    if "startup_profile" not in st.session_state:
        st.session_state.startup_profile = get_default_profile()
    if "state" not in st.session_state:
        # change this , no unecessary transactions
        transactions = generate_mock_transactions(months=6)
        transactions_data = transactions.to_dict('records') if transactions is not None else []
        st.session_state.state = {
            "messages": [], "startup_profile": st.session_state.startup_profile,
            "cash_balance": 1200000, "monthly_revenue": 85000,
            "computed_metrics": None, "scenario_overrides": {}, "active_scenario": None,
            "scenario_history": [], "forecast_results": None, "runway_forecast": None,
            "recommendations": [], "assumptions_ledger": [], "next_action": None,
            "requires_recompute": False, "current_agent": "", "error_state": None,
            "_loop_count": 0, "scenario_processed": False, "transactions_data": transactions_data,
            "transactions_data": [],
            "raw_files": [],  # ← NEW: Store uploaded files here
            "parsing_status": "no_files",  # ← NEW: Track parsing status
        }
    if "langsmith_initialized" not in st.session_state:
        setup_langsmith()
        st.session_state.langsmith_initialized = True
    if "user" not in st.session_state:
        st.session_state.user = None


def parse_count(user_input: str) -> int:
    """Extract count of people to hire from user input."""
    patterns = [
        r'(?:hire|add|recruit)\s*(\d+)',
        r'(\d+)\s*(?:more|additional|extra)',
        r'(\d+)\s*(?:people|ppl|persons?|employees?|staff|members?)',
        r'another\s+(\d+)',
    ]
    for p in patterns:
        m = re.search(p, user_input)
        if m:
            return int(m.group(1))
    return 1


def parse_salary(user_input: str) -> float:
    """Extract salary from user input. Returns None if not found."""
    patterns = [
        r'(\d{3,})\s*(?:/month|/mo|per\s*month|monthly)',
        r'(?:salary|pay)\s*(?:of|is)?\s*\$?\s*(\d{3,})',
        r'\$\s*(\d{3,})',
        r'at\s+\$?\s*(\d{3,})',
        r'having\s+\$?\s*(\d{3,})',
        r'with\s+\$?\s*(\d{3,})',
        r'(\d{4,})\s*(?:each|per\s*person)',
        r'(\d{3,})\s*(?:dollars?|usd|bucks)',
    ]
    for p in patterns:
        m = re.search(p, user_input)
        if m:
            return float(m.group(1).replace(',', ''))
    return None


def parse_role(user_input: str) -> str:
    """Extract role from user input."""
    role_map = {
        "manager": "manager", "designer": "designer", "engineer": "engineer",
        "developer": "developer", "dev": "developer", "sales": "salesperson",
        "marketing": "marketer", "hr": "HR specialist", "accountant": "accountant",
    }
    for key, value in role_map.items():
        if key in user_input:
            return value
    return "employee"

def get_cumulative_hiring_cost() -> float:
    """
    Calculate total monthly cost of ALL people hired so far in the conversation.
    """
    ui_msgs = st.session_state.get("messages", [])
    user_msgs = []
    for msg in ui_msgs:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg["content"].lower().strip()
            user_msgs.append(content)
    
    # Include all messages EXCEPT current (last one)
    previous_msgs = user_msgs[:-1] if len(user_msgs) > 1 else []
    
    total_monthly_cost = 0
    
    for content in previous_msgs:
        hiring_keywords = ["hire", "hiring", "add", "recruit", "people", "ppl", "employee", "staff"]
        if not any(w in content for w in hiring_keywords):
            continue
        
        count = parse_count(content)
        salary = parse_salary(content)
        
        # If no salary in this message, use the last known salary
        if not salary:
            # Look back for salary in earlier messages
            for earlier_content in previous_msgs:
                s = parse_salary(earlier_content)
                if s:
                    salary = s
                    break
        
        if not salary:
            salary = 5000  # default
        
        total_monthly_cost += count * salary
    
    return total_monthly_cost

def is_addition_query(user_input: str) -> bool:
    """Check if query is adding to previous context."""
    addition_words = [
        "more", "another", "also", "additional", "extra",
        "same salary", "same pay", "same rate", "as well", "too",
        "on top", "plus", "with them", "along with"
    ]
    return any(w in user_input for w in addition_words)


def get_last_hiring_context() -> dict:
    """
    Get the last hiring context from conversation.
    Uses ONLY session_state messages (single source of truth).
    Excludes current query.
    """
    result = {"salary": None, "role": "employee", "total_count": 0}
    
    # Use ONLY session_state messages
    ui_msgs = st.session_state.get("messages", [])
    
    # Get only user messages, EXCLUDE the last one (current query)
    user_msgs = []
    for msg in ui_msgs:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg["content"].lower().strip()
            user_msgs.append(content)
    
    # Remove current query (last message)
    previous_msgs = user_msgs[:-1] if len(user_msgs) > 1 else []
    
    logger.info(f"📝 Previous user messages for context: {len(previous_msgs)}")
    for i, c in enumerate(previous_msgs):
        logger.info(f"  [{i}] {c[:80]}")
    
    # Process each previous message
    for content in previous_msgs:
        # Only count hiring-related messages
        hiring_keywords = ["hire", "hiring", "add", "recruit", "people", "ppl", "employee", "staff"]
        if not any(w in content for w in hiring_keywords):
            logger.info(f"  ⏭️ Skipping (not hiring): {content[:50]}")
            continue
        
        # Extract salary (keep latest)
        s = parse_salary(content)
        if s:
            result["salary"] = s
            logger.info(f"  💰 Salary: ${s:,.0f} from: '{content[:50]}'")
        
        # Extract role (keep latest non-employee)
        r = parse_role(content)
        if r and r != "employee":
            result["role"] = r
            logger.info(f"  👤 Role: {r} from: '{content[:50]}'")
        
        # Extract count and ACCUMULATE
        c = parse_count(content)
        if c:
            result["total_count"] += c
            logger.info(f"  🔢 Count: +{c} → running total: {result['total_count']} from: '{content[:50]}'")
    
    logger.info(f"📊 FINAL CONTEXT: salary=${result['salary']}, role={result['role']}, total_count={result['total_count']}")
    return result


def generate_dynamic_response(prompt: str, state: Dict[str, Any]) -> str:
    """
    Generate a dynamic, contextual response based on the user's query and current state.
    """
    user_input = prompt.lower()
    user_input = re.sub(r'(\d+)\s*([a-zA-Z])', r'\1 \2', user_input)
    user_input = user_input.replace("ppl", "people").replace("aat", "at")
    
    metrics = state.get("computed_metrics", {}) or {}
    runway = state.get("runway_forecast", {}) or {}
    recommendations = state.get("recommendations", []) or []
    
    response_parts = []
    
    # ================================================================
    # HIRING / SCENARIO QUERIES
    # ================================================================
    hiring_keywords = [
        "hire", "hiring", "add", "recruit", "employee", "employees",
        "people", "person", "more", "another", "ppl", "staff", "member"
    ]
    is_hiring = any(w in user_input for w in hiring_keywords)
    
    if is_hiring:
        response_parts.append("### 🧪 Hiring Analysis\n")
        
        # Get previous context (excludes current query)
        prev = get_last_hiring_context()
        
        # Parse current query
        count = parse_count(user_input)
        role = parse_role(user_input)
        salary = parse_salary(user_input)
        
        # Decide: addition or fresh scenario?
        is_addition = is_addition_query(user_input)
        
        # --- Resolve salary ---
        if salary:
            final_salary = salary
            logger.info(f"💰 Using explicit salary: ${final_salary:,.0f}")
        elif is_addition and prev["salary"]:
            final_salary = prev["salary"]
            logger.info(f"💰 Using previous salary: ${final_salary:,.0f}")
        else:
            final_salary = 5000
            logger.info(f"💰 Using default salary: ${final_salary:,.0f}")
        
        # --- Resolve role ---
        if role == "employee" and is_addition and prev["role"] != "employee":
            final_role = prev["role"]
            logger.info(f"👤 Using previous role: {final_role}")
        else:
            final_role = role
            logger.info(f"👤 Using explicit/default role: {final_role}")
        
        # --- Calculate costs ---
        monthly_cost = count * final_salary
        annual_cost = monthly_cost * 12
        
        # --- Display header ---
        if is_addition and prev["total_count"] > 0:
            total = prev["total_count"] + count
            response_parts.append(
                f"**➕ Adding {count} more {final_role}{'s' if count > 1 else ''}** "
                f"at ${final_salary:,.0f}/month each"
            )
            response_parts.append(
                f"*(Previously hired: {prev['total_count']} | "
                f"New: {count} | Total: {total} {final_role}s)*\n"
            )
        else:
            response_parts.append(
                f"**👥 Hiring {count} {final_role}{'s' if count > 1 else ''}** "
                f"at ${final_salary:,.0f}/month each\n"
            )
        
        # --- Cost details ---
        response_parts.append(f"- **Monthly Cost:** ${monthly_cost:,.0f}")
        response_parts.append(f"- **Annual Cost:** ${annual_cost:,.0f}")
        
        # --- Impact on metrics ---
        if metrics:
            # Get base metrics from state
            base_burn = metrics.get('net_burn', 0)
            cash = metrics.get('cash_balance', 0)
            
            # Calculate cumulative cost of ALL previous hires (excluding current query)
            previous_hires_cost = get_cumulative_hiring_cost()
            
            # Current query cost
            current_hire_cost = count * final_salary
            
            # Total cost including all hires
            total_hiring_cost = previous_hires_cost + current_hire_cost
            
            # New burn = base burn + ALL hiring costs
            new_burn = base_burn + total_hiring_cost
            
            # Current burn = base burn + previous hires cost (before this query)
            current_burn_with_previous = base_burn + previous_hires_cost
            
            response_parts.append(f"- **Base Net Burn:** ${base_burn:,.0f}/month")
            
            if previous_hires_cost > 0:
                response_parts.append(f"- **Previous Hires Cost:** ${previous_hires_cost:,.0f}/month")
                response_parts.append(f"- **Current Burn (with previous hires):** ${current_burn_with_previous:,.0f}/month")
            
            response_parts.append(f"- **This Hire Cost:** ${current_hire_cost:,.0f}/month")
            response_parts.append(f"- **New Net Burn (all hires):** ${new_burn:,.0f}/month")
            response_parts.append(f"- **Increase from base:** +{(total_hiring_cost/base_burn*100):.1f}%\n")
            
            # --- Runway impact ---
            if base_burn > 0:
                # Runway before any hires
                original_runway = cash / base_burn
                # Runway with previous hires (current state)
                current_runway = cash / current_burn_with_previous if current_burn_with_previous > 0 else 0
                # Runway after this hire
                new_runway = cash / new_burn if new_burn > 0 else 0
                
                response_parts.append(f"**⏱️ Runway Impact:**")
                
                if previous_hires_cost > 0:
                    response_parts.append(f"- Original (no hires): **{original_runway:.1f} months**")
                    response_parts.append(f"- Current (with previous hires): **{current_runway:.1f} months**")
                    response_parts.append(f"- After this hire: **{new_runway:.1f} months**")
                    response_parts.append(f"- Change from current: **-{current_runway - new_runway:.1f} months**\n")
                else:
                    response_parts.append(f"- Before hiring: **{original_runway:.1f} months**")
                    response_parts.append(f"- After hiring: **{new_runway:.1f} months**")
                    response_parts.append(f"- Reduction: **-{original_runway - new_runway:.1f} months**\n")
                
                # --- Severity assessment ---
                if new_runway < 3:
                    response_parts.append(f"🚨 **CRITICAL:** Only {new_runway:.1f} months! Immediate action required!")
                elif new_runway < 6:
                    response_parts.append(f"🔴 **URGENT:** {new_runway:.1f} months. Start fundraising now.")
                elif new_runway < 12:
                    response_parts.append(f"⚠️ **WARNING:** {new_runway:.1f} months. Plan fundraising accordingly.")
                else:
                    response_parts.append(f"✅ **HEALTHY:** {new_runway:.1f} months runway.")
        return "\n".join(response_parts)
    
    # ================================================================
    # BURN RATE QUERIES
    # ================================================================
    if any(w in user_input for w in ["burn", "expense", "spending", "cost", "rate"]):
        if metrics:
            response_parts.append("### 🔥 Burn Rate Analysis\n")
            
            gross_burn = metrics.get('gross_burn', 0)
            net_burn = metrics.get('net_burn', 0)
            net_burn_avg = metrics.get('net_burn_3m_avg', net_burn)
            recurring = metrics.get('recurring_expenses', 0)
            one_time = metrics.get('one_time_expenses', 0)
            revenue = metrics.get('monthly_revenue', 0)
            
            response_parts.append(f"- **Gross Burn:** ${gross_burn:,.0f}/month")
            response_parts.append(f"- **Net Burn:** ${net_burn:,.0f}/month")
            response_parts.append(f"- **3-Month Avg:** ${net_burn_avg:,.0f}/month")
            response_parts.append(f"- **Recurring Expenses:** ${recurring:,.0f}/month")
            
            if one_time > 0:
                response_parts.append(f"- **One-Time Expenses:** ${one_time:,.0f}")
            
            response_parts.append(f"- **Monthly Revenue:** ${revenue:,.0f}/month")
            
            if revenue > 0 and net_burn > revenue:
                gap = net_burn - revenue
                response_parts.append(f"\n⚠️ **Burn exceeds revenue by ${gap:,.0f}/month**")
            elif revenue > 0:
                response_parts.append(f"\n✅ Revenue covers your burn rate.")
            
            return "\n".join(response_parts)
    
    # ================================================================
    # RUNWAY QUERIES
    # ================================================================
    if any(w in user_input for w in ["runway", "how long", "survive", "forecast"]):
        if runway:
            p50_months = runway.get('p50_days', 0) // 30
            p10_months = runway.get('p10_days', 0) // 30
            p90_months = runway.get('p90_days', 0) // 30
            p50_date = runway.get('p50_date', 'N/A')[:10]
            
            response_parts.append("### ✈️ Runway Forecast\n")
            response_parts.append(f"- **Expected (P50):** {p50_months} months (until {p50_date})")
            response_parts.append(f"- **Pessimistic (P10):** {p10_months} months")
            response_parts.append(f"- **Optimistic (P90):** {p90_months} months\n")
            
            if p50_months < 3:
                response_parts.append(f"🚨 **CRITICAL:** Only {p50_months} months!")
            elif p50_months < 6:
                response_parts.append(f"🔴 **URGENT:** {p50_months} months. Fundraise now.")
            elif p50_months < 12:
                response_parts.append(f"🟠 **WARNING:** {p50_months} months. Plan ahead.")
            elif p50_months < 18:
                response_parts.append(f"🟡 **FAIR:** {p50_months} months.")
            else:
                response_parts.append(f"✅ **HEALTHY:** {p50_months} months.")
            
            return "\n".join(response_parts)
        
        elif metrics:
            cash = metrics.get('cash_balance', 0)
            net_burn = metrics.get('net_burn', 0)
            if net_burn > 0:
                runway_months = cash / net_burn
                response_parts.append("### ✈️ Runway Estimate\n")
                response_parts.append(f"- **Cash:** ${cash:,.0f}")
                response_parts.append(f"- **Net Burn:** ${net_burn:,.0f}/month")
                response_parts.append(f"- **Runway:** **{runway_months:.1f} months**")
            
            return "\n".join(response_parts)
    
    # ================================================================
    # RECOMMENDATION QUERIES
    # ================================================================
    if any(w in user_input for w in ["recommend", "advice", "suggest", "improve"]):
        if recommendations:
            response_parts.append("### 💡 Top Recommendations\n")
            
            for i, rec in enumerate(recommendations[:3], 1):
                if isinstance(rec, dict):
                    priority = rec.get('priority', 'MEDIUM')
                    title = rec.get('title', '')
                    desc = rec.get('description', '')
                    actions = rec.get('suggested_actions', [])
                    impact = rec.get('impact_estimate', '')
                    
                    emoji = "🔴" if priority == "HIGH" else "🟠" if priority == "MEDIUM" else "🟢"
                    
                    response_parts.append(f"**{i}. {emoji} {title}**")
                    if desc:
                        response_parts.append(f"  *{desc}*")
                    if actions:
                        response_parts.append(f"  **Actions:** {', '.join(actions[:3])}")
                    if impact:
                        response_parts.append(f"  **Impact:** {impact}")
                    response_parts.append("")
            
            return "\n".join(response_parts)
        else:
            response_parts.append("### 💡 Recommendations\n")
            response_parts.append("I need to analyze your financial data first.")
            response_parts.append("Try asking about your **burn rate** or **runway** first!")
            return "\n".join(response_parts)
    
    # ================================================================
    # DEFAULT: Financial Summary
    # ================================================================
    response_parts.append("### 📊 Financial Summary\n")
    
    if metrics:
        cash = metrics.get('cash_balance', 0)
        net_burn = metrics.get('net_burn', 0)
        revenue = metrics.get('monthly_revenue', 0)
        
        response_parts.append(f"- **Cash Balance:** ${cash:,.0f}")
        response_parts.append(f"- **Net Burn:** ${net_burn:,.0f}/month")
        response_parts.append(f"- **Monthly Revenue:** ${revenue:,.0f}/month")
        
        if net_burn > 0:
            response_parts.append(f"- **Runway:** {cash/net_burn:.1f} months")
    
    if runway:
        response_parts.append(f"- **Forecast Runway:** {runway.get('p50_days', 0)//30} months")
    
    response_parts.append(f"\n💬 **Try asking me:**")
    response_parts.append(f"- *What's our burn rate?*")
    response_parts.append(f"- *What does our runway look like?*")
    response_parts.append(f"- *What if I hire 3 engineers at $8000/month?*")
    response_parts.append(f"- *What if I add 2 more at the same salary?*")
    response_parts.append(f"- *What recommendations do you have?*")
    
    return "\n".join(response_parts)


def handle_query(prompt: str):
    """Handle user query - V2 Agentic."""
    logger.info(f"📝 Processing: {prompt[:80]}...")
    
    # Add user message to UI
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    
    # Add HumanMessage to graph state
    if "state" in st.session_state:
        st.session_state.state["messages"].append(HumanMessage(content=prompt))
    
    if len(st.session_state.messages) == 1:
        save_current_chat()
    
    with st.spinner("🤔 FinCFO is analyzing..."):
        try:
            with tracing_context(
                thread_id=st.session_state.thread_id,
                metadata={"prompt": prompt[:100]},
                tags=["user-query"],
            ):
                config = {
                    "configurable": {"thread_id": st.session_state.thread_id},
                    "recursion_limit": 50
                }
                
                # Invoke graph (now uses V2 supervisor)
                result = st.session_state.graph.invoke(
                    st.session_state.state,
                    config
                )
                
                st.session_state.state = result
                
                # Extract assistant response from messages
                if result.get("messages"):
                    last_msg = result["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        content = last_msg.content

                        # check if we need to remove this line
                    elif isinstance(last_msg, dict):
                        content = last_msg.get("content", "")
                    else:
                        content = str(last_msg)
                    
                    if content:
                        # cleaned = clean_response(content)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": content
                        })
                
                save_current_chat()
                st.rerun()
                
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}"
            })
            st.error(f"❌ Error: {str(e)}")



def clean_response(content: str) -> str:
    """Clean and normalize LLM response for proper markdown rendering."""
    if not content:
        return content
    
    content = content.strip()
    
    # Fix: Remove broken bold markers that have spaces
    content = re.sub(r'\*\*\s+', '**', content)
    content = re.sub(r'\s+\*\*', '**', content)
    
    # Fix: Normalize bold formatting - ensure proper **text** pattern
    # Fix patterns like "**$85,000/month**" that are correct
    # But fix patterns like "** $85,000/month **" (spaces inside)
    content = re.sub(r'\*\*\s+([^*]+?)\s+\*\*', r'**\1**', content)
    
    # Fix: Remove empty bold markers
    content = content.replace('****', '')
    
    # Fix: Ensure single newlines for markdown
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Fix: Remove trailing/leading whitespace from each line
    lines = content.split('\n')
    lines = [line.strip() for line in lines]
    content = '\n'.join(lines)
    
    return content


def main():
    """Main application."""
    init_session_state()
    
    is_logged_in = render_login_signup()
    if not is_logged_in:
        st.title("💰 FinCFO")
        st.markdown("### Please login or sign up to continue")
        st.markdown("""
        **Features:**
        - 💬 Chat with FinCFO about your finances
        - 📊 Upload CSV/Excel data for analysis
        - 💾 Save and access chat history
        - 📈 Get financial insights and recommendations
        """)
        st.stop()
    
    with st.sidebar:
        new_profile = render_startup_profile(st.session_state.startup_profile)
        if startup_profile_changed(st.session_state.startup_profile, new_profile):
            st.session_state.startup_profile = new_profile
            st.session_state.state["startup_profile"] = new_profile
            if st.session_state.user:
                from user_auth import save_user_profile
                save_user_profile(st.session_state.user["email"], new_profile)
            st.rerun()
        render_chat_history()
        render_quick_actions(handle_query)
    
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        st.title("💰 FinCFO")
        if st.session_state.startup_profile.get("name"):
            st.caption(f"{st.session_state.startup_profile['name']} • {st.session_state.startup_profile['stage']} • {st.session_state.startup_profile['currency']}")
    with col2:
        if st.button("💾 Save", use_container_width=True):
            if st.session_state.messages:
                save_current_chat()
                st.success("✅ Saved!")
                st.rerun()
    with col3:
        if st.button("🆕 New", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if "state" in st.session_state:
                st.session_state.state["messages"] = []
                st.session_state.state["computed_metrics"] = None
                st.session_state.state["forecast_results"] = None
                st.session_state.state["runway_forecast"] = None
                st.session_state.state["recommendations"] = []
                st.session_state.state["next_action"] = None
                st.session_state.state["active_scenario"] = None
                st.session_state.state["scenario_overrides"] = {}
                st.session_state.state["_loop_count"] = 0
            st.rerun()
    
    st.divider()
    render_chat(st.session_state.messages, handle_query)


if __name__ == "__main__":
    main()