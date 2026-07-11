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
        st.session_state.state = {
            "messages": [],
            "startup_profile": st.session_state.startup_profile,
            "cash_balance": 1200000,
            "monthly_revenue": 85000,
            "computed_metrics": None,
            "scenario_overrides": {},
            "active_scenario": None,
            "scenario_history": [],
            "forecast_results": None,
            "runway_forecast": None,
            "recommendations": [],
            "assumptions_ledger": [],
            "next_action": None,
            "requires_recompute": False,
            "current_agent": "",
            "error_state": None,
            "_loop_count": 0,
            "scenario_processed": False,
            "transactions_data": [],
            "raw_files": [],
            "parsing_status": "no_files",
        }
    if "langsmith_initialized" not in st.session_state:
        setup_langsmith()
        st.session_state.langsmith_initialized = True
    if "user" not in st.session_state:
        st.session_state.user = None


def handle_query(prompt: str):
    """Handle user query - Agentic with HITL support."""
    logger.info(f"📝 Processing: {prompt[:80]}...")

    # -----------------------------
    # Add user message to UI
    # -----------------------------
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    # Add HumanMessage to graph state
    st.session_state.state["messages"].append(
        HumanMessage(content=prompt)
    )

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
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    },
                    "recursion_limit": 50,
                }

                logger.info(
                    f"Messages before graph: {len(st.session_state.state['messages'])}"
                )

                # -----------------------------------------
                # Execute graph
                # -----------------------------------------
                for event in st.session_state.graph.stream(
                    st.session_state.state,
                    config,
                ):

                    # Uncomment while debugging
                    # print(event)

                    if "__interrupt__" in event:

                        interrupt_data = event["__interrupt__"][0]["value"]

                        reason = interrupt_data.get(
                            "reason",
                            "Confirmation required",
                        )
                        context = interrupt_data.get("context", "")

                        st.warning(reason)

                        if context:
                            st.info(context)

                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button("✅ Proceed"):

                                list(
                                    st.session_state.graph.stream(
                                        None,
                                        config,
                                        interrupt_mode="resume",
                                        resume_value="proceed",
                                    )
                                )

                                st.rerun()

                        with col2:
                            if st.button("❌ Cancel"):

                                list(
                                    st.session_state.graph.stream(
                                        None,
                                        config,
                                        interrupt_mode="resume",
                                        resume_value="cancel",
                                    )
                                )

                                st.rerun()

                        return

                # -----------------------------------------
                # IMPORTANT:
                # Get FINAL graph state from checkpointer
                # -----------------------------------------
                final_state = (
                    st.session_state.graph
                    .get_state(config)
                    .values
                )

                # Keep local copy synced
                st.session_state.state = final_state

                # -----------------------------------------
                # Find latest assistant response
                # -----------------------------------------
                assistant_msg = None

                for msg in reversed(final_state["messages"]):

                    if (
                        isinstance(msg, AIMessage)
                        and not getattr(msg, "tool_calls", None)
                    ):
                        assistant_msg = msg
                        break

                if assistant_msg:

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg.content,
                        }
                    )

                save_current_chat()

                st.rerun()

        except Exception as e:

            logger.error(
                f"❌ Error: {e}",
                exc_info=True,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"❌ Error: {e}",
                }
            )

            st.error(str(e))

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