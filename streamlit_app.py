"""
AI CFO: Autonomous Financial Intelligence Platform
Streamlit Frontend with LangGraph Backend
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import logging

from ai_cfo_langgraph_skeleton import (
    run_ai_cfo,
    generate_mock_transactions,
    GlobalState,
)
from langsmith_config import tracing_context

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI CFO",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0e1117;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4a4a4a;
    }
    .recommendation-high {
        border-left: 4px solid #ff4b4b;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    .recommendation-medium {
        border-left: 4px solid #ffa500;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    .recommendation-low {
        border-left: 4px solid #00b894;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    .assumption-card {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        margin: 0.25rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "state" not in st.session_state:
        # Generate mock transactions
        transactions = generate_mock_transactions(months=6)
        
        st.session_state.state = {
            "messages": [],
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
            "next_action": "end",
            "requires_recompute": False,
            "current_agent": "",
            "error_state": None,
            "transactions_df": transactions,
        }
    
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render sidebar with startup information."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/financial-analytics.png", width=64)
        st.markdown("## AI CFO Dashboard")
        
        # Startup info
        st.markdown("### 🏢 Startup Profile")
        st.text_input("Startup Name", value="AI CFO Demo", key="startup_name")
        st.selectbox(
            "Stage",
            options=["Pre-seed", "Seed", "Series A", "Series B", "Series C+"],
            index=1,
            key="startup_stage",
        )
        st.selectbox(
            "Currency",
            options=["USD", "EUR", "GBP"],
            index=0,
            key="currency",
        )
        
        st.divider()
        
        # Thread info
        st.markdown("### 🔄 Session")
        st.caption(f"Thread ID: {st.session_state.thread_id}")
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.messages = []
            st.session_state.state = None
            st.session_state.last_result = None
            init_session_state()
            st.rerun()
        
        st.divider()
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        quick_queries = [
            "What's our burn rate?",
            "What if we hire 2 engineers?",
            "Forecast runway",
            "Show recommendations",
        ]
        for query in quick_queries:
            if st.button(query, use_container_width=True, key=f"qa_{query}"):
                process_query(query)
        
        st.divider()
        
        # Status
        st.markdown("### 📊 Status")
        if st.session_state.last_result:
            metrics = st.session_state.last_result.get("computed_metrics")
            if metrics:
                st.metric("Runway (months)", f"{metrics.get('cash_runway_months', 0):.1f}")
                st.metric("Net Burn", f"${metrics.get('net_burn', 0):,.0f}")
                st.metric("Cash", f"${metrics.get('cash_balance', 0):,.0f}")


def render_metrics_dashboard():
    """Render main metrics dashboard."""
    if not st.session_state.last_result:
        st.info("👋 Ask a question about your finances to get started.")
        return
    
    result = st.session_state.last_result
    metrics = result.get("computed_metrics")
    
    if not metrics:
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Cash Balance</div>
            <div class="metric-value">${metrics.get('cash_balance', 0):,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔥 Net Burn</div>
            <div class="metric-value">${metrics.get('net_burn_3m_avg', 0):,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Runway (P50)</div>
            <div class="metric-value">{metrics.get('cash_runway_months', 0):.1f} mo</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Burn Multiple</div>
            <div class="metric-value">{metrics.get('burn_multiple', 0):.1f}x</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional metrics
    st.markdown("### 📊 Detailed Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Gross Burn (3mo avg)", f"${metrics.get('gross_burn', 0):,.0f}")
        st.metric("Recurring Expenses", f"${metrics.get('recurring_expenses', 0):,.0f}")
        st.metric("One-Time Expenses", f"${metrics.get('one_time_expenses', 0):,.0f}")
    
    with col2:
        st.metric("Monthly Revenue", f"${metrics.get('monthly_revenue', 0):,.0f}")
        st.metric("Fully Loaded Ratio", f"{metrics.get('fully_loaded_ratio', 0):.2f}x")
    
    # Forecast section
    forecast = result.get("forecast_results")
    if forecast:
        st.markdown("### 📈 Forecast")
        st.json(forecast, expanded=False)
    
    # Recommendations
    recommendations = result.get("recommendations", [])
    if recommendations:
        st.markdown("### 💡 Recommendations")
        for rec in recommendations:
            priority_class = {
                "HIGH": "recommendation-high",
                "MEDIUM": "recommendation-medium",
                "LOW": "recommendation-low",
            }.get(rec.get("priority", "LOW"), "recommendation-low")
            
            st.markdown(f"""
            <div class="{priority_class}">
                <strong>{rec.get('priority', 'MEDIUM')} - {rec.get('title', '')}</strong>
                <p>{rec.get('description', '')}</p>
                <p><strong>Actions:</strong> {', '.join(rec.get('suggested_actions', []))}</p>
                <p><strong>Impact:</strong> {rec.get('impact_estimate', '')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Assumptions ledger
    assumptions = result.get("assumptions_ledger", [])
    if assumptions:
        st.markdown("### 📋 Assumptions Ledger")
        with st.expander("View all assumptions"):
            for assumption in assumptions[-10:]:  # Show last 10
                st.markdown(f"""
                <div class="assumption-card">
                    <strong>{assumption.get('source', 'unknown')}</strong>: 
                    {assumption.get('parameter', '')} = {assumption.get('value', '')}
                    <br><small>{assumption.get('rationale', '')}</small>
                </div>
                """, unsafe_allow_html=True)


def render_chat():
    """Render chat interface."""
    st.markdown("### 💬 AI CFO Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        if hasattr(message, 'type'):
            if message.type == "human":
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message.content)
        else:
            # Fallback for plain dict messages
            with st.chat_message("assistant"):
                st.write(message.get("content", ""))
    
    # Chat input
    if prompt := st.chat_input("Ask about your finances..."):
        process_query(prompt)


def process_query(query: str):
    """
    Process a user query through the AI CFO graph.
    """
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Show thinking state
    with st.spinner("🤔 AI CFO is analyzing..."):
        try:
            # Run graph with tracing context
            with tracing_context(
                thread_id=st.session_state.thread_id,
                metadata={
                    "startup_stage": st.session_state.get("startup_stage", "seed"),
                    "currency": st.session_state.get("currency", "USD"),
                    "source": "streamlit",
                },
            ):
                # Get current state
                current_state = st.session_state.state.copy()
                
                # Run the graph
                result = run_ai_cfo(
                    user_input=query,
                    thread_id=st.session_state.thread_id,
                    state=current_state,
                )
            
            # Update state
            st.session_state.state = result
            st.session_state.last_result = result
            
            # Add assistant message
            if result.get("messages"):
                last_msg = result["messages"][-1]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": last_msg.content,
                })
            
            # Rerun to update UI
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Error processing query: {e}", exc_info=True)


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main Streamlit application."""
    # Initialize session state
    init_session_state()
    
    # Layout
    st.markdown('<div class="main-header">💰 AI CFO</div>', unsafe_allow_html=True)
    st.caption("Autonomous Financial Intelligence for Startups")
    
    # Sidebar
    render_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_chat()
    
    with col2:
        render_metrics_dashboard()
    
    # Footer
    st.divider()
    st.caption(f"AI CFO v1.0 | Thread: {st.session_state.thread_id} | Tracing: {'✅' if os.getenv('LANGSMITH_TRACING') == 'true' else '❌'}")


if __name__ == "__main__":
    main()