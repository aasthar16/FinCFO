"""
Startup profile UI component.
"""

import streamlit as st
from typing import Dict, Any
from datetime import date, datetime
from utils.file_upload import handle_file_upload


def render_startup_profile(current_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render startup profile configuration UI.
    """
    st.sidebar.markdown("### 🏢 Startup Profile")
    
    # Startup name
    name = st.sidebar.text_input(
        "Startup Name",
        value=current_profile.get("name", ""),
        key="profile_name",
        placeholder="Enter your startup name"
    )
    
    # Startup stage
    stage = st.sidebar.selectbox(
        "Stage",
        options=["Pre-seed", "Seed", "Series A", "Series B", "Series C+"],
        index=0,
        key="profile_stage"
    )
    
    # Currency
    currency = st.sidebar.selectbox(
        "Currency",
        options=["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "INR", "BRL"],
        index=0,
        key="profile_currency"
    )
    
    # File upload
    uploaded_df = handle_file_upload()
    if uploaded_df is not None:
        st.session_state.transactions_df = uploaded_df
        st.sidebar.success("✅ Data loaded successfully!")
    
    # Additional details
    with st.sidebar.expander("📋 Additional Details (Optional)"):
        industry = st.text_input(
            "Industry",
            value=current_profile.get("industry", ""),
            placeholder="e.g., SaaS, Fintech"
        )
        
        country = st.text_input(
            "Country",
            value=current_profile.get("country", ""),
            placeholder="Operating country"
        )
        
        founded_date_str = current_profile.get("founded_date")
        if founded_date_str:
            try:
                founded_date_val = datetime.fromisoformat(founded_date_str).date()
            except (ValueError, TypeError):
                founded_date_val = None
        else:
            founded_date_val = None
        
        founded_date = st.date_input(
            "Founded Date (Optional)",
            value=founded_date_val,
            key="profile_founded",
            help="When was the startup founded?"
        )
    
    # Show current profile summary
    if name:
        st.sidebar.success(f"✅ {name} ({stage})")
        st.sidebar.caption(f"💱 Currency: {currency}")
    
    # Return updated profile
    return {
        "name": name or "Unnamed Startup",
        "stage": stage,
        "currency": currency,
        "industry": industry if industry else None,
        "country": country if country else None,
        "founded_date": founded_date.isoformat() if founded_date else None,
    }


def startup_profile_changed(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    """Check if startup profile has changed."""
    if not old or not new:
        return True
    
    keys = ["name", "stage", "currency", "industry", "country", "founded_date"]
    for key in keys:
        if old.get(key) != new.get(key):
            return True
    return False