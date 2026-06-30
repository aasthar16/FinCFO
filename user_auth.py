"""
User authentication using PostgreSQL.
"""

import streamlit as st
import hashlib
from config.database import (
    create_user_in_db,
    get_user_from_db,
    save_chat_to_db,
    get_user_chats_from_db,
    delete_chat_from_db,
    save_user_profile_to_db,
    get_user_profile_from_db
)
import logging
logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(email: str, name: str, password: str) -> bool:
    return create_user_in_db(email, name, hash_password(password))


def verify_user(email: str, password: str):
    user = get_user_from_db(email)
    if not user:
        return None
    if user['password_hash'] == hash_password(password):
        profile = get_user_profile_from_db(email)
        return {
            "email": user['email'],
            "name": user['name'],
            "created_at": user['created_at'].isoformat() if user['created_at'] else None,
            "profile": profile or {}
        }
    return None


def save_user_chat(email: str, chat_name: str, messages: list):
    """Save chat - messages should be list of dicts with role/content."""
    return save_chat_to_db(email, chat_name, messages)


def get_user_chats(email: str) -> list:
    """Get all chats for a user."""
    chats = get_user_chats_from_db(email)
    # Ensure messages are in correct format
    for chat in chats:
        if "messages" in chat:
            valid_msgs = []
            for msg in chat["messages"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    valid_msgs.append(msg)
            chat["messages"] = valid_msgs
    return chats


def delete_user_chat(email: str, chat_name: str):
    return delete_chat_from_db(email, chat_name)


def save_user_profile(email: str, profile_data: dict):
    return save_user_profile_to_db(email, profile_data)


def render_login_signup():
    """Render login/signup UI."""
    st.sidebar.markdown("### 🔐 Welcome")
    
    if "user" in st.session_state and st.session_state.user:
        user = st.session_state.user
        st.sidebar.success(f"👋 Welcome, {user['name']}!")
        if st.sidebar.button("🚪 Logout"):
            st.session_state.user = None
            st.session_state.messages = []
            st.rerun()
        return True
    
    tab1, tab2 = st.sidebar.tabs(["🔑 Login", "📝 Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if email and password:
                    user = verify_user(email, password)
                    if user:
                        st.session_state.user = user
                        if user.get('profile'):
                            st.session_state.startup_profile = user['profile']
                        # Load user's chats
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
                else:
                    st.warning("Please fill all fields")
    
    with tab2:
        with st.form("signup_form"):
            name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Sign Up"):
                if not name or not email or not password:
                    st.warning("Please fill all fields")
                elif password != confirm:
                    st.error("Passwords do not match")
                elif create_user(email, name, password):
                    st.success("✅ Account created! Please login.")
                else:
                    st.error("❌ Email already exists")
    
    return False