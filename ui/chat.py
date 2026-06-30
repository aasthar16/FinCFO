"""
Chat UI component with history.
"""

import streamlit as st
from typing import List, Dict, Any, Callable
from datetime import datetime
from user_auth import get_user_chats, save_user_chat, delete_user_chat


def render_chat_history():
    """
    Render chat history in sidebar.
    """
    if "user" not in st.session_state or not st.session_state.user:
        return
    
    user = st.session_state.user
    email = user["email"]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Chat History")
    
    chats = get_user_chats(email)
    
    if not chats:
        st.sidebar.caption("No chats yet")
        return
    
    chats = sorted(chats, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    for chat in chats:
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            if st.button(
                f"💬 {chat['name']}",
                key=f"chat_{chat['name']}",
                use_container_width=True
            ):
                loaded_messages = chat.get("messages", [])
                valid_messages = []
                for msg in loaded_messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        valid_messages.append(msg)
                st.session_state.messages = valid_messages
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"delete_{chat['name']}"):
                delete_user_chat(email, chat["name"])
                st.rerun()


def save_current_chat():
    """
    Save current chat to user's history.
    """
    if "user" not in st.session_state or not st.session_state.user:
        return
    
    if not st.session_state.messages:
        return
    
    email = st.session_state.user["email"]
    
    chat_name = "New Chat"
    for msg in st.session_state.messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            chat_name = content[:30] + ("..." if len(content) > 30 else "")
            break
    
    save_user_chat(email, chat_name, st.session_state.messages)


def render_chat(messages: List[Dict[str, Any]], on_submit: Callable):
    """
    Render the chat interface with messages and input.
    """
    # Display chat messages
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
        else:
            role = "user" if hasattr(message, 'type') and message.type == "human" else "assistant"
            content = message.content if hasattr(message, 'content') else str(message)
        
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
    
    # Chat input - NO key to avoid duplicates
    prompt = st.chat_input("Ask about your finances...")
    if prompt:
        on_submit(prompt)
        
def render_quick_actions(on_action: Callable):
    """
    Render quick action buttons.
    """
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    quick_queries = [
        ("💰 Burn Rate", "What's our burn rate?"),
        ("📈 Forecast", "What does our runway look like?"),
        ("🧪 Scenario", "What if we hire 2 engineers?"),
        ("💡 Recommendations", "What recommendations do you have?"),
    ]
    
    for label, query in quick_queries:
        if st.sidebar.button(label, use_container_width=True, key=f"qa_{label}"):
            on_action(query)