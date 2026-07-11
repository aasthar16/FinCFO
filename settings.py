"""
Application settings and configuration.
Infrastructure settings only (database, APIs, etc.)
Business config (startup stage, currency) are user inputs in the UI.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Settings:
    """Infrastructure settings - database, APIs, services."""
    
    # ===== REQUIRED (no defaults) =====
    database_url: str           # PostgreSQL connection URL
    groq_api_key: str           # Groq API key for LLM
    
    # ===== Optional with defaults =====
    # App
    app_name: str = "FinCFO"
    app_version: str = "1.0.0"
    
    # LLM
    groq_model: str = "llama-3.3-70b-versatile"
    
    # LangSmith (observability - optional)
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "ai-cfo-platform"
    langsmith_tracing: bool = True
    
    # Feature flags
    enable_tracing: bool = True
    enable_scenarios: bool = True


def get_settings() -> Settings:
    """
    Get settings from Streamlit secrets.
    """
    try:
        import streamlit as st
        
        if not hasattr(st, 'secrets'):
            raise ValueError("Streamlit secrets not found")
        
        # Check required secrets
        if 'postgres' not in st.secrets:
            raise ValueError("Missing 'postgres' section in secrets.toml")
        if 'url' not in st.secrets['postgres']:
            raise ValueError("Missing 'postgres.url' in secrets.toml")
        if 'groq' not in st.secrets:
            raise ValueError("Missing 'groq' section in secrets.toml")
        if 'api_key' not in st.secrets['groq']:
            raise ValueError("Missing 'groq.api_key' in secrets.toml")
        
        return Settings(
            database_url=st.secrets['postgres']['url'],
            groq_api_key=st.secrets['groq']['api_key'],
            groq_model=st.secrets.get('groq', {}).get('model', 'llama-3.3-70b-versatile'),
            langsmith_api_key=st.secrets.get('langsmith', {}).get('api_key'),
            langsmith_project=st.secrets.get('langsmith', {}).get('project', 'ai-cfo-platform'),
            langsmith_tracing=st.secrets.get('langsmith', {}).get('tracing', True),
        )
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        raise


settings = get_settings()