"""
LangSmith observability - Silently disabled if not configured.
"""

import os
import logging
from typing import Optional
from contextlib import contextmanager
from functools import wraps
from contextvars import ContextVar

# Silence LangSmith logs
logging.getLogger('langsmith').setLevel(logging.WARNING)

_current_thread_id: ContextVar[Optional[str]] = ContextVar("thread_id", default=None)
_current_metadata: ContextVar[dict] = ContextVar("metadata", default={})
_is_tracing_enabled = False


def setup_langsmith() -> bool:
    """Initialize LangSmith if configured."""
    global _is_tracing_enabled
    
    api_key = os.getenv('LANGSMITH_API_KEY')
    tracing_enabled = os.getenv('LANGSMITH_TRACING', 'false').lower() == 'true'
    
    if not tracing_enabled or not api_key:
        _is_tracing_enabled = False
        return False
    
    try:
        from langsmith import Client
        Client(api_key=api_key)
        _is_tracing_enabled = True
        return True
    except Exception:
        _is_tracing_enabled = False
        return False


def is_tracing_enabled() -> bool:
    return _is_tracing_enabled


@contextmanager
def tracing_context(thread_id: str, metadata: dict = None, tags: list = None):
    """
    Silent tracing context - accepts only thread_id, metadata, and tags.
    """
    if not _is_tracing_enabled:
        yield
        return
    
    token = _current_thread_id.set(thread_id)
    if metadata:
        token_meta = _current_metadata.set(metadata)
    try:
        yield
    finally:
        _current_thread_id.reset(token)
        if metadata:
            _current_metadata.reset(token_meta)


def traced(name: str = None, tags: list = None):
    """Silent tracing decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _is_tracing_enabled:
                return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_assumption(*args, **kwargs):
    """Silent assumption logging."""
    pass


def log_metric(*args, **kwargs):
    """Silent metric logging."""
    pass


def trace_chat_completion():
    """Silent chat completion tracing."""
    return None


def get_client():
    """Silent client getter."""
    return None