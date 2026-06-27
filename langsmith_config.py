"""
LangSmith Observability Configuration
Centralized tracing setup with graceful fallback.
"""

import os
import json
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
from functools import wraps
import logging
from contextvars import ContextVar

import langsmith
from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai

logger = logging.getLogger(__name__)

# Context variables for propagating tracing context
_current_thread_id: ContextVar[Optional[str]] = ContextVar("thread_id", default=None)
_current_metadata: ContextVar[Dict[str, Any]] = ContextVar("metadata", default={})

# Global client
_client: Optional[Client] = None
_is_tracing_enabled = False


def setup_langsmith() -> bool:
    """
    Initialize LangSmith tracing from environment variables.
    Returns True if tracing is enabled, False otherwise.
    """
    global _client, _is_tracing_enabled
    
    # Check if tracing is explicitly enabled
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    api_key = os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT", "ai-cfo-platform")
    
    if not tracing_enabled or not api_key:
        logger.info(
            "LangSmith tracing disabled: missing LANGSMITH_TRACING=true or LANGSMITH_API_KEY"
        )
        _is_tracing_enabled = False
        return False
    
    try:
        _client = Client(
            api_key=api_key,
            api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        )
        _is_tracing_enabled = True
        logger.info(f"LangSmith tracing enabled for project: {project}")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmith client: {e}")
        _is_tracing_enabled = False
        return False


def get_client() -> Optional[Client]:
    """Get the LangSmith client instance."""
    return _client


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return _is_tracing_enabled


@contextmanager
def tracing_context(
    thread_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
) -> Generator[None, None, None]:
    """
    Context manager for propagating tracing context.
    
    Usage:
        with tracing_context(thread_id="session-123", metadata={"user": "test"}):
            # All traceable calls here will have this context
            process_data()
    """
    token_thread = _current_thread_id.set(thread_id)
    
    default_metadata = {
        "startup_stage": os.getenv("STARTUP_STAGE", "seed"),
        "currency": os.getenv("CURRENCY", "USD"),
        "source": "streamlit",
    }
    if metadata:
        default_metadata.update(metadata)
    
    # Add thread_id to metadata
    default_metadata["thread_id"] = thread_id
    
    token_metadata = _current_metadata.set(default_metadata)
    
    # Set LangSmith environment variables for this context
    os.environ["LANGSMITH_RUN"] = thread_id
    
    try:
        yield
    finally:
        _current_thread_id.reset(token_thread)
        _current_metadata.reset(token_metadata)


def get_current_metadata() -> Dict[str, Any]:
    """Get the current tracing metadata from context."""
    metadata = _current_metadata.get({}).copy()
    thread_id = _current_thread_id.get()
    if thread_id:
        metadata["thread_id"] = thread_id
    return metadata


def get_current_tags() -> list:
    """Get default tags for the current context."""
    tags = ["ai-cfo"]
    thread_id = _current_thread_id.get()
    if thread_id:
        tags.append(f"thread:{thread_id}")
    return tags


def traced(name: Optional[str] = None, tags: Optional[list] = None):
    """
    Decorator for tracing functions with LangSmith.
    Automatically propagates context and metadata.
    
    Usage:
        @traced("burn_analysis", tags=["burn", "expense"])
        def compute_burn(data):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return func(*args, **kwargs)
            
            # Get context
            metadata = get_current_metadata()
            default_tags = get_current_tags()
            if tags:
                default_tags.extend(tags)
            
            # Apply traceable wrapper
            @traceable(
                name=name or func.__name__,
                run_type="chain",
                metadata=metadata,
                tags=default_tags,
            )
            def traced_func(*a, **kw):
                return func(*a, **kw)
            
            return traced_func(*args, **kwargs)
        return wrapper
    return decorator


def trace_chat_completion():
    """
    Wrap OpenAI chat completion for tracing.
    Use this to wrap the OpenAI client.
    """
    from openai import OpenAI
    
    if is_tracing_enabled() and _client:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return wrap_openai(openai_client, project_name=os.getenv("LANGSMITH_PROJECT", "ai-cfo-platform"))
    
    # Fallback: return unwrapped client
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def log_assumption(
    source: str,
    parameter: str,
    value: Any,
    rationale: str,
    confidence: Optional[float] = None,
) -> None:
    """
    Log an assumption to LangSmith for auditability.
    
    Args:
        source: Agent or function making the assumption
        parameter: The parameter being assumed
        value: The assumed value
        rationale: Why this assumption was made
        confidence: Optional confidence score (0-1)
    """
    if not is_tracing_enabled():
        return
    
    metadata = get_current_metadata()
    thread_id = _current_thread_id.get()
    
    assumption = {
        "source": source,
        "parameter": parameter,
        "value": value,
        "rationale": rationale,
        "confidence": confidence,
        "timestamp": pendulum.now().isoformat(),
        "thread_id": thread_id,
    }
    
    # Log as a custom trace
    client = get_client()
    if client:
        try:
            # This creates a trace with the assumption as metadata
            with traceable(
                name="assumption_log",
                run_type="chain",
                metadata={**metadata, "assumption": json.dumps(assumption)},
                tags=["assumption", "audit"],
            ):
                pass  # Just logging the metadata
        except Exception as e:
            logger.debug(f"Failed to log assumption: {e}")


def log_metric(
    metric_name: str,
    value: float,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a metric to LangSmith for monitoring.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        context: Additional context
    """
    if not is_tracing_enabled():
        return
    
    metadata = get_current_metadata()
    thread_id = _current_thread_id.get()
    
    metric_data = {
        "metric": metric_name,
        "value": value,
        "context": context or {},
        "timestamp": pendulum.now().isoformat(),
        "thread_id": thread_id,
    }
    
    try:
        with traceable(
            name=f"metric_{metric_name}",
            run_type="chain",
            metadata={**metadata, "metric": json.dumps(metric_data)},
            tags=["metric", metric_name],
        ):
            pass
    except Exception as e:
        logger.debug(f"Failed to log metric: {e}")


# Initialize tracing on module import
setup_langsmith()

# Add pendulum import
import pendulum