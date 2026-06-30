"""
LLM Client Configuration - Groq Integration.
"""

from typing import Optional
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from settings import settings


def get_llm():
    """
    Get the LLM instance (Groq by default).
    """
    if settings.groq_api_key:
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.1,
            max_tokens=4096,
        )
    else:
        # Fallback to OpenAI if Groq key not available
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.1,
        )


def get_llm_with_tracing():
    """
    Get LLM with LangSmith tracing enabled.
    """
    from config.langsmith import trace_chat_completion
    
    if settings.groq_api_key:
        from langchain_groq import ChatGroq
        
        # Wrap with tracing
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.1,
            max_tokens=4096,
        )
        
        # Enable LangSmith tracing
        try:
            # from langsmith.wrappers import wrap_openai
            # Groq uses OpenAI-compatible API, so we can use the same wrapper
            llm = trace_chat_completion(llm)
        except:
            pass
        
        return llm
    else:
        # Fallback to OpenAI
        return trace_chat_completion()


def get_available_models():
    """
    Get available Groq models.
    """
    return {
        
        "llama3-70b-8192": "Llama 3 70B",
        
    }