"""
Groq LLM Service - Core LLM interactions with structured outputs.
Uses Groq's JSON mode for reliable parsing.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from groq import Groq
from settings import settings

logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq(api_key=settings.groq_api_key)
DEFAULT_MODEL = settings.groq_model  # "mixtral-8x7b-32768" or "llama-3.1-70b-versatile"


def call_llm(
    system_prompt: str,
    user_message: str,
    model: str = None,
    temperature: float = 0.2,
    max_tokens: int = 500,
    response_format: Optional[str] = None,  # "json_object" for structured output
) -> str:
    """
    Core LLM call with Groq.
    
    Args:
        system_prompt: System instructions
        user_message: User query
        model: Groq model name (defaults to settings.groq_model)
        temperature: 0.0-1.0 (lower = more deterministic)
        max_tokens: Maximum response tokens
        response_format: "json_object" for JSON mode
    
    Returns:
        LLM response text
    """
    model = model or DEFAULT_MODEL
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}
    
    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        logger.debug(f"LLM response: {content[:200]}...")
        return content
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise


def call_llm_with_history(
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    user_message: str,
    model: str = None,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    """
    LLM call with conversation history for context.
    
    Args:
        system_prompt: System instructions
        conversation_history: List of {"role": "user/assistant", "content": "..."}
        user_message: Current user query
        model: Groq model name
        temperature: Response creativity
        max_tokens: Maximum tokens
    
    Returns:
        LLM response text
    """
    model = model or DEFAULT_MODEL
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add last 10 messages for context
    for msg in conversation_history[-10:]:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"][:500]  # Truncate long messages
            })
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise


def extract_json_from_llm(
    system_prompt: str,
    user_message: str,
    model: str = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Get structured JSON output from LLM using Groq's JSON mode.
    
    Args:
        system_prompt: System instructions (must mention JSON output)
        user_message: User query
        model: Groq model
        temperature: Use 0.0 for most deterministic parsing
    
    Returns:
        Parsed JSON dict
    """
    response = call_llm(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        temperature=temperature,
        max_tokens=1000,
        response_format="json_object",
    )
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from text
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        logger.error(f"Failed to parse JSON from: {response[:200]}")
        return {}