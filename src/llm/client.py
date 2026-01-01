# src/llm/client.py
from __future__ import annotations

from typing import List, Dict

import httpx

from src.config import (
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
)


class LLMError(RuntimeError):
    pass


def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    Call an OpenAI-compatible chat completion endpoint (Groq etc.)
    and return the assistant's text content.
    """
    if not LLM_API_KEY:
        raise LLMError("LLM_API_KEY (or GROQ_API_KEY) is not set in .env")

    payload = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": messages,
    }

    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}

    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMError(f"LLM HTTP error: {exc}") from exc

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LLMError(f"Unexpected LLM response format: {data}") from exc
