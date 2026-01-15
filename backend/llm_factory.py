# LLM Factory - Unified interface for Gemini and Ollama
import os
from typing import Optional, Literal
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import logging
logger = logging.getLogger(__name__)

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")  # Coding-focused
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class LLMFactory:
    """Factory for creating LLM instances with provider fallback."""

    @staticmethod
    def create_llm(
        provider: Optional[Literal["gemini", "ollama"]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs
    ):
        """Create an LLM instance based on provider preference."""
        provider = provider or LLM_PROVIDER

        if provider == "ollama":
            return LLMFactory._create_ollama(model or OLLAMA_MODEL, temperature, **kwargs)
        else:
            return LLMFactory._create_gemini(model or GEMINI_MODEL, temperature, **kwargs)

    @staticmethod
    def _create_ollama(model: str, temperature: float = 0.3, **kwargs):
        """Create Ollama LLM with chat interface."""
        try:
            # Filter out unsupported kwargs for Ollama
            ollama_kwargs = {k: v for k, v in kwargs.items() if k not in ['top_p', 'google_api_key']}

            llm = ChatOllama(
                model=model,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature,
                **ollama_kwargs
            )
            logger.info(f"Ollama LLM initialized: {model} @ {OLLAMA_BASE_URL}")
            return llm
        except Exception as e:
            logger.warning(f"Failed to create Ollama LLM ({model}): {e}")
            logger.info("Falling back to Gemini...")
            return LLMFactory._create_gemini()

    @staticmethod
    def _create_gemini(model: str = GEMINI_MODEL, temperature: float = 0.3, **kwargs):
        """Create Gemini LLM."""
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set")
            return None

        try:
            # Extract Gemini-specific kwargs
            gemini_kwargs = {k: v for k, v in kwargs.items() if k in ['top_p', 'top_k', 'max_output_tokens']}
            gemini_kwargs['model'] = model
            gemini_kwargs['google_api_key'] = GEMINI_API_KEY
            gemini_kwargs['temperature'] = temperature

            llm = ChatGoogleGenerativeAI(**gemini_kwargs)
            logger.info(f"Gemini LLM initialized: {model}")
            return llm
        except Exception as e:
            logger.error(f"Failed to create Gemini LLM: {e}")
            return None

    @staticmethod
    def get_coder_llm():
        """Get specialized coding LLM (qwen2.5-coder for code tasks)."""
        if LLM_PROVIDER == "ollama":
            return LLMFactory._create_ollama(
                model="qwen2.5-coder:3b",
                temperature=0.1  # Lower temperature for code accuracy
            )
        return LLMFactory.create_llm(temperature=0.1)

    @staticmethod
    def get_reasoning_llm():
        """Get reasoning-focused LLM (granite3.1-moe for analysis)."""
        if LLM_PROVIDER == "ollama":
            return LLMFactory._create_ollama(
                model="granite3.1-moe:3b",
                temperature=0.3
            )
        return LLMFactory.create_llm(temperature=0.3)

    @staticmethod
    def get_fast_llm():
        """Get fast LLM for simple tasks."""
        if LLM_PROVIDER == "ollama":
            return LLMFactory._create_ollama(
                model="qwen2.5-coder:3b",  # Fast and capable
                temperature=0.2
            )
        return LLMFactory.create_llm(temperature=0.2)

    @staticmethod
    def is_available() -> bool:
        """Check if any LLM provider is available."""
        if LLM_PROVIDER == "ollama":
            return LLMFactory._check_ollama_available()
        return bool(GEMINI_API_KEY)

    @staticmethod
    def _check_ollama_available() -> bool:
        """Check if Ollama is running and accessible."""
        try:
            import requests
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_llm(provider: Optional[str] = None, **kwargs):
    """Convenience function to get LLM instance."""
    return LLMFactory.create_llm(provider=provider, **kwargs)


# Example usage:
if __name__ == "__main__":
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Ollama available: {LLMFactory._check_ollama_available()}")

    llm = LLMFactory.create_llm()
    if llm:
        print(f"LLM initialized: {type(llm).__name__}")

        # Test simple invocation
        response = llm.invoke("Say 'Hello from [provider]'")
        print(f"Response: {response.content}")
