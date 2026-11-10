"""
LLM Client Wrapper with Automatic Model Fallback for Verity Financial RAG.

Provides `get_llm_with_fallback()` and `AllModelsExhaustedError` to handle
Gemini free-tier quota exhaustion (429 / RESOURCE_EXHAUSTED) gracefully across primary
and secondary fallback models.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Process-wide cache mapping primary_model -> currently active working model
_active_model_cache: Dict[str, str] = {}
_llm_call_hooks: List[Any] = []
_last_model_used: Optional[str] = None


def get_last_model_used() -> Optional[str]:
    """Returns the model identifier that successfully handled the most recent LLM invocation."""
    return _last_model_used


def register_llm_call_hook(hook: Any) -> None:
    """Registers a callback function fn(model_name: str) called on each LLM invocation."""
    if hook not in _llm_call_hooks:
        _llm_call_hooks.append(hook)


def unregister_llm_call_hook(hook: Any) -> None:
    """Unregisters a previously registered LLM call hook."""
    if hook in _llm_call_hooks:
        _llm_call_hooks.remove(hook)


def notify_llm_call(model: str) -> None:
    """Notifies registered hooks that an LLM call is being dispatched to model."""
    for hook in list(_llm_call_hooks):
        try:
            hook(model)
        except Exception as e:
            logger.warning(f"Error in LLM call hook: {e}")


def reset_fallback_cache() -> None:
    """Resets the process-wide active model fallback cache."""
    global _active_model_cache
    _active_model_cache.clear()
    logger.info("Process-wide LLM fallback model cache reset.")


class AllModelsExhaustedError(Exception):
    """Raised when all primary and fallback models fail due to API quota exhaustion."""

    def __init__(self, attempted_models: List[str], last_error: Exception):
        self.attempted_models = attempted_models
        self.last_error = last_error
        msg = (
            f"All configured LLM models in fallback chain failed due to quota exhaustion/errors: "
            f"{attempted_models}. Last error: {last_error}"
        )
        super().__init__(msg)


def is_quota_error(e: Exception) -> bool:
    """Checks if an exception is a rate limit or quota exhaustion error (429 / RESOURCE_EXHAUSTED / NOT_FOUND)."""
    err_str = str(e).upper()
    return (
        "429" in err_str
        or "RESOURCE_EXHAUSTED" in err_str
        or "QUOTA" in err_str
        or "NOT_FOUND" in err_str
        or "NOT FOUND" in err_str
    )


class FallbackLLMWrapper:
    """
    Wrapper around ChatGoogleGenerativeAI that automatically attempts a primary model
    and falls back through a sequence of secondary models on quota/rate limit errors.
    """

    def __init__(
        self,
        primary_model: str,
        fallback_models: Optional[List[str]] = None,
        structured_schema: Optional[Type[BaseModel]] = None,
        **llm_kwargs: Any,
    ):
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.structured_schema = structured_schema
        self.llm_kwargs = llm_kwargs

    def _get_candidate_models(self) -> List[str]:
        """Returns ordered list of candidate models, placing the cached active working model first if present."""
        full_chain = [self.primary_model] + [m for m in self.fallback_models if m != self.primary_model]
        cached_model = _active_model_cache.get(self.primary_model)
        if cached_model and cached_model in full_chain:
            full_chain.remove(cached_model)
            full_chain.insert(0, cached_model)
        return full_chain

    def with_structured_output(self, schema: Type[BaseModel], **kwargs: Any) -> "FallbackLLMWrapper":
        """Returns a new FallbackLLMWrapper configured for structured output."""
        return FallbackLLMWrapper(
            primary_model=self.primary_model,
            fallback_models=self.fallback_models,
            structured_schema=schema,
            **self.llm_kwargs,
        )

    def invoke(self, input_messages: Any, **kwargs: Any) -> Any:
        """
        Attempts to invoke the LLM using candidate models sequentially.
        Updates cache on success; falls back or raises AllModelsExhaustedError on failure.
        """
        candidates = self._get_candidate_models()
        attempted: List[str] = []
        last_exception: Optional[Exception] = None

        for model in candidates:
            attempted.append(model)
            notify_llm_call(model)
            try:
                llm = ChatGoogleGenerativeAI(model=model, **self.llm_kwargs)
                target = llm.with_structured_output(self.structured_schema) if self.structured_schema else llm
                result = target.invoke(input_messages, **kwargs)

                # Cache working model for subsequent calls in this process
                global _last_model_used
                _last_model_used = model
                _active_model_cache[self.primary_model] = model
                return result

            except Exception as e:
                last_exception = e
                if is_quota_error(e):
                    next_models = [m for m in candidates if m not in attempted]
                    if next_models:
                        logger.warning(
                            f"LLM call to model '{model}' failed ({e}). "
                            f"Falling back to next model '{next_models[0]}' in chain..."
                        )
                    else:
                        logger.warning(f"LLM call to model '{model}' failed ({e}). No further fallback models available.")
                else:
                    # Re-raise non-quota error directly
                    raise e

        raise AllModelsExhaustedError(attempted_models=attempted, last_error=last_exception)


def get_llm_with_fallback(
    primary_model: str,
    fallback_models: Optional[List[str]] = None,
    **llm_kwargs: Any,
) -> FallbackLLMWrapper:
    """
    Factory function returning a FallbackLLMWrapper for ChatGoogleGenerativeAI.
    """
    return FallbackLLMWrapper(
        primary_model=primary_model,
        fallback_models=fallback_models,
        **llm_kwargs,
    )
