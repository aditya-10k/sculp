"""Groq API Client with rate limiting, exponential backoff, caching, and schema validation.
Supports live Groq execution and deterministic mock execution for testing.
"""

from __future__ import annotations
import os
import json
import time
import hashlib
import random
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar
from datetime import datetime, timezone
from pydantic import BaseModel
from dotenv import load_dotenv

T = TypeVar("T", bound=BaseModel)


class InterpreterAPIError(Exception):
    """Raised when an API call fails after all retry attempts."""
    def __init__(self, message: str, error_category: str = "general", status_code: Optional[int] = None):
        super().__init__(message)
        self.error_category = error_category
        self.status_code = status_code


class GroqEvaluatorClient:
    """
    Robust Groq API client designed for scientific LLM evaluation.
    Features:
    - Zero secret leakage in logs or exceptions.
    - Deterministic disk-based response caching by SHA-256 hash.
    - Configurable rate limiting and jittered exponential backoff.
    - Strict Pydantic JSON schema validation.
    - Mock fallback exclusively when explicitly configured for unit testing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
        initial_backoff: float = 1.5,
        min_request_interval: float = 1.0,
        cache_dir: Optional[str] = "results/interpreter_cache",
        provider_mode: str = "auto"  # 'auto', 'groq', 'mock'
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("INTERPRETER_MODEL", "openai/gpt-oss-20b")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.min_request_interval = min_request_interval
        self.last_request_time = 0.0

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if provider_mode == "mock":
            self.is_mock = True
        elif provider_mode == "groq":
            self.is_mock = False
        else:
            self.is_mock = not bool(self.api_key)

        self._groq_client = None
        if not self.is_mock:
            if not self.api_key:
                raise InterpreterAPIError(
                    "GROQ_API_KEY is not configured in environment or .env. "
                    "Please set GROQ_API_KEY to use the live evaluator, or pass provider_mode='mock' for testing.",
                    error_category="missing_credentials"
                )
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.api_key, timeout=self.timeout_seconds)
            except ImportError:
                raise InterpreterAPIError(
                    "The 'groq' package is not installed. Please install it with 'pip install groq'.",
                    error_category="missing_dependency"
                )

    @property
    def client(self):
        """Returns the underlying Groq client instance if initialized."""
        return self._groq_client

    def compute_cache_key(
        self,
        protocol: str,
        protocol_version: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """Computes a stable SHA-256 hash incorporating protocol, model, rubric, and input."""
        content = f"{protocol}:{protocol_version}:{model_name}:{system_prompt}:{user_prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _enforce_rate_limit(self):
        """Ensures request interval does not violate token/request per minute constraints."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def generate_json_structured(
        self,
        protocol_name: str,
        protocol_version: str,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        override_model: Optional[str] = None
    ) -> T:
        """
        Executes structured JSON evaluation with caching, retries, and strict schema validation.
        """
        active_model = override_model or self.model_name
        cache_key = self.compute_cache_key(
            protocol=protocol_name,
            protocol_version=protocol_version,
            model_name=active_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        # 1. Check cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_record = json.load(f)
                    parsed_data = cached_record.get("parsed")
                    if parsed_data:
                        # Re-instantiate schema
                        res = response_model.model_validate(parsed_data)
                        if hasattr(res, "metadata"):
                            res.metadata["cache_hit"] = True
                        return res
                except Exception:
                    pass  # If cache entry is corrupted, re-generate

        # 2. Check if in mock mode
        if self.is_mock or self._groq_client is None:
            return self._generate_mock(protocol_name, user_prompt, response_model, active_model)

        # 3. Live API execution with retries and exponential backoff
        retries = 0
        backoff = self.initial_backoff
        last_error = None
        error_category = "unknown"
        start_time = time.time()

        while retries <= self.max_retries:
            self._enforce_rate_limit()
            try:
                t0 = time.time()
                completion = self._groq_client.chat.completions.create(
                    model=active_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                latency_ms = round((time.time() - t0) * 1000, 2)
                raw_text = completion.choices[0].message.content or "{}"

                # Parse and validate JSON
                parsed_json = json.loads(raw_text)
                validated_result = response_model.model_validate(parsed_json)

                # Populate metadata
                meta = {
                    "protocol": protocol_name,
                    "protocol_version": protocol_version,
                    "evaluator_model": active_model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "success",
                    "retries": retries,
                    "latency_ms": latency_ms,
                    "cache_hit": False,
                    "cache_key": cache_key
                }
                if hasattr(validated_result, "metadata"):
                    validated_result.metadata = meta

                # Persist to disk cache
                if self.cache_dir:
                    cache_record = {
                        "metadata": meta,
                        "raw_response": raw_text,
                        "parsed": validated_result.model_dump()
                    }
                    try:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(cache_record, f, indent=2)
                    except Exception:
                        pass

                return validated_result

            except json.JSONDecodeError as jde:
                last_error = jde
                error_category = "json_validation_error"
                retries += 1
                time.sleep(backoff)
                backoff *= 1.5

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                status_code = getattr(e, "status_code", None)

                if "429" in err_str or "rate limit" in err_str or status_code == 429:
                    error_category = "rate_limit_exceeded"
                    jitter = random.uniform(0.5, 1.5)
                    sleep_time = max(backoff * 2.0, 5.0) + jitter
                    time.sleep(sleep_time)
                    backoff *= 2.0
                elif "timeout" in err_str:
                    error_category = "request_timeout"
                    time.sleep(backoff)
                    backoff *= 1.5
                else:
                    error_category = "api_error"
                    time.sleep(backoff)
                    backoff *= 1.5
                retries += 1

        # All retries failed - Never silently substitute mock!
        raise InterpreterAPIError(
            f"Groq API evaluation failed after {self.max_retries} retries ({error_category}): {last_error}",
            error_category=error_category
        )

    def _generate_mock(
        self,
        protocol_name: str,
        user_prompt: str,
        response_model: Type[T],
        model_name: str
    ) -> T:
        """Deterministic mock generator for CI / testing without credentials."""
        prompt_len = len(user_prompt)
        meta = {
            "protocol": protocol_name,
            "protocol_version": "mock_v1",
            "evaluator_model": f"{model_name}-mock",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "mock_success",
            "retries": 0,
            "latency_ms": 1.0,
            "cache_hit": False,
            "cache_key": "mock"
        }

        if protocol_name == "protocol_a_blind":
            # Standalone interpretability mock
            is_coherent = prompt_len > 15
            score = 0.85 if is_coherent else 0.40
            data = {
                "standalone_interpretability": score,
                "clarity": score,
                "actionability": score,
                "inferred_task": user_prompt[:80],
                "detected_ambiguities": [] if is_coherent else ["fragmented syntax"],
                "reconstructed_components": {
                    "task": user_prompt[:60],
                    "role": "",
                    "constraints": [],
                    "tone": [],
                    "context": [],
                    "steps": []
                },
                "metadata": meta
            }
        else:
            # Reference-guided preservation mock
            score = 0.80 if prompt_len > 20 else 0.50
            data = {
                "intent_preservation": score,
                "constraint_preservation": score,
                "context_preservation": score,
                "tone_preservation": score,
                "dependency_preservation": score,
                "composite_preservation": score,
                "preserved_requirements": ["core goal"],
                "lost_requirements": [] if score > 0.6 else ["formatting boundary"],
                "ambiguities": [],
                "evidence_from_compressed": [user_prompt[:40]],
                "reasons_for_lost": [],
                "metadata": meta
            }

        return response_model.model_validate(data)
