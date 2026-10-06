"""Blind LLM Interpreter implementation with caching and multi-provider support."""

from __future__ import annotations
import os
import json
import hashlib
import re
from pathlib import Path
from typing import Optional, Dict, Any

from src.interpretation.prompts import (
    INTERPRETER_SYSTEM_PROMPT,
    INTERPRETER_USER_TEMPLATE,
    ReconstructedSemantics
)


class BlindInterpreter:
    """
    Blind Interpreter LLM.
    Evaluates compressed prompts without seeing original prompts, scores, or ground truth.
    Supports API providers (Gemini, OpenAI, Groq) and deterministic heuristic offline mode.
    """

    def __init__(
        self,
        provider: str = "mock",
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        cache_dir: Optional[str] = "results/interpreter_cache"
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize provider client if API key is present."""
        if self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self._client = genai.GenerativeModel(self.model_name)
                else:
                    self.provider = "mock"
            except Exception:
                self.provider = "mock"

        elif self.provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self._client = openai.OpenAI(api_key=api_key)
                else:
                    self.provider = "mock"
            except Exception:
                self.provider = "mock"

        elif self.provider == "groq":
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if api_key:
                    self._client = Groq(api_key=api_key)
                else:
                    self.provider = "mock"
            except Exception:
                self.provider = "mock"

    def _get_cache_path(self, compressed_prompt: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(f"{self.provider}:{self.model_name}:{compressed_prompt}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def interpret(self, compressed_prompt: str) -> ReconstructedSemantics:
        """
        Interprets compressed prompt blind to original context.
        """
        # 1. Check cache
        cache_path = self._get_cache_path(compressed_prompt)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ReconstructedSemantics(**data)
            except Exception:
                pass

        # 2. Call provider
        if self.provider == "mock" or self._client is None:
            recon = self._heuristic_interpretation(compressed_prompt)
        elif self.provider == "gemini":
            recon = self._call_gemini(compressed_prompt)
        elif self.provider == "openai":
            recon = self._call_openai(compressed_prompt)
        elif self.provider == "groq":
            recon = self._call_groq(compressed_prompt)
        else:
            recon = self._heuristic_interpretation(compressed_prompt)

        # 3. Save cache
        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(recon.model_dump(), f, indent=2)
            except Exception:
                pass

        return recon

    def _call_gemini(self, compressed_prompt: str) -> ReconstructedSemantics:
        prompt_text = f"{INTERPRETER_SYSTEM_PROMPT}\n\n{INTERPRETER_USER_TEMPLATE.format(compressed_prompt=compressed_prompt)}"
        try:
            response = self._client.generate_content(
                prompt_text,
                generation_config={"temperature": self.temperature, "response_mime_type": "application/json"}
            )
            raw = response.text
            return self._parse_json_response(raw)
        except Exception:
            return self._heuristic_interpretation(compressed_prompt)

    def _call_openai(self, compressed_prompt: str) -> ReconstructedSemantics:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": INTERPRETER_SYSTEM_PROMPT},
                    {"role": "user", "content": INTERPRETER_USER_TEMPLATE.format(compressed_prompt=compressed_prompt)}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            return self._parse_json_response(raw)
        except Exception:
            return self._heuristic_interpretation(compressed_prompt)

    def _call_groq(self, compressed_prompt: str) -> ReconstructedSemantics:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": INTERPRETER_SYSTEM_PROMPT},
                    {"role": "user", "content": INTERPRETER_USER_TEMPLATE.format(compressed_prompt=compressed_prompt)}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            return self._parse_json_response(raw)
        except Exception:
            return self._heuristic_interpretation(compressed_prompt)

    def _parse_json_response(self, text: str) -> ReconstructedSemantics:
        """Clean markdown wrapping and parse into ReconstructedSemantics."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            return ReconstructedSemantics(
                task=data.get("task", ""),
                role=data.get("role", ""),
                constraints=data.get("constraints", []),
                tone=data.get("tone", []),
                context=data.get("context", []),
                required_actions=data.get("required_actions", []),
                ordering_dependencies=data.get("ordering_dependencies", []),
                raw_response=text
            )
        except Exception:
            return self._heuristic_interpretation(text)

    def _heuristic_interpretation(self, prompt: str) -> ReconstructedSemantics:
        """
        Deterministic, offline semantic interpreter fallback.
        Parses remaining lexical cues without hallucinating missing information.
        """
        text = prompt.strip()
        lower = text.lower()

        # Task recovery: look for imperative verbs and primary clause
        task = ""
        action_match = re.search(
            r'\b(summarize|explain|describe|write|draft|extract|generate|refactor|convert|identify|provide|review|assess|diagnose|formulate|calculate)\b[^\.\n]*',
            text, re.IGNORECASE
        )
        if action_match:
            task = action_match.group(0).strip()
        elif text:
            # Fallback to first sentence or fragment
            task = text.split(".")[0].strip()

        # Role
        role = ""
        role_match = re.search(r'\byou are (?:a|an) ([a-zA-Z\s-]+?)(?:\.|\,|$)', text, re.IGNORECASE)
        if role_match:
            role = role_match.group(1).strip()

        # Constraints
        constraints = []
        if re.search(r'\b(do not|never|under no circumstances|must not|prohibit|avoid)\b', lower):
            neg_matches = re.findall(r'(?:do not|never|must not|prohibit|avoid)\s+[^.,;\n]+', text, re.IGNORECASE)
            constraints.extend([m.strip() for m in neg_matches])
        if re.search(r'\b(exactly \d+|under \d+|fewer than \d+|\d+ words|\d+ sentences|bullet points)\b', lower):
            num_matches = re.findall(r'(?:exactly|under|fewer than|\b)\s*\d+\s*(?:words|sentences|bullet points|points)', text, re.IGNORECASE)
            constraints.extend([m.strip() for m in num_matches])
        if "json" in lower:
            constraints.append("output strictly valid JSON")

        # Tone
        tone = []
        tone_words = ["empathetic", "professional", "concise", "authoritative", "urgent", "sober", "whimsical",
                      "reverent", "playful", "diplomatic", "apologetic", "sarcastic", "encouraging"]
        for tw in tone_words:
            if tw in lower:
                tone.append(tw)

        # Context
        context = []
        if re.search(r'\b(customer|patient|client|blizzard|outage|series b|lawsuit|rent|mysql|redis|monarch)\b', lower):
            ctx_matches = re.findall(r'([^.,;\n]*(?:customer|patient|client|outage|flight|budget|traffic)[^.,;\n]*)', text, re.IGNORECASE)
            context.extend([m.strip() for m in ctx_matches[:3] if len(m.strip()) > 5])

        # Ordering / steps
        steps = []
        step_matches = re.findall(r'(?:step \d+|first|second|third|then|next|finally|phase \d+|stage \d+)[^.,;\n]+', text, re.IGNORECASE)
        if len(step_matches) >= 2:
            steps = [s.strip() for s in step_matches]

        return ReconstructedSemantics(
            task=task,
            role=role,
            constraints=list(set(constraints)),
            tone=list(set(tone)),
            context=list(set(context)),
            required_actions=[task] if task else [],
            ordering_dependencies=steps,
            raw_response="[heuristic_reconstruction]"
        )
