"""LLM-as-a-Judge secondary evaluation module."""

from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

JUDGE_PROMPT_TEMPLATE = """You are an expert impartial evaluator assessing prompt meaning survival after compression.

Original Prompt:
\"\"\"{original_prompt}\"\"\"

Compressed Prompt:
\"\"\"{compressed_prompt}\"\"\"

Ground Truth Constraints: {constraints}
Ground Truth Tone: {tone}
Ground Truth Steps: {steps}

Rate semantic preservation on a scale of 0.0 to 1.0 for each component:
1. "intent": Does the compressed prompt preserve the core goal and action?
2. "constraints": Are negative/positive limits and formatting rules preserved?
3. "context": Is essential background preserved?
4. "tone": Is the intended emotional tone or formality preserved?
5. "dependencies": Are multi-step workflows and dependencies preserved?
6. "overall": Overall semantic preservation score (0.0 to 1.0)
7. "reason": Brief justification

Respond ONLY with valid JSON matching:
{{
  "intent": 0.0,
  "constraints": 0.0,
  "context": 0.0,
  "tone": 0.0,
  "dependencies": 0.0,
  "overall": 0.0,
  "reason": "..."
}}"""


class JudgeScores(BaseModel):
    intent: float = 0.0
    constraints: float = 0.0
    context: float = 0.0
    tone: float = 0.0
    dependencies: float = 0.0
    overall: float = 0.0
    reason: str = ""


class LLMJudge:
    """Secondary evaluator implementing LLM-as-a-Judge with multi-provider and heuristic fallback."""

    def __init__(
        self,
        provider: str = "mock",
        model_name: str = "gemini-1.5-flash",
        cache_dir: Optional[str] = "results/judge_cache"
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._init_client()

    def _init_client(self):
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
                from src.interpretation.evaluator_protocols import ProtocolBEvaluator
                from src.interpretation.groq_client import GroqEvaluatorClient
                self.groq_client = GroqEvaluatorClient(
                    model_name=self.model_name,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None
                )
                self.proto_b_evaluator = ProtocolBEvaluator(self.groq_client)
                self._client = self.groq_client
            except Exception:
                self.provider = "mock"

    def evaluate(
        self,
        original_prompt: str,
        compressed_prompt: str,
        constraints: list,
        tone: list,
        steps: list
    ) -> JudgeScores:
        if self.provider == "groq" and hasattr(self, "proto_b_evaluator"):
            from src.data.schema import GroundTruth
            gt = GroundTruth(
                intent=original_prompt,
                constraints=constraints or [],
                tone=tone or [],
                context=[],
                steps=steps or []
            )
            res_b = self.proto_b_evaluator.evaluate_preservation(
                compressed_prompt=compressed_prompt,
                ground_truth=gt
            )
            return JudgeScores(
                intent=res_b.intent_preservation,
                constraints=res_b.constraint_preservation,
                context=res_b.context_preservation,
                tone=res_b.tone_preservation,
                dependencies=res_b.dependency_preservation,
                overall=res_b.composite_preservation,
                reason="; ".join(res_b.reasons_for_lost) if res_b.reasons_for_lost else "Requirements preserved"
            )

        # Check cache
        if self.cache_dir:
            key = f"{self.provider}:{original_prompt}:{compressed_prompt}"
            h = hashlib.sha256(key.encode("utf-8")).hexdigest()
            cache_file = self.cache_dir / f"{h}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return JudgeScores(**json.load(f))
                except Exception:
                    pass

        scores = self._run_evaluation(original_prompt, compressed_prompt, constraints, tone, steps)

        if self.cache_dir:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(scores.model_dump(), f, indent=2)
            except Exception:
                pass

        return scores

    def _run_evaluation(self, orig: str, comp: str, constraints: list, tone: list, steps: list) -> JudgeScores:
        if self.provider == "gemini" and self._client:
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                original_prompt=orig,
                compressed_prompt=comp,
                constraints=constraints,
                tone=tone,
                steps=steps
            )
            try:
                resp = self._client.generate_content(
                    prompt,
                    generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
                )
                data = json.loads(resp.text)
                return JudgeScores(**data)
            except Exception:
                pass

        # Heuristic fallback based on token retention and key phrase preservation
        orig_words = set(orig.lower().split())
        comp_words = set(comp.lower().split())
        overlap = len(orig_words & comp_words) / max(1, len(orig_words))

        has_negation = any(w in comp_words for w in ["not", "never", "no", "without"]) if any(w in orig_words for w in ["not", "never", "no", "without"]) else True
        constraint_score = overlap * (1.0 if has_negation else 0.2)
        
        return JudgeScores(
            intent=round(overlap, 3),
            constraints=round(constraint_score, 3),
            context=round(overlap * 0.9, 3),
            tone=round(overlap * 0.85, 3),
            dependencies=round(overlap * 0.8, 3),
            overall=round(overlap * 0.9, 3),
            reason="Heuristic evaluation based on lexical overlap and syntactic preservation"
        )
