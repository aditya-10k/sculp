"""Evaluator Protocols for SCULP:
- Protocol A: Blind Standalone Interpretability (Prompt-isolated).
- Protocol B: Reference-Guided Semantic Preservation (Ground-truth reference criteria).
- Evaluator Reliability and Agreement testing across independent LLMs.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np

from src.data.schema import GroundTruth
from src.interpretation.prompts import (
    PROTOCOL_A_RUBRIC_VERSION,
    PROTOCOL_A_SYSTEM_PROMPT,
    PROTOCOL_A_USER_TEMPLATE,
    BlindInterpretabilityResult,
    PROTOCOL_B_RUBRIC_VERSION,
    PROTOCOL_B_SYSTEM_PROMPT,
    PROTOCOL_B_USER_TEMPLATE,
    ReferenceGuidedPreservationResult
)
from src.interpretation.groq_client import GroqEvaluatorClient, InterpreterAPIError


class ProtocolAEvaluator:
    """
    Protocol A: Blind Standalone Interpretability Evaluator.
    STRICT ISOLATION GUARANTEE:
    This evaluator accepts ONLY the compressed prompt string.
    Original prompt text, ground truth annotations, compression ratios, and attribution scores
    are strictly excluded from the prompt payload and context.
    """

    def __init__(
        self,
        client: Optional[GroqEvaluatorClient] = None,
        model_name: Optional[str] = None
    ):
        self.client = client or GroqEvaluatorClient(model_name=model_name)
        self.model_name = self.client.model_name

    def evaluate_standalone(
        self,
        compressed_prompt: str,
        override_model: Optional[str] = None
    ) -> BlindInterpretabilityResult:
        """
        Evaluates the standalone interpretability of the compressed prompt.
        """
        user_prompt = PROTOCOL_A_USER_TEMPLATE.format(
            compressed_prompt=compressed_prompt.strip()
        )
        return self.client.generate_json_structured(
            protocol_name="protocol_a_blind",
            protocol_version=PROTOCOL_A_RUBRIC_VERSION,
            system_prompt=PROTOCOL_A_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=BlindInterpretabilityResult,
            override_model=override_model
        )


class ProtocolBEvaluator:
    """
    Protocol B: Reference-Guided Semantic Preservation Evaluator.
    This evaluator compares the compressed prompt against structured reference criteria
    derived from the original prompt (intent, constraints, context, tone, steps).
    It is NOT blind to the reference criteria and is transparently reported as reference-guided.
    """

    def __init__(
        self,
        client: Optional[GroqEvaluatorClient] = None,
        model_name: Optional[str] = None
    ):
        self.client = client or GroqEvaluatorClient(model_name=model_name)
        self.model_name = self.client.model_name

    def evaluate_preservation(
        self,
        compressed_prompt: str,
        ground_truth: GroundTruth,
        override_model: Optional[str] = None
    ) -> ReferenceGuidedPreservationResult:
        """
        Evaluates semantic preservation against structured reference criteria.
        """
        user_prompt = PROTOCOL_B_USER_TEMPLATE.format(
            intent=ground_truth.intent or "None specified",
            constraints=", ".join(ground_truth.constraints) if ground_truth.constraints else "None",
            context=", ".join(ground_truth.context) if ground_truth.context else "None",
            tone=", ".join(ground_truth.tone) if ground_truth.tone else "Neutral / standard",
            steps=" -> ".join(ground_truth.steps) if ground_truth.steps else "None",
            compressed_prompt=compressed_prompt.strip()
        )

        return self.client.generate_json_structured(
            protocol_name="protocol_b_reference_guided",
            protocol_version=PROTOCOL_B_RUBRIC_VERSION,
            system_prompt=PROTOCOL_B_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=ReferenceGuidedPreservationResult,
            override_model=override_model
        )


class EvaluatorAgreementSuite:
    """
    Evaluates reliability and agreement between two independent LLMs
    (e.g., Primary Model vs Secondary Model) on a predefined evaluation subset.
    """

    def __init__(
        self,
        client: Optional[GroqEvaluatorClient] = None,
        primary_model: str = "openai/gpt-oss-20b",
        secondary_model: str = "qwen/qwen3.8-27b"
    ):
        self.client = client or GroqEvaluatorClient()
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.proto_a = ProtocolAEvaluator(self.client)
        self.proto_b = ProtocolBEvaluator(self.client)

    def evaluate_agreement_sample(
        self,
        compressed_prompt: str,
        ground_truth: GroundTruth
    ) -> Dict[str, Any]:
        """
        Runs both models on the same prompt and returns comparative scores.
        """
        # Model 1
        res_a_1 = self.proto_a.evaluate_standalone(compressed_prompt, override_model=self.primary_model)
        res_b_1 = self.proto_b.evaluate_preservation(compressed_prompt, ground_truth, override_model=self.primary_model)

        # Model 2
        res_a_2 = self.proto_a.evaluate_standalone(compressed_prompt, override_model=self.secondary_model)
        res_b_2 = self.proto_b.evaluate_preservation(compressed_prompt, ground_truth, override_model=self.secondary_model)

        diff_proto_a = abs(res_a_1.standalone_interpretability - res_a_2.standalone_interpretability)
        diff_proto_b = abs(res_b_1.composite_preservation - res_b_2.composite_preservation)

        return {
            "primary_model": self.primary_model,
            "secondary_model": self.secondary_model,
            "primary_protocol_a": res_a_1.standalone_interpretability,
            "secondary_protocol_a": res_a_2.standalone_interpretability,
            "delta_protocol_a": round(diff_proto_a, 4),
            "primary_protocol_b": res_b_1.composite_preservation,
            "secondary_protocol_b": res_b_2.composite_preservation,
            "delta_protocol_b": round(diff_proto_b, 4),
        }
