"""Semantic Fidelity evaluation suite aggregating component metrics, embedding similarity, and LLM judge."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.data.schema import GroundTruth
from src.interpretation.prompts import ReconstructedSemantics
from src.evaluation.intent import IntentEvaluator
from src.evaluation.constraints import ConstraintEvaluator
from src.evaluation.tone import ToneEvaluator
from src.evaluation.context import ContextEvaluator
from src.evaluation.dependencies import DependencyEvaluator
from src.evaluation.judge import LLMJudge, JudgeScores


@dataclass
class SemanticFidelityMetrics:
    intent_score: float
    constraint_recall: float
    constraint_precision: float
    constraint_f1: float
    tone_f1: float
    context_f1: float
    step_recall: float
    ordering_preservation: float
    dependency_preservation: float
    semantic_fidelity: float  # SF = (I + C + T + X + D) / 5
    embedding_similarity: float
    llm_judge_score: float
    llm_judge_breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intent": round(self.intent_score, 4),
            "constraints": round(self.constraint_f1, 4),
            "constraint_recall": round(self.constraint_recall, 4),
            "constraint_precision": round(self.constraint_precision, 4),
            "tone": round(self.tone_f1, 4),
            "context": round(self.context_f1, 4),
            "step_recall": round(self.step_recall, 4),
            "ordering_preservation": round(self.ordering_preservation, 4),
            "dependencies": round(self.dependency_preservation, 4),
            "semantic_fidelity": round(self.semantic_fidelity, 4),
            "embedding_similarity": round(self.embedding_similarity, 4),
            "llm_judge": round(self.llm_judge_score, 4),
            "llm_judge_breakdown": self.llm_judge_breakdown
        }


class FidelityEvaluatorSuite:
    """
    Evaluates prompt meaning survival across LLM pipelines.
    Computes component-wise ground-truth fidelity, composite SF, embedding similarity, and judge scores.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        judge_provider: str = "mock"
    ):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.intent_evaluator = IntentEvaluator(embedding_model_name, device=self.device)
        self.constraint_evaluator = ConstraintEvaluator(embedding_model_name, device=self.device)
        self.tone_evaluator = ToneEvaluator(embedding_model_name, device=self.device)
        self.context_evaluator = ContextEvaluator(embedding_model_name, device=self.device)
        self.dependency_evaluator = DependencyEvaluator(embedding_model_name, device=self.device)
        self.judge = LLMJudge(provider=judge_provider)

    def evaluate(
        self,
        original_prompt: str,
        compressed_prompt: str,
        ground_truth: GroundTruth,
        reconstructed: ReconstructedSemantics
    ) -> SemanticFidelityMetrics:
        # 1. Intent Score (I)
        intent_score = self.intent_evaluator.evaluate(
            ground_truth.intent,
            reconstructed.task
        )

        # 2. Constraint Score (C)
        c_metrics = self.constraint_evaluator.evaluate(
            ground_truth.constraints,
            reconstructed.constraints
        )
        constraint_score = c_metrics["f1"]

        # 3. Tone Score (T)
        t_metrics = self.tone_evaluator.evaluate(
            ground_truth.tone,
            reconstructed.tone
        )
        tone_score = t_metrics["f1"]

        # 4. Context Score (X)
        x_metrics = self.context_evaluator.evaluate(
            ground_truth.context,
            reconstructed.context
        )
        context_score = x_metrics["f1"]

        # 5. Dependency Score (D)
        d_metrics = self.dependency_evaluator.evaluate(
            ground_truth.steps,
            reconstructed.ordering_dependencies
        )
        dependency_score = d_metrics["dependency_preservation"]

        # 6. Composite Semantic Fidelity (SF)
        # SF = (I + C + T + X + D) / 5
        sf = (intent_score + constraint_score + tone_score + context_score + dependency_score) / 5.0
        sf = max(0.0, min(1.0, sf))

        # 7. Embedding Semantic Fidelity
        e_orig = self.embedding_model.encode(original_prompt, convert_to_tensor=True, show_progress_bar=False)
        e_comp = self.embedding_model.encode(compressed_prompt, convert_to_tensor=True, show_progress_bar=False)
        emb_sim = float(cos_sim(e_orig, e_comp)[0][0].detach().cpu().item())
        emb_sim = max(0.0, min(1.0, emb_sim))

        # 8. LLM Judge
        judge_res = self.judge.evaluate(
            original_prompt=original_prompt,
            compressed_prompt=compressed_prompt,
            constraints=ground_truth.constraints,
            tone=ground_truth.tone,
            steps=ground_truth.steps
        )

        return SemanticFidelityMetrics(
            intent_score=intent_score,
            constraint_recall=c_metrics["recall"],
            constraint_precision=c_metrics["precision"],
            constraint_f1=constraint_score,
            tone_f1=tone_score,
            context_f1=context_score,
            step_recall=d_metrics["step_recall"],
            ordering_preservation=d_metrics["ordering_preservation"],
            dependency_preservation=dependency_score,
            semantic_fidelity=sf,
            embedding_similarity=emb_sim,
            llm_judge_score=judge_res.overall,
            llm_judge_breakdown=judge_res.model_dump()
        )
