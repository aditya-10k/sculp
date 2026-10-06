"""Intent semantic similarity evaluator using Sentence Transformers."""

from __future__ import annotations
from typing import Optional
import torch
from sentence_transformers.util import cos_sim
from src.evaluation.encoder import CachedSentenceEncoder


class IntentEvaluator:
    """
    Evaluates semantic preservation of prompt intent by computing cosine similarity
    between ground truth intent and reconstructed task description.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        encoder: Optional[CachedSentenceEncoder] = None
    ):
        self.encoder = encoder or CachedSentenceEncoder(model_name=model_name, device=device)

    def evaluate(self, ground_truth_intent: str, predicted_task: str) -> float:
        gt = ground_truth_intent.strip()
        pred = predicted_task.strip()

        if not gt:
            return 1.0 if not pred else 0.5
        if not pred:
            return 0.0

        e_gt = self.encoder.encode(gt)
        e_pred = self.encoder.encode(pred)

        if e_gt.dim() == 1:
            e_gt = e_gt.unsqueeze(0)
        if e_pred.dim() == 1:
            e_pred = e_pred.unsqueeze(0)

        sim = float(cos_sim(e_gt, e_pred)[0][0].detach().cpu().item())
        return max(0.0, min(1.0, sim))
