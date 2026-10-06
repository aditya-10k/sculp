"""Tone fidelity evaluator measuring preservation of stylistic attributes."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch
from sentence_transformers.util import cos_sim
from src.evaluation.encoder import CachedSentenceEncoder


class ToneEvaluator:
    """
    Evaluates preservation of tone attributes (e.g. empathetic, professional, humorous).
    Returns tone recall, precision, and F1.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        semantic_threshold: float = 0.65,
        encoder: Optional[CachedSentenceEncoder] = None
    ):
        self.encoder = encoder or CachedSentenceEncoder(model_name=model_name, device=device)
        self.threshold = semantic_threshold

    def evaluate(self, ground_truth_tone: List[str], predicted_tone: List[str]) -> Dict[str, float]:
        gt_list = [t.strip().lower() for t in ground_truth_tone if t.strip()]
        pred_list = [t.strip().lower() for t in predicted_tone if t.strip()]

        if not gt_list:
            return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
        if not pred_list:
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

        e_gt = self.encoder.encode_list(gt_list)
        e_pred = self.encoder.encode_list(pred_list)

        sim_matrix = cos_sim(e_gt, e_pred).detach().cpu().numpy()

        matched_gt = sum(1 for i in range(len(gt_list)) if sim_matrix[i].max() >= self.threshold)
        matched_pred = sum(1 for j in range(len(pred_list)) if sim_matrix[:, j].max() >= self.threshold)

        recall = matched_gt / len(gt_list)
        precision = matched_pred / len(pred_list)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(f1), 4)
        }
