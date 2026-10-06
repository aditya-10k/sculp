"""Constraint fidelity evaluator (precision, recall, F1) using semantic matching."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch
from sentence_transformers.util import cos_sim
from src.evaluation.encoder import CachedSentenceEncoder


class ConstraintEvaluator:
    """
    Evaluates whether ground-truth constraints were preserved in the reconstructed prompt.
    Calculates constraint precision, recall, and F1.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        semantic_threshold: float = 0.72,
        encoder: Optional[CachedSentenceEncoder] = None
    ):
        self.encoder = encoder or CachedSentenceEncoder(model_name=model_name, device=device)
        self.threshold = semantic_threshold

    def evaluate(self, ground_truth_constraints: List[str], predicted_constraints: List[str]) -> Dict[str, float]:
        gt_list = [c.strip() for c in ground_truth_constraints if c.strip()]
        pred_list = [c.strip() for c in predicted_constraints if c.strip()]

        if not gt_list:
            if not pred_list:
                return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
            else:
                return {"recall": 1.0, "precision": 0.8, "f1": 0.88}

        if not pred_list:
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

        e_gt = self.encoder.encode_list(gt_list)
        e_pred = self.encoder.encode_list(pred_list)

        sim_matrix = cos_sim(e_gt, e_pred).detach().cpu().numpy()

        matched_gt = sum(1 for i in range(len(gt_list)) if sim_matrix[i].max() >= self.threshold)
        matched_pred = sum(1 for j in range(len(pred_list)) if sim_matrix[:, j].max() >= self.threshold)

        recall = matched_gt / len(gt_list)
        precision = matched_pred / len(pred_list)
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        return {
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(f1), 4)
        }
