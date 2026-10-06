"""Constraint fidelity evaluator (precision, recall, F1) using semantic matching."""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class ConstraintEvaluator:
    """
    Evaluates whether ground-truth constraints were preserved in the reconstructed prompt.
    Calculates constraint precision, recall, and F1.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        semantic_threshold: float = 0.72
    ):
        dev = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model = SentenceTransformer(model_name, device=dev)
        self.threshold = semantic_threshold

    def evaluate(self, ground_truth_constraints: List[str], predicted_constraints: List[str]) -> Dict[str, float]:
        """
        Compare ground truth constraints to predicted constraints.
        Returns: {recall, precision, f1}
        """
        gt_list = [c.strip() for c in ground_truth_constraints if c.strip()]
        pred_list = [c.strip() for c in predicted_constraints if c.strip()]

        if not gt_list:
            # If prompt has no constraints, perfect preservation if none predicted, or slight penalty if hallucinations
            if not pred_list:
                return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
            else:
                return {"recall": 1.0, "precision": 0.8, "f1": 0.88}

        if not pred_list:
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

        # Encode both sets
        e_gt = self.model.encode(gt_list, convert_to_tensor=True, show_progress_bar=False)
        e_pred = self.model.encode(pred_list, convert_to_tensor=True, show_progress_bar=False)

        sim_matrix = cos_sim(e_gt, e_pred).detach().cpu().numpy()  # [len(gt), len(pred)]

        # Match each GT constraint to best prediction
        matched_gt = 0
        for i in range(len(gt_list)):
            best_sim = sim_matrix[i].max()
            if best_sim >= self.threshold:
                matched_gt += 1

        matched_pred = 0
        for j in range(len(pred_list)):
            best_sim = sim_matrix[:, j].max()
            if best_sim >= self.threshold:
                matched_pred += 1

        recall = matched_gt / len(gt_list)
        precision = matched_pred / len(pred_list)

        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(f1), 4)
        }
