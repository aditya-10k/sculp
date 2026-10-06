"""Multi-step and dependency preservation evaluator."""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class DependencyEvaluator:
    """
    Evaluates multi-step procedure preservation, calculating:
    - step_recall: Fraction of ground truth steps successfully recovered.
    - ordering_preservation: Fraction of pairwise step orderings preserved.
    - dependency_preservation: Joint score (step_recall * ordering_preservation).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        semantic_threshold: float = 0.65
    ):
        dev = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model = SentenceTransformer(model_name, device=dev)
        self.threshold = semantic_threshold

    def evaluate(self, ground_truth_steps: List[str], predicted_steps: List[str]) -> Dict[str, float]:
        gt_steps = [s.strip() for s in ground_truth_steps if s.strip()]
        pred_steps = [s.strip() for s in predicted_steps if s.strip()]

        if not gt_steps:
            return {
                "step_recall": 1.0,
                "ordering_preservation": 1.0,
                "dependency_preservation": 1.0
            }

        if not pred_steps:
            return {
                "step_recall": 0.0,
                "ordering_preservation": 0.0,
                "dependency_preservation": 0.0
            }

        e_gt = self.model.encode(gt_steps, convert_to_tensor=True, show_progress_bar=False)
        e_pred = self.model.encode(pred_steps, convert_to_tensor=True, show_progress_bar=False)

        sim_matrix = cos_sim(e_gt, e_pred).detach().cpu().numpy()

        # For each GT step, find best matching predicted step index
        recovered_pairs: List[Tuple[int, int]] = []  # (gt_idx, pred_idx)
        for i in range(len(gt_steps)):
            best_idx = int(sim_matrix[i].argmax())
            best_sim = sim_matrix[i, best_idx]
            if best_sim >= self.threshold:
                recovered_pairs.append((i, best_idx))

        step_recall = len(recovered_pairs) / len(gt_steps)

        # Ordering preservation: check concordant pairs among recovered steps
        if len(recovered_pairs) < 2:
            ordering_preservation = 1.0 if step_recall > 0 else 0.0
        else:
            concordant = 0
            total_pairs = 0
            for i in range(len(recovered_pairs)):
                for j in range(i + 1, len(recovered_pairs)):
                    gt_i, pred_i = recovered_pairs[i]
                    gt_j, pred_j = recovered_pairs[j]
                    total_pairs += 1
                    # Since gt_i < gt_j by construction, order is preserved if pred_i < pred_j
                    if pred_i <= pred_j:
                        concordant += 1
            ordering_preservation = concordant / total_pairs if total_pairs > 0 else 1.0

        dependency_preservation = step_recall * ordering_preservation

        return {
            "step_recall": round(float(step_recall), 4),
            "ordering_preservation": round(float(ordering_preservation), 4),
            "dependency_preservation": round(float(dependency_preservation), 4)
        }
