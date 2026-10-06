"""Intent semantic similarity evaluator using Sentence Transformers."""

from __future__ import annotations
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class IntentEvaluator:
    """
    Evaluates semantic preservation of prompt intent by computing cosine similarity
    between ground truth intent and reconstructed task description.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):
        dev = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model = SentenceTransformer(model_name, device=dev)

    def evaluate(self, ground_truth_intent: str, predicted_task: str) -> float:
        """
        Returns cosine similarity in [0.0, 1.0].
        """
        gt = ground_truth_intent.strip()
        pred = predicted_task.strip()

        if not gt:
            return 1.0 if not pred else 0.5
        if not pred:
            return 0.0

        e_gt = self.model.encode(gt, convert_to_tensor=True, show_progress_bar=False)
        e_pred = self.model.encode(pred, convert_to_tensor=True, show_progress_bar=False)

        sim = float(cos_sim(e_gt, e_pred)[0][0].detach().cpu().item())
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, sim))
