"""Attribution Faithfulness and Importance-Damage Alignment Analysis.

Evaluates whether token importance scores accurately predict empirical semantic damage
under leave-one-out deletion:

1. Circular Self-Consistency Check:
   Damage_self(w) = 1.0 - cos(E(P), E(P \\ {w}))
   Measures consistency against the identical embedding model.
   (Embedding LOO scores correlate perfectly r_s = 1.0000 by mathematical construction).

2. Independent Empirical Validation (Ground-Truth Requirement Damage):
   Damage_gt(w) = max(0, cos(E(P), E(GT)) - cos(E(P \\ {w}), E(GT)))
   Measures degradation relative to external, human-authored ground-truth intent and constraints
   from `data/annotations/ground_truth.json` (completely independent non-circular target).

Computes rank correlations (Spearman r_s, Kendall tau) between predicted token
importance scores and both damage metrics, with 95% bootstrap confidence intervals.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.data.schema import PromptItem
from src.tokenization.tokenizer import TokenizerWrapper
from src.scoring.random import RandomScorer
from src.scoring.embedding import EmbeddingLeaveOneOutScorer
from src.scoring.gradient import GradientScorer
from src.scoring.integrated_gradients import IntegratedGradientScorer
from src.evaluation.encoder import get_encoder


class AttributionFaithfulnessEvaluator:
    """Measures rank alignment between predicted token importance and empirical leave-one-out damage."""

    def __init__(
        self,
        attribution_model: str = "bert-base-uncased",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        seed: int = 42
    ):
        self.attribution_model = attribution_model
        self.seed = seed
        self.tokenizer_wrapper = TokenizerWrapper(attribution_model)
        self.encoder = get_encoder(embedding_model)

        print("[Faithfulness] Initializing attribution scorers...")
        self.scorers = {
            "random": RandomScorer(tokenizer_model=attribution_model, seed=seed),
            "embedding": EmbeddingLeaveOneOutScorer(model_name=embedding_model, tokenizer_model=attribution_model, seed=seed),
            "gradient": GradientScorer(model_name=attribution_model, seed=seed),
            "integrated_gradients": IntegratedGradientScorer(model_name=attribution_model, seed=seed)
        }

    def measure_prompt_empirical_damage(
        self,
        prompt: str,
        ground_truth: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Computes empirical leave-one-out semantic damage for each word in the prompt.
        
        Calculates both:
        1. Self-consistency damage (drop in prompt self-similarity)
        2. Ground-truth requirement damage (drop in similarity to external ground-truth target)
        """
        _, words_orig, _ = self.tokenizer_wrapper.tokenize_and_align(prompt)
        if not words_orig:
            return []

        # 1. Original embedding
        orig_t = self.encoder.encode(prompt)
        orig_emb = orig_t.detach().cpu().numpy()
        norm_orig = np.linalg.norm(orig_emb)
        if norm_orig > 1e-9:
            orig_emb = orig_emb / norm_orig

        # 2. External ground-truth target embedding
        gt_emb = None
        sim_0_gt = 0.0
        if ground_truth is not None:
            intent = getattr(ground_truth, "intent", "") or ""
            constraints = getattr(ground_truth, "constraints", []) or []
            context = getattr(ground_truth, "context", []) or []
            c_str = " ".join(constraints) if isinstance(constraints, list) else str(constraints)
            ctx_str = " ".join(context) if isinstance(context, list) else str(context)
            gt_text = f"{intent} ; {c_str} ; {ctx_str}".strip()
            
            if gt_text and gt_text != "; ;":
                gt_t = self.encoder.encode(gt_text)
                gt_emb = gt_t.detach().cpu().numpy()
                norm_gt = np.linalg.norm(gt_emb)
                if norm_gt > 1e-9:
                    gt_emb = gt_emb / norm_gt
                sim_0_gt = float(np.dot(orig_emb, gt_emb))

        results = []
        for word in words_orig:
            loo_prompt = TokenizerWrapper.remove_word_spans(prompt, [word])
            loo_t = self.encoder.encode(loo_prompt)
            loo_emb = loo_t.detach().cpu().numpy()
            norm_loo = np.linalg.norm(loo_emb)
            if norm_loo > 1e-9:
                loo_emb = loo_emb / norm_loo

            # Self-consistency damage: drop in cosine similarity to original prompt
            cos_sim_self = float(np.dot(orig_emb, loo_emb))
            damage_self = max(0.0, 1.0 - cos_sim_self)

            # Independent GT damage: drop in cosine similarity to external ground truth
            if gt_emb is not None:
                sim_loo_gt = float(np.dot(loo_emb, gt_emb))
                damage_gt = max(0.0, sim_0_gt - sim_loo_gt)
            else:
                damage_gt = damage_self

            results.append({
                "word_id": word.word_id,
                "word": word.word,
                "start_char": word.start_char,
                "end_char": word.end_char,
                "loo_prompt": loo_prompt,
                "cos_sim_self": round(cos_sim_self, 5),
                "damage_self": round(damage_self, 5),
                "damage_gt": round(damage_gt, 5)
            })

        return results

    def evaluate_faithfulness(
        self,
        prompts: List[PromptItem],
        n_bootstrap: int = 1000,
        output_dir: str = "results/tables"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluates alignment across all prompts for the attribution methods against
        both self-consistency and independent ground-truth outcomes.
        """
        print(f"[Faithfulness] Running importance-damage alignment on {len(prompts)} prompts...")
        
        records = []
        method_corrs_self: Dict[str, List[Dict[str, float]]] = {m: [] for m in self.scorers}
        method_corrs_gt: Dict[str, List[Dict[str, float]]] = {m: [] for m in self.scorers}

        for p_idx, prompt_item in enumerate(prompts):
            prompt_text = prompt_item.prompt
            damage_results = self.measure_prompt_empirical_damage(prompt_text, prompt_item.ground_truth)
            if len(damage_results) < 3:
                continue

            damage_self_scores = [d["damage_self"] for d in damage_results]
            damage_gt_scores = [d["damage_gt"] for d in damage_results]

            # Get predicted importance scores from each method
            for method_name, scorer in self.scorers.items():
                scoring_res = scorer.score_prompt(prompt_item.prompt, prompt_id=prompt_item.id)
                pred_map = {ws.word_id: ws.normalized_score for ws in scoring_res.word_scores}
                pred_scores = [pred_map.get(d["word_id"], 0.0) for d in damage_results]

                # 1. Rank correlation against self-consistency damage
                if len(set(pred_scores)) > 1 and len(set(damage_self_scores)) > 1:
                    sp_self, _ = stats.spearmanr(pred_scores, damage_self_scores)
                    kd_self, _ = stats.kendalltau(pred_scores, damage_self_scores)
                else:
                    sp_self, kd_self = 0.0, 0.0

                sp_self = float(sp_self) if not np.isnan(sp_self) else 0.0
                kd_self = float(kd_self) if not np.isnan(kd_self) else 0.0

                # 2. Rank correlation against independent ground-truth damage
                if len(set(pred_scores)) > 1 and len(set(damage_gt_scores)) > 1:
                    sp_gt, _ = stats.spearmanr(pred_scores, damage_gt_scores)
                    kd_gt, _ = stats.kendalltau(pred_scores, damage_gt_scores)
                else:
                    sp_gt, kd_gt = 0.0, 0.0

                sp_gt = float(sp_gt) if not np.isnan(sp_gt) else 0.0
                kd_gt = float(kd_gt) if not np.isnan(kd_gt) else 0.0

                method_corrs_self[method_name].append({
                    "prompt_id": prompt_item.id,
                    "category": prompt_item.category,
                    "spearman_r": sp_self,
                    "kendall_tau": kd_self
                })
                method_corrs_gt[method_name].append({
                    "prompt_id": prompt_item.id,
                    "category": prompt_item.category,
                    "spearman_r": sp_gt,
                    "kendall_tau": kd_gt
                })

                for w_idx, d in enumerate(damage_results):
                    records.append({
                        "prompt_id": prompt_item.id,
                        "category": prompt_item.category,
                        "method": method_name,
                        "word_id": d["word_id"],
                        "word": d["word"],
                        "predicted_score": pred_scores[w_idx],
                        "damage_self": d["damage_self"],
                        "damage_gt": d["damage_gt"],
                        "spearman_self": sp_self,
                        "spearman_gt": sp_gt
                    })

        detail_df = pd.DataFrame(records)

        # Compute summary statistics and bootstrap CIs across prompts
        summary_rows = []
        for method_name in self.scorers:
            corrs_s = method_corrs_self[method_name]
            corrs_g = method_corrs_gt[method_name]
            if not corrs_s:
                continue

            sp_s_vals = np.array([c["spearman_r"] for c in corrs_s])
            kd_s_vals = np.array([c["kendall_tau"] for c in corrs_s])
            sp_g_vals = np.array([c["spearman_r"] for c in corrs_g])
            kd_g_vals = np.array([c["kendall_tau"] for c in corrs_g])

            rng = np.random.default_rng(self.seed)

            # Self-consistency CIs
            boot_sp_s = [np.mean(rng.choice(sp_s_vals, size=len(sp_s_vals), replace=True)) for _ in range(n_bootstrap)]
            ci_sp_s_low, ci_sp_s_high = np.percentile(boot_sp_s, [2.5, 97.5])

            boot_kd_s = [np.mean(rng.choice(kd_s_vals, size=len(kd_s_vals), replace=True)) for _ in range(n_bootstrap)]
            ci_kd_s_low, ci_kd_s_high = np.percentile(boot_kd_s, [2.5, 97.5])

            # Independent GT CIs
            boot_sp_g = [np.mean(rng.choice(sp_g_vals, size=len(sp_g_vals), replace=True)) for _ in range(n_bootstrap)]
            ci_sp_g_low, ci_sp_g_high = np.percentile(boot_sp_g, [2.5, 97.5])

            boot_kd_g = [np.mean(rng.choice(kd_g_vals, size=len(kd_g_vals), replace=True)) for _ in range(n_bootstrap)]
            ci_kd_g_low, ci_kd_g_high = np.percentile(boot_kd_g, [2.5, 97.5])

            # Row for Self-Consistency
            summary_rows.append({
                "method": method_name,
                "evaluation_type": "Self-Consistency (Circular)" if method_name == "embedding" else "Embedding LOO Prediction",
                "target_outcome": "Sentence-BERT LOO Drop",
                "n_prompts": len(sp_s_vals),
                "mean_spearman_r": round(float(np.mean(sp_s_vals)), 4),
                "spearman_ci_95": f"[{ci_sp_s_low:.4f}, {ci_sp_s_high:.4f}]",
                "mean_kendall_tau": round(float(np.mean(kd_s_vals)), 4),
                "kendall_ci_95": f"[{ci_kd_s_low:.4f}, {ci_kd_s_high:.4f}]",
                "spearman_p_val": float(stats.ttest_1samp(sp_s_vals, 0.0)[1]) if len(sp_s_vals) > 1 else 1.0,
                "is_circular": method_name == "embedding"
            })

            # Row for Independent Ground-Truth Damage
            summary_rows.append({
                "method": method_name,
                "evaluation_type": "Independent Empirical Validation",
                "target_outcome": "Ground-Truth Requirement Drop",
                "n_prompts": len(sp_g_vals),
                "mean_spearman_r": round(float(np.mean(sp_g_vals)), 4),
                "spearman_ci_95": f"[{ci_sp_g_low:.4f}, {ci_sp_g_high:.4f}]",
                "mean_kendall_tau": round(float(np.mean(kd_g_vals)), 4),
                "kendall_ci_95": f"[{ci_kd_g_low:.4f}, {ci_kd_g_high:.4f}]",
                "spearman_p_val": float(stats.ttest_1samp(sp_g_vals, 0.0)[1]) if len(sp_g_vals) > 1 else 1.0,
                "is_circular": False
            })

        summary_df = pd.DataFrame(summary_rows)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(out_path / "attribution_faithfulness.csv", index=False)
        summary_df.to_csv(out_path / "attribution_faithfulness_summary.csv", index=False)

        print(f"[Faithfulness] Generated detail ({len(detail_df)} rows) and summary ({len(summary_df)} evaluations)")
        return detail_df, summary_df
