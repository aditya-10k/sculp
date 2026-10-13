"""Attribution Faithfulness and Importance-Damage Alignment Analysis.

Evaluates whether token importance scores accurately predict semantic and functional damage
under leave-one-out deletion and progressive compression across four distinct outcomes:

1. Self-Consistency Check (Circular for Embedding LOO):
   Damage_self(w) = 1.0 - cos(E(P), E(P \\ {w}))
   Measures consistency against the identical Sentence-BERT encoder (`all-MiniLM-L6-v2`).
   For Embedding LOO, r_s = 1.0000 by mathematical construction (self-consistency, not validation).

2. Reference-Guided Embedding Proxy:
   Damage_ref_proxy(w) = max(0, cos(E(P), E(T_ref)) - cos(E(P \\ {w}), E(T_ref)))
   Measures drop in Sentence-BERT similarity to external human-authored ground-truth annotations.
   Explicitly labeled as a reference-guided embedding proxy (not an encoder-free outcome).

3. Independent Deterministic Requirement Drop (Encoder-Free):
   Damage_det_req(w) = max(0, R_det(P, G) - R_det(P \\ {w}, G))
   Measures exact lexical/stem and functional operator loss against human-annotated intent,
   constraints, context, tone, and procedural steps without any neural embedding encoder.

4. Independent Downstream Task-Success Loss (Functional LLM Execution):
   Correlates cumulative importance of pruned words against observed downstream task-success
   loss (S_orig - S_comp) measured by deterministic validators on live LLM completions.

Statistical Methodology:
- Correlations (Spearman r_s, Kendall tau) are computed strictly within individual prompts/tasks
  first, then aggregated across prompts/tasks (mean, SD, median, Fisher-z hierarchical mean).
- Prompts/tasks are bootstrapped as intact clusters (B = 10,000 resamples).
- Planned multiple comparisons are adjusted via the Holm-Bonferroni step-down procedure.
- Direct paired tests (paired t-test, Wilcoxon signed-rank, cluster-bootstrap difference CI)
  evaluate whether Gradient Attribution and Embedding LOO differ significantly.
"""

from __future__ import annotations
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
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
from src.analysis.statistics import holm_bonferroni_correction


STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "may", "might", "that", "this", "these", "those",
    "it", "its", "they", "them", "their", "our", "your", "my", "we", "you", "who",
    "whom", "which", "what", "where", "when", "why", "how", "into", "onto", "upon",
    "about", "following", "given", "attached", "below", "please"
}

CRITICAL_OPERATORS: Set[str] = {
    "not", "no", "never", "without", "dont", "don't", "cannot", "neither", "nor",
    "exactly", "strictly", "only", "unless", "except", "most", "least", "under",
    "over", "between", "maximum", "minimum", "every", "all", "first", "second",
    "third", "finally", "then", "next", "lastly", "step", "stage", "phase",
    "json", "markdown", "html", "csv", "bullet", "bullets", "french"
}


def _stem(word: str) -> str:
    """Deterministic lightweight lexical stemmer (encoder-free)."""
    w = re.sub(r"[^a-z0-9\$\.%#]", "", word.lower().strip())
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w[:6] if len(w) > 6 else w


class AttributionFaithfulnessEvaluator:
    """Measures rank alignment between predicted token importance and leave-one-out
    damage across circular, reference-guided proxy, and independent deterministic outcomes.
    """

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

    @staticmethod
    def _build_corpus_idf(prompts: List[PromptItem]) -> Dict[str, float]:
        """Builds deterministic inverse-document-frequency table across prompts."""
        doc_counts: Dict[str, int] = {}
        n_docs = max(1, len(prompts))
        for p in prompts:
            tokens = re.findall(r"[\$#]?\w+", p.prompt.lower())
            stems = {_stem(t) for t in tokens if _stem(t)}
            for s in stems:
                doc_counts[s] = doc_counts.get(s, 0) + 1
        return {
            s: math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
            for s, df in doc_counts.items()
        }

    @staticmethod
    def _extract_gt_requirement_stems(ground_truth: Optional[Any]) -> Dict[str, Set[str]]:
        """Extracts non-stopword lexical stems from each human ground-truth field."""
        if ground_truth is None:
            return {"intent": set(), "constraints": set(), "context": set(), "tone": set(), "steps": set()}

        def _stems_from_field(val: Any) -> Set[str]:
            if not val:
                return set()
            text = " ".join(val) if isinstance(val, list) else str(val)
            raw_tokens = re.findall(r"[\$#]?\w+", text.lower())
            return {_stem(t) for t in raw_tokens if t not in STOPWORDS and len(_stem(t)) >= 2}

        return {
            "intent": _stems_from_field(getattr(ground_truth, "intent", "")),
            "constraints": _stems_from_field(getattr(ground_truth, "constraints", [])),
            "context": _stems_from_field(getattr(ground_truth, "context", [])),
            "tone": _stems_from_field(getattr(ground_truth, "tone", [])),
            "steps": _stems_from_field(getattr(ground_truth, "steps", []))
        }

    @classmethod
    def compute_deterministic_requirement_score(
        cls,
        text: str,
        gt_stems: Dict[str, Set[str]],
        corpus_idf: Dict[str, float],
        orig_operator_counts: Dict[str, int]
    ) -> float:
        """Computes encoder-free deterministic requirement retention score R_det(text, G).
        
        Combines:
        1. Component-weighted human ground-truth stem coverage (weighted by corpus IDF):
           - Constraints & Steps: weight 2.5
           - Context & Intent: weight 1.8
           - Tone: weight 1.2
        2. Exact retention of critical functional operators (negations, numeric bounds,
           step sequence markers, format identifiers): weight 2.5
        3. Content-word IDF specificity for non-stopword domain terms: weight 0.5
        """
        raw_tokens = re.findall(r"[\$#]?\w+", text.lower())
        present_stems = {_stem(t) for t in raw_tokens if _stem(t)}

        total_score = 0.0

        # 1. Human ground-truth stem recall weighted by component criticality and IDF
        weights = {
            "constraints": 2.5,
            "steps": 2.5,
            "context": 1.8,
            "intent": 1.8,
            "tone": 1.2
        }
        for comp, w_mult in weights.items():
            req_stems = gt_stems.get(comp, set())
            for s in req_stems:
                if s in present_stems:
                    total_score += w_mult * corpus_idf.get(s, 1.5)

        # 2. Critical operator / number count preservation
        curr_op_counts: Dict[str, int] = {}
        for t in raw_tokens:
            is_num = any(c.isdigit() for c in t)
            if t in CRITICAL_OPERATORS or is_num:
                curr_op_counts[t] = curr_op_counts.get(t, 0) + 1

        for op, orig_cnt in orig_operator_counts.items():
            kept_cnt = min(orig_cnt, curr_op_counts.get(op, 0))
            total_score += 2.5 * kept_cnt * corpus_idf.get(_stem(op), 1.8)

        # 3. Domain content-word specificity (distinguishes content words from grammatical stopwords)
        for s in present_stems:
            if s not in STOPWORDS and len(s) >= 3:
                total_score += 0.5 * corpus_idf.get(s, 1.0)

        return total_score

    def measure_prompt_empirical_damage(
        self,
        prompt: str,
        ground_truth: Optional[Any] = None,
        corpus_idf: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Computes leave-one-out damage for each word in the prompt across three token-level outcomes:
        1. damage_self: Sentence-BERT LOO self-similarity drop (Circular for Embedding LOO)
        2. damage_ref_proxy: Reference-Guided Embedding Proxy drop (Sentence-BERT similarity to GT text)
        3. damage_det_req: Independent Deterministic Requirement drop (Encoder-free lexical/operator loss)
        """
        _, words_orig, _ = self.tokenizer_wrapper.tokenize_and_align(prompt)
        if not words_orig:
            return []

        if corpus_idf is None:
            corpus_idf = {}

        # 1. Original Sentence-BERT embedding
        orig_t = self.encoder.encode(prompt)
        orig_emb = orig_t.detach().cpu().numpy()
        norm_orig = np.linalg.norm(orig_emb)
        if norm_orig > 1e-9:
            orig_emb = orig_emb / norm_orig

        # 2. Reference-Guided Embedding Proxy target
        gt_emb = None
        sim_0_gt = 0.0
        if ground_truth is not None:
            intent = getattr(ground_truth, "intent", "") or ""
            constraints = getattr(ground_truth, "constraints", []) or []
            context = getattr(ground_truth, "context", []) or []
            steps = getattr(ground_truth, "steps", []) or []
            c_str = " ".join(constraints) if isinstance(constraints, list) else str(constraints)
            ctx_str = " ".join(context) if isinstance(context, list) else str(context)
            st_str = " ".join(steps) if isinstance(steps, list) else str(steps)
            gt_text = f"{intent} ; {c_str} ; {ctx_str} ; {st_str}".strip()

            if gt_text and gt_text != "; ; ;":
                gt_t = self.encoder.encode(gt_text)
                gt_emb = gt_t.detach().cpu().numpy()
                norm_gt = np.linalg.norm(gt_emb)
                if norm_gt > 1e-9:
                    gt_emb = gt_emb / norm_gt
                sim_0_gt = float(np.dot(orig_emb, gt_emb))

        # 3. Encoder-free Deterministic Requirement baseline
        gt_stems = self._extract_gt_requirement_stems(ground_truth)
        orig_raw_tokens = re.findall(r"[\$#]?\w+", prompt.lower())
        orig_op_counts: Dict[str, int] = {}
        for t in orig_raw_tokens:
            if t in CRITICAL_OPERATORS or any(c.isdigit() for c in t):
                orig_op_counts[t] = orig_op_counts.get(t, 0) + 1

        det_score_orig = self.compute_deterministic_requirement_score(
            prompt, gt_stems, corpus_idf, orig_op_counts
        )
        det_denom = max(1e-6, det_score_orig)

        results = []
        for word in words_orig:
            loo_prompt = TokenizerWrapper.remove_word_spans(prompt, [word])
            loo_t = self.encoder.encode(loo_prompt)
            loo_emb = loo_t.detach().cpu().numpy()
            norm_loo = np.linalg.norm(loo_emb)
            if norm_loo > 1e-9:
                loo_emb = loo_emb / norm_loo

            # Outcome 1: Self-consistency damage
            cos_sim_self = float(np.dot(orig_emb, loo_emb))
            damage_self = max(0.0, 1.0 - cos_sim_self)

            # Outcome 2: Reference-Guided Embedding Proxy damage
            if gt_emb is not None:
                sim_loo_gt = float(np.dot(loo_emb, gt_emb))
                damage_ref_proxy = max(0.0, sim_0_gt - sim_loo_gt)
            else:
                damage_ref_proxy = damage_self

            # Outcome 3: Independent Deterministic Requirement damage (Encoder-Free)
            det_score_loo = self.compute_deterministic_requirement_score(
                loo_prompt, gt_stems, corpus_idf, orig_op_counts
            )
            damage_det_req = max(0.0, (det_score_orig - det_score_loo) / det_denom)

            results.append({
                "word_id": word.word_id,
                "word": word.word,
                "start_char": word.start_char,
                "end_char": word.end_char,
                "loo_prompt": loo_prompt,
                "cos_sim_self": round(cos_sim_self, 5),
                "damage_self": round(damage_self, 5),
                "damage_gt": round(damage_ref_proxy, 5),  # backward-compatible alias
                "damage_ref_proxy": round(damage_ref_proxy, 5),
                "damage_det_req": round(damage_det_req, 5)
            })

        return results

    @staticmethod
    def _safe_rank_corr(x: List[float], y: List[float]) -> Tuple[float, float]:
        if len(set(x)) > 1 and len(set(y)) > 1:
            sp, _ = stats.spearmanr(x, y)
            kd, _ = stats.kendalltau(x, y)
            sp_f = float(sp) if not np.isnan(sp) else 0.0
            kd_f = float(kd) if not np.isnan(kd) else 0.0
            return sp_f, kd_f
        return 0.0, 0.0

    @staticmethod
    def _cluster_bootstrap_stats(
        prompt_vals: np.ndarray,
        n_bootstrap: int = 10000,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Computes hierarchical prompt-cluster bootstrap CIs, Fisher-z mean, t-test and Wilcoxon p-values."""
        n = len(prompt_vals)
        if n == 0:
            return {
                "mean": 0.0, "sd": 0.0, "median": 0.0, "fisher_z_mean": 0.0,
                "ci_low": 0.0, "ci_high": 0.0, "p_ttest": 1.0, "p_wilcoxon": 1.0
            }

        rng = np.random.default_rng(seed)
        boot_means = np.array([
            np.mean(rng.choice(prompt_vals, size=n, replace=True))
            for _ in range(n_bootstrap)
        ])
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        # Fisher z-transformation hierarchical mean
        clipped = np.clip(prompt_vals, -0.999, 0.999)
        z_vals = np.arctanh(clipped)
        fisher_mean = float(np.tanh(np.mean(z_vals)))

        if n > 1 and np.std(prompt_vals, ddof=1) > 1e-12:
            _, p_t = stats.ttest_1samp(prompt_vals, 0.0)
            p_ttest = float(p_t)
            try:
                nonzero = prompt_vals[np.abs(prompt_vals) > 1e-12]
                if len(nonzero) >= 5:
                    _, p_w = stats.wilcoxon(nonzero)
                    p_wilcox = float(p_w)
                else:
                    p_wilcox = p_ttest
            except Exception:
                p_wilcox = p_ttest
        else:
            p_ttest = 0.0 if abs(np.mean(prompt_vals)) > 0.99 else 1.0
            p_wilcox = p_ttest

        return {
            "mean": float(np.mean(prompt_vals)),
            "sd": float(np.std(prompt_vals, ddof=1)) if n > 1 else 0.0,
            "median": float(np.median(prompt_vals)),
            "fisher_z_mean": fisher_mean,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_ttest": p_ttest,
            "p_wilcoxon": p_wilcox
        }

    def evaluate_downstream_task_damage_alignment(
        self,
        downstream_df: pd.DataFrame,
        n_bootstrap: int = 10000
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, float]]]]:
        """Evaluates whether predicted importance of pruned tokens correlates with
        observed functional downstream task-success loss (S_orig - S_comp) across compressed variants.
        """
        if downstream_df is None or downstream_df.empty:
            return [], {}

        uncomp = downstream_df[downstream_df["method"] == "uncompressed"]
        comp = downstream_df[downstream_df["method"] != "uncompressed"]
        if uncomp.empty or comp.empty:
            return [], {}

        orig_prompts = dict(zip(uncomp["prompt_id"], uncomp["prompt_text"]))
        orig_scores = dict(zip(uncomp["prompt_id"], uncomp["task_success"]))
        orig_cats = dict(zip(uncomp["prompt_id"], uncomp["category"]))

        method_task_corrs: Dict[str, List[Dict[str, float]]] = {m: [] for m in self.scorers}

        for pid, group in comp.groupby("prompt_id"):
            if pid not in orig_prompts:
                continue
            orig_text = str(orig_prompts[pid])
            s_orig = float(orig_scores.get(pid, 1.0))
            cat = str(orig_cats.get(pid, "unknown"))

            _, words_orig, _ = self.tokenizer_wrapper.tokenize_and_align(orig_text)
            if not words_orig:
                continue

            # Observed downstream task loss for each compressed run of this task
            task_losses = [max(0.0, s_orig - float(row["task_success"])) for _, row in group.iterrows()]
            if len(set(task_losses)) <= 1:
                # If task never failed (or failed identically on all runs), rank correlation is undefined
                continue

            # Precompute word importance scores for each attribution scorer
            for m_name, scorer in self.scorers.items():
                sc_res = scorer.score_prompt(orig_text, prompt_id=str(pid))
                word_score_list = sc_res.word_scores
                total_imp = sum(ws.normalized_score for ws in word_score_list) or 1.0

                pred_losses = []
                for _, row in group.iterrows():
                    comp_text = str(row["prompt_text"]).lower()
                    comp_tokens = re.findall(r"[\$#]?\w+", comp_text)
                    comp_counts: Dict[str, int] = {}
                    for t in comp_tokens:
                        comp_counts[t] = comp_counts.get(t, 0) + 1

                    # Determine which original words were removed in this compressed variant
                    removed_imp = 0.0
                    for ws in word_score_list:
                        w_clean = re.sub(r"[^\w\$#]", "", ws.word.lower())
                        if not w_clean:
                            continue
                        if comp_counts.get(w_clean, 0) > 0:
                            comp_counts[w_clean] -= 1
                        else:
                            removed_imp += ws.normalized_score
                    pred_losses.append(removed_imp / total_imp)

                sp_r, kd_t = self._safe_rank_corr(pred_losses, task_losses)
                method_task_corrs[m_name].append({
                    "prompt_id": str(pid),
                    "category": cat,
                    "spearman_r": sp_r,
                    "kendall_tau": kd_t
                })

        summary_rows = []
        raw_p_vals = []
        temp_stats = []
        for m_name in self.scorers:
            corrs = method_task_corrs.get(m_name, [])
            if not corrs:
                continue
            sp_vals = np.array([c["spearman_r"] for c in corrs])
            kd_vals = np.array([c["kendall_tau"] for c in corrs])
            st_sp = self._cluster_bootstrap_stats(sp_vals, n_bootstrap=n_bootstrap, seed=self.seed)
            st_kd = self._cluster_bootstrap_stats(kd_vals, n_bootstrap=n_bootstrap, seed=self.seed)
            raw_p_vals.append(st_sp["p_ttest"])
            temp_stats.append((m_name, len(sp_vals), st_sp, st_kd))

        adj_p_vals = holm_bonferroni_correction(raw_p_vals) if raw_p_vals else []
        for idx, (m_name, n_tasks, st_sp, st_kd) in enumerate(temp_stats):
            summary_rows.append({
                "method": m_name,
                "evaluation_type": "Independent Functional Task Validation",
                "target_outcome": "Downstream Task-Success Loss",
                "n_prompts": n_tasks,
                "mean_spearman_r": round(st_sp["mean"], 4),
                "sd_spearman_r": round(st_sp["sd"], 4),
                "median_spearman_r": round(st_sp["median"], 4),
                "fisher_z_spearman_r": round(st_sp["fisher_z_mean"], 4),
                "spearman_ci_95": f"[{st_sp['ci_low']:.4f}, {st_sp['ci_high']:.4f}]",
                "mean_kendall_tau": round(st_kd["mean"], 4),
                "kendall_ci_95": f"[{st_kd['ci_low']:.4f}, {st_kd['ci_high']:.4f}]",
                "spearman_p_val": st_sp["p_ttest"],
                "spearman_p_holm": adj_p_vals[idx] if idx < len(adj_p_vals) else st_sp["p_ttest"],
                "wilcoxon_p_val": st_sp["p_wilcoxon"],
                "uses_embedding_encoder": False,
                "is_circular": False
            })

        return summary_rows, method_task_corrs

    def compute_pairwise_method_comparisons(
        self,
        per_outcome_corrs: Dict[str, Dict[str, List[Dict[str, float]]]],
        n_bootstrap: int = 10000
    ) -> pd.DataFrame:
        """Directly tests whether attribution methods differ significantly from each other
        (e.g., Gradient vs Embedding LOO) using paired per-prompt differences, cluster bootstrap CIs,
        paired t-tests, paired Wilcoxon signed-rank tests, and Holm-Bonferroni correction.
        """
        pairs = [
            ("gradient", "embedding"),
            ("gradient", "integrated_gradients"),
            ("embedding", "integrated_gradients"),
            ("gradient", "random"),
            ("embedding", "random"),
            ("integrated_gradients", "random")
        ]
        rows = []
        for outcome_name, method_map in per_outcome_corrs.items():
            outcome_rows = []
            raw_ps = []
            for m_a, m_b in pairs:
                list_a = method_map.get(m_a, [])
                list_b = method_map.get(m_b, [])
                if not list_a or not list_b or len(list_a) != len(list_b):
                    continue
                vals_a = np.array([c["spearman_r"] for c in list_a])
                vals_b = np.array([c["spearman_r"] for c in list_b])
                diffs = vals_a - vals_b
                st_diff = self._cluster_bootstrap_stats(diffs, n_bootstrap=n_bootstrap, seed=self.seed)
                raw_ps.append(st_diff["p_ttest"])
                outcome_rows.append({
                    "target_outcome": outcome_name,
                    "method_a": m_a,
                    "method_b": m_b,
                    "n_clusters": len(diffs),
                    "mean_r_a": round(float(np.mean(vals_a)), 4),
                    "mean_r_b": round(float(np.mean(vals_b)), 4),
                    "mean_diff_r": round(st_diff["mean"], 4),
                    "diff_ci_95": f"[{st_diff['ci_low']:.4f}, {st_diff['ci_high']:.4f}]",
                    "paired_ttest_p": st_diff["p_ttest"],
                    "paired_wilcoxon_p": st_diff["p_wilcoxon"]
                })
            adj_ps = holm_bonferroni_correction(raw_ps) if raw_ps else []
            for idx, r in enumerate(outcome_rows):
                p_h = adj_ps[idx] if idx < len(adj_ps) else r["paired_ttest_p"]
                r["p_holm_adjusted"] = p_h
                r["significant_holm_05"] = bool(p_h < 0.05)
                rows.append(r)

        return pd.DataFrame(rows)

    def evaluate_faithfulness(
        self,
        prompts: List[PromptItem],
        n_bootstrap: int = 10000,
        output_dir: str = "results/tables",
        downstream_df: Optional[pd.DataFrame] = None,
        all_corpus_prompts: Optional[List[PromptItem]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluates within-prompt importance-damage rank correlations across all outcomes,
        performing prompt-cluster bootstrapping, Holm-Bonferroni correction, and paired method tests.
        """
        print(f"[Faithfulness] Running importance-damage alignment on {len(prompts)} prompts (B={n_bootstrap} cluster resamples)...")

        corpus_for_idf = all_corpus_prompts if all_corpus_prompts else prompts
        corpus_idf = self._build_corpus_idf(corpus_for_idf)

        records = []
        method_corrs_self: Dict[str, List[Dict[str, float]]] = {m: [] for m in self.scorers}
        method_corrs_ref_proxy: Dict[str, List[Dict[str, float]]] = {m: [] for m in self.scorers}
        method_corrs_det_req: Dict[str, List[Dict[str, float]]] = {m: [] for m in self.scorers}

        for prompt_item in prompts:
            prompt_text = prompt_item.prompt
            damage_results = self.measure_prompt_empirical_damage(
                prompt_text, prompt_item.ground_truth, corpus_idf=corpus_idf
            )
            if len(damage_results) < 3:
                continue

            damage_self_scores = [d["damage_self"] for d in damage_results]
            damage_ref_scores = [d["damage_ref_proxy"] for d in damage_results]
            damage_det_scores = [d["damage_det_req"] for d in damage_results]

            for method_name, scorer in self.scorers.items():
                scoring_res = scorer.score_prompt(prompt_item.prompt, prompt_id=prompt_item.id)
                pred_map = {ws.word_id: ws.normalized_score for ws in scoring_res.word_scores}
                pred_scores = [pred_map.get(d["word_id"], 0.0) for d in damage_results]

                sp_self, kd_self = self._safe_rank_corr(pred_scores, damage_self_scores)
                sp_ref, kd_ref = self._safe_rank_corr(pred_scores, damage_ref_scores)
                sp_det, kd_det = self._safe_rank_corr(pred_scores, damage_det_scores)

                method_corrs_self[method_name].append({
                    "prompt_id": prompt_item.id,
                    "category": prompt_item.category,
                    "spearman_r": sp_self,
                    "kendall_tau": kd_self
                })
                method_corrs_ref_proxy[method_name].append({
                    "prompt_id": prompt_item.id,
                    "category": prompt_item.category,
                    "spearman_r": sp_ref,
                    "kendall_tau": kd_ref
                })
                method_corrs_det_req[method_name].append({
                    "prompt_id": prompt_item.id,
                    "category": prompt_item.category,
                    "spearman_r": sp_det,
                    "kendall_tau": kd_det
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
                        "damage_gt": d["damage_ref_proxy"],  # backward-compatible alias
                        "damage_ref_proxy": d["damage_ref_proxy"],
                        "damage_det_req": d["damage_det_req"],
                        "spearman_self": sp_self,
                        "spearman_gt": sp_ref,
                        "spearman_ref_proxy": sp_ref,
                        "spearman_det_req": sp_det
                    })

        detail_df = pd.DataFrame(records)

        # Build summary rows with Holm-Bonferroni correction per target outcome family
        outcome_specs = [
            (
                "Sentence-BERT LOO Drop",
                method_corrs_self,
                lambda m: "Self-Consistency (Circular)" if m == "embedding" else "Cross-Encoder LOO Prediction",
                True,
                lambda m: m == "embedding"
            ),
            (
                "Reference-Guided Embedding Proxy Drop",
                method_corrs_ref_proxy,
                lambda m: "Reference-Guided Embedding Proxy",
                True,
                lambda m: False
            ),
            (
                "Independent Deterministic Requirement Drop",
                method_corrs_det_req,
                lambda m: "Independent Deterministic Validation",
                False,
                lambda m: False
            )
        ]

        summary_rows = []
        for target_name, corr_dict, eval_type_fn, uses_emb, is_circ_fn in outcome_specs:
            family_rows = []
            raw_ps = []
            for method_name in self.scorers:
                corrs = corr_dict[method_name]
                if not corrs:
                    continue
                sp_vals = np.array([c["spearman_r"] for c in corrs])
                kd_vals = np.array([c["kendall_tau"] for c in corrs])

                st_sp = self._cluster_bootstrap_stats(sp_vals, n_bootstrap=n_bootstrap, seed=self.seed)
                st_kd = self._cluster_bootstrap_stats(kd_vals, n_bootstrap=n_bootstrap, seed=self.seed)
                raw_ps.append(st_sp["p_ttest"])

                family_rows.append({
                    "method": method_name,
                    "evaluation_type": eval_type_fn(method_name),
                    "target_outcome": target_name,
                    "n_prompts": len(sp_vals),
                    "mean_spearman_r": round(st_sp["mean"], 4),
                    "sd_spearman_r": round(st_sp["sd"], 4),
                    "median_spearman_r": round(st_sp["median"], 4),
                    "fisher_z_spearman_r": round(st_sp["fisher_z_mean"], 4),
                    "spearman_ci_95": f"[{st_sp['ci_low']:.4f}, {st_sp['ci_high']:.4f}]",
                    "mean_kendall_tau": round(st_kd["mean"], 4),
                    "kendall_ci_95": f"[{st_kd['ci_low']:.4f}, {st_kd['ci_high']:.4f}]",
                    "spearman_p_val": st_sp["p_ttest"],
                    "wilcoxon_p_val": st_sp["p_wilcoxon"],
                    "uses_embedding_encoder": uses_emb,
                    "is_circular": is_circ_fn(method_name)
                })

            adj_ps = holm_bonferroni_correction(raw_ps) if raw_ps else []
            for idx, r in enumerate(family_rows):
                r["spearman_p_holm"] = adj_ps[idx] if idx < len(adj_ps) else r["spearman_p_val"]
                summary_rows.append(r)

        per_outcome_corrs = {
            "Sentence-BERT LOO Drop": method_corrs_self,
            "Reference-Guided Embedding Proxy Drop": method_corrs_ref_proxy,
            "Independent Deterministic Requirement Drop": method_corrs_det_req
        }

        # Optional Outcome 4: Downstream Task-Success Loss alignment
        if downstream_df is not None and not downstream_df.empty:
            ds_rows, ds_corrs = self.evaluate_downstream_task_damage_alignment(
                downstream_df, n_bootstrap=n_bootstrap
            )
            summary_rows.extend(ds_rows)
            if ds_corrs:
                per_outcome_corrs["Downstream Task-Success Loss"] = ds_corrs

        summary_df = pd.DataFrame(summary_rows)
        pairwise_df = self.compute_pairwise_method_comparisons(per_outcome_corrs, n_bootstrap=n_bootstrap)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(out_path / "attribution_faithfulness.csv", index=False)
        summary_df.to_csv(out_path / "attribution_faithfulness_summary.csv", index=False)
        pairwise_df.to_csv(out_path / "attribution_pairwise_tests.csv", index=False)

        print(
            f"[Faithfulness] Generated detail ({len(detail_df)} rows), "
            f"summary ({len(summary_df)} rows), and pairwise tests ({len(pairwise_df)} rows)"
        )
        return detail_df, summary_df
