"""Research Enhancement & Final Statistical Validity Orchestrator.

Executes the validated experimental suite:
1. Semantic Stress Test Suite (7 critical dimensions, 14 minimal pairs, 252 runs) + Reversal Audit
2. Functional Downstream Task-Success Evaluation (15 stratified tasks: 3 per category, 285 runs),
   Matched Actual-Removal AUC comparison, Non-Monotonicity Audit, and Repeated-Trial Variability Study
3. Attribution Faithfulness & Importance-Damage Alignment (30 stratified prompts, 4 outcome targets:
   Circular Self-Consistency, Reference-Guided Embedding Proxy, Independent Deterministic Requirement Drop,
   and Downstream Task-Success Loss) with prompt-cluster bootstrapping (B=10,000), Holm-Bonferroni
   correction, and direct paired tests (Gradient vs Embedding LOO)
4. Rule-Based Critical-Span Pruning vs Ordinary Word Pruning at Matched Token Budgets (126 paired runs
   across Embedding, Gradient, and Integrated Gradients scorers) with cluster-bootstrap uncertainty
5. Regenerates publication figures
"""

from __future__ import annotations
import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.data.dataset import DatasetManager
from src.data.schema import PromptItem
from src.scoring.random import RandomScorer
from src.scoring.embedding import EmbeddingLeaveOneOutScorer
from src.scoring.gradient import GradientScorer
from src.scoring.integrated_gradients import IntegratedGradientScorer
from src.baselines.llmlingua import LLMLinguaCompressor
from src.baselines.llmlingua2 import LLMLingua2Compressor
from src.pruning.pruner import PromptPruner
from src.pruning.dependency_pruner import DependencyAwarePruner
from src.evaluation.stress_tests import StressTestEvaluator, load_stress_test_prompts, classify_outcome
from src.evaluation.downstream import DownstreamEvaluator, BENCHMARK_TASKS
from src.analysis.attribution_faithfulness import AttributionFaithfulnessEvaluator
from src.analysis.plots import (
    plot_actual_rate_distortion_with_auc,
    plot_downstream_task_retention,
    plot_protocols_comparison,
    plot_attribution_faithfulness_summary,
    plot_stress_test_outcomes
)


def _cluster_boot_rate_ci(
    df: pd.DataFrame,
    cluster_col: str,
    val_col: str,
    target_val: str,
    n_boot: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Computes cluster-level proportion and defensible 95% CI.
    When cluster rates observe zero events, computes the Wilson score interval
    upper bound on the number of clusters to replace inappropriate zero-width intervals.
    """
    clusters = df[cluster_col].unique()
    k = len(clusters)
    if k == 0:
        return 0.0, 0.0, 0.0
    cluster_rates = np.array([
        np.mean(df[df[cluster_col] == c][val_col] == target_val)
        for c in clusters
    ])
    mean_rate = float(np.mean(cluster_rates))

    if np.all(cluster_rates == 0.0):
        # Defensible Wilson score interval for 0 events across k clusters (z=1.95996)
        z = 1.95996
        wilson_upper = (z * z) / (k + z * z)
        return 0.0, 0.0, float(wilson_upper)
    elif np.all(cluster_rates == 1.0):
        z = 1.95996
        wilson_lower = k / (k + z * z)
        return 1.0, float(wilson_lower), 1.0

    rng = np.random.default_rng(seed)
    boot_means = [
        np.mean(rng.choice(cluster_rates, size=k, replace=True))
        for _ in range(n_boot)
    ]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return mean_rate, float(ci_low), float(ci_high)


def run_all_enhancements():
    print("=" * 75)
    print("SCULP FINAL STATISTICAL & EXPERIMENTAL VALIDITY RUNNER")
    print("=" * 75)
    t0 = time.time()

    tables_dir = PROJECT_ROOT / "results" / "tables"
    plots_dir = PROJECT_ROOT / "results" / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Semantic Stress Test Suite (252 Runs + Reversal Audit)
    # -------------------------------------------------------------
    print("\n[Phase 1] Executing Semantic Stress Test Suite (14 minimal pairs x 6 methods x 3 CRs)...")
    stress_evaluator = StressTestEvaluator()
    stress_df = stress_evaluator.evaluate_all(
        ratios=[0.3, 0.5, 0.7],
        output_csv=str(tables_dir / "stress_test_results.csv")
    )
    rev_audit_df = stress_df[stress_df["outcome"] == "POLARITY_REVERSED"].copy()
    rev_audit_df.to_csv(tables_dir / "polarity_reversals_audit.csv", index=False)
    print(f"[Phase 1] Verified {len(rev_audit_df)} genuine polarity reversals (saved to polarity_reversals_audit.csv)")
    plot_stress_test_outcomes(stress_df, str(plots_dir / "stress_test_outcomes.png"))

    # -------------------------------------------------------------
    # 2. Functional Downstream Task Evaluation (15 Stratified Tasks)
    # -------------------------------------------------------------
    print("\n[Phase 2] Executing Functional Downstream Task Evaluation (15 stratified tasks: 3 per category)...")
    downstream_evaluator = DownstreamEvaluator()
    downstream_df = downstream_evaluator.run_benchmark(
        scorers=stress_evaluator.scorers,
        baselines=stress_evaluator.baselines,
        pruner=stress_evaluator.pruner,
        tasks=BENCHMARK_TASKS,
        ratios=[0.0, 0.3, 0.5, 0.7],
        output_csv=str(tables_dir / "downstream_results.csv")
    )
    print("[Phase 2] Executing Downstream Generation Variability Study (K=3 trials at T=0.0 and T=0.3)...")
    downstream_evaluator.run_variability_study(
        scorers=stress_evaluator.scorers,
        pruner=stress_evaluator.pruner,
        subset_task_ids=["p005", "p024", "p041", "p061", "p081"],
        n_trials=3,
        output_csv=str(tables_dir / "downstream_variability.csv")
    )
    plot_downstream_task_retention(downstream_df, str(plots_dir / "downstream_task_retention.png"))

    # -------------------------------------------------------------
    # 3. Attribution Faithfulness & Importance-Damage Alignment
    # -------------------------------------------------------------
    print("\n[Phase 3] Executing Attribution Faithfulness Evaluation on 30 Stratified Prompts (6 per category)...")
    dm = DatasetManager(str(PROJECT_ROOT / "data" / "processed" / "prompts.json"))
    all_prompts = dm.all_prompts()
    categories = sorted(list({p.category for p in all_prompts}))
    stratified_30 = []
    for cat in categories:
        cat_prompts = [p for p in all_prompts if p.category == cat]
        stratified_30.extend(cat_prompts[:6])

    faithfulness_evaluator = AttributionFaithfulnessEvaluator()
    detail_df, summary_df = faithfulness_evaluator.evaluate_faithfulness(
        prompts=stratified_30,
        n_bootstrap=10000,
        output_dir=str(tables_dir),
        downstream_df=downstream_df,
        all_corpus_prompts=all_prompts
    )
    plot_attribution_faithfulness_summary(summary_df, str(plots_dir / "attribution_faithfulness.png"))

    print("[Phase 3b] Executing Deterministic Requirement Metric Sensitivity Analysis (4 Weighting Schemes)...")
    faithfulness_evaluator.run_requirement_sensitivity_analysis(
        prompts=stratified_30,
        attr_df=detail_df,
        output_csv=str(tables_dir / "requirement_sensitivity_analysis.csv")
    )

    # -------------------------------------------------------------
    # 4. Rule-Based Critical-Span Pruning vs Ordinary Word Pruning
    # -------------------------------------------------------------
    print("\n[Phase 4] Executing Rule-Based Critical-Span Pruning Experiment at Matched Token Budgets...")
    span_pruner = DependencyAwarePruner()
    standard_pruner = stress_evaluator.pruner
    stress_prompts_list = load_stress_test_prompts()

    span_records = []
    scorer_names = ["embedding", "gradient", "integrated_gradients"]

    for s_name in scorer_names:
        scorer = stress_evaluator.scorers[s_name]
        for item in stress_prompts_list:
            sc_res = scorer.score_prompt(item.prompt, prompt_id=item.id)
            for r in [0.3, 0.5, 0.7]:
                w_res = standard_pruner.prune(sc_res, target_removal=r)
                # Match the exact token removal budget achieved by ordinary word pruning
                w_tokens_removed = max(1, w_res.original_tokens - w_res.remaining_tokens)
                s_res = span_pruner.prune(
                    sc_res,
                    target_removal=r,
                    target_tokens_to_remove_override=w_tokens_removed
                )

                w_outcome, w_rat = classify_outcome(item, w_res.compressed_prompt)
                s_outcome, s_rat = classify_outcome(item, s_res.compressed_prompt)

                span_records.append({
                    "prompt_id": item.id,
                    "pair_id": item.pair_id,
                    "category": item.feature_type,
                    "variant": item.variant,
                    "scorer": s_name,
                    "target_removal": r,
                    "original_prompt": item.prompt,
                    "critical_requirement": item.critical_requirement,
                    "word_pruned_text": w_res.compressed_prompt,
                    "word_actual_removal": round(w_res.actual_removal, 4),
                    "word_outcome": w_outcome,
                    "word_rationale": w_rat,
                    "span_pruned_text": s_res.compressed_prompt,
                    "span_actual_removal": round(s_res.actual_removal, 4),
                    "span_outcome": s_outcome,
                    "span_rationale": s_rat,
                    "compression_diff": round(s_res.actual_removal - w_res.actual_removal, 4),
                    "spans_total": s_res.metadata.get("spans_total", 0),
                    "spans_removed": s_res.metadata.get("spans_removed", 0)
                })

    span_df = pd.DataFrame(span_records)
    span_df.to_csv(tables_dir / "dependency_pruning_results.csv", index=False)

    # Generate summary table with cluster-bootstrapped CIs across prompts
    summary_span_rows = []
    paired_span_rows = []

    for scope_label, sub_df in [("all_scorers_pooled", span_df)] + [(sn, span_df[span_df["scorer"] == sn]) for sn in scorer_names]:
        n_runs = len(sub_df)
        n_clusters = len(sub_df["pair_id"].unique())

        for mode_prefix, col_out, col_cr in [
            ("ordinary_word_pruning", "word_outcome", "word_actual_removal"),
            ("rule_based_span_pruning", "span_outcome", "span_actual_removal")
        ]:
            rev_cnt = int((sub_df[col_out] == "POLARITY_REVERSED").sum())
            amb_cnt = int((sub_df[col_out] == "AMBIGUOUS").sum())
            rem_cnt = int((sub_df[col_out] == "REMOVED").sum())
            pres_cnt = int((sub_df[col_out] == "PRESERVED").sum())

            rev_m, rev_lo, rev_hi = _cluster_boot_rate_ci(sub_df, "pair_id", col_out, "POLARITY_REVERSED")
            pres_m, pres_lo, pres_hi = _cluster_boot_rate_ci(sub_df, "pair_id", col_out, "PRESERVED")
            amb_m, amb_lo, amb_hi = _cluster_boot_rate_ci(sub_df, "pair_id", col_out, "AMBIGUOUS")
            rem_m, rem_lo, rem_hi = _cluster_boot_rate_ci(sub_df, "pair_id", col_out, "REMOVED")

            summary_span_rows.append({
                "scorer_scope": scope_label,
                "pruning_mode": mode_prefix,
                "cluster_unit": "pair_id (N=7 minimal pairs)",
                "n_clusters": n_clusters,
                "n_evaluations": n_runs,
                "mean_actual_removal": round(float(sub_df[col_cr].mean()), 4),
                "preserved_count": pres_cnt,
                "preserved_rate": round(pres_m, 4),
                "preserved_ci_95": f"[{pres_lo:.4f}, {pres_hi:.4f}]",
                "polarity_reversed_count": rev_cnt,
                "polarity_reversed_rate": round(rev_m, 4),
                "polarity_reversed_ci_95": f"[{rev_lo:.4f}, {rev_hi:.4f}]",
                "ambiguous_count": amb_cnt,
                "ambiguous_rate": round(amb_m, 4),
                "ambiguous_ci_95": f"[{amb_lo:.4f}, {amb_hi:.4f}]",
                "removed_count": rem_cnt,
                "removed_rate": round(rem_m, 4),
                "removed_ci_95": f"[{rem_lo:.4f}, {rem_hi:.4f}]"
            })

        # Paired minimal-pair cluster comparison (N=7 semantic contrast pairs)
        p_df = sub_df.groupby("pair_id").apply(lambda g: pd.Series({
            "w_rev": (g["word_outcome"] == "POLARITY_REVERSED").mean(),
            "s_rev": (g["span_outcome"] == "POLARITY_REVERSED").mean(),
            "w_pres": (g["word_outcome"] == "PRESERVED").mean(),
            "s_pres": (g["span_outcome"] == "PRESERVED").mean(),
            "w_rem": g["word_actual_removal"].mean(),
            "s_rem": g["span_actual_removal"].mean()
        })).reset_index()

        diff_rev = p_df["w_rev"] - p_df["s_rev"]
        diff_pres = p_df["s_pres"] - p_df["w_pres"]
        diff_rem = p_df["s_rem"] - p_df["w_rem"]

        t_res_rev = stats.ttest_1samp(diff_rev, 0.0) if np.std(diff_rev, ddof=1) > 1e-12 else (0.0, 1.0)
        t_res_pres = stats.ttest_1samp(diff_pres, 0.0) if np.std(diff_pres, ddof=1) > 1e-12 else (0.0, 1.0)

        w_res_rev = stats.wilcoxon(p_df["w_rev"], p_df["s_rev"]) if (diff_rev != 0).any() else (0.0, 1.0)
        w_p_rev = float(w_res_rev.pvalue) if hasattr(w_res_rev, "pvalue") else float(w_res_rev[1])
        w_res_pres = stats.wilcoxon(p_df["s_pres"], p_df["w_pres"]) if (diff_pres != 0).any() else (0.0, 1.0)
        w_p_pres = float(w_res_pres.pvalue) if hasattr(w_res_pres, "pvalue") else float(w_res_pres[1])

        rng = np.random.default_rng(42)
        n_c = len(p_df)
        boot_diff_rev = [np.mean(rng.choice(diff_rev.to_numpy(), size=n_c, replace=True)) for _ in range(10000)]
        boot_diff_pres = [np.mean(rng.choice(diff_pres.to_numpy(), size=n_c, replace=True)) for _ in range(10000)]
        ci_rev_low, ci_rev_high = np.percentile(boot_diff_rev, [2.5, 97.5])
        ci_pres_low, ci_pres_high = np.percentile(boot_diff_pres, [2.5, 97.5])

        paired_span_rows.append({
            "scorer_scope": scope_label,
            "cluster_unit": "pair_id (N=7 minimal pairs)",
            "n_clusters": n_clusters,
            "n_evaluations": n_runs,
            "mean_word_actual_removal": round(float(p_df["w_rem"].mean()), 4),
            "mean_span_actual_removal": round(float(p_df["s_rem"].mean()), 4),
            "removal_diff": round(float(diff_rem.mean()), 4),
            "word_reversed_rate": round(float(p_df["w_rev"].mean()), 4),
            "span_reversed_rate": round(float(p_df["s_rev"].mean()), 4),
            "paired_reversal_reduction": round(float(diff_rev.mean()), 4),
            "paired_reversal_reduction_ci_95": f"[{ci_rev_low:.4f}, {ci_rev_high:.4f}]",
            "paired_reversal_p_val": float(t_res_rev.pvalue) if hasattr(t_res_rev, "pvalue") else float(t_res_rev[1]),
            "paired_reversal_wilcoxon_p": round(w_p_rev, 4),
            "word_preserved_rate": round(float(p_df["w_pres"].mean()), 4),
            "span_preserved_rate": round(float(p_df["s_pres"].mean()), 4),
            "paired_preservation_gain": round(float(diff_pres.mean()), 4),
            "paired_preservation_gain_ci_95": f"[{ci_pres_low:.4f}, {ci_pres_high:.4f}]",
            "paired_preservation_p_val": float(t_res_pres.pvalue) if hasattr(t_res_pres, "pvalue") else float(t_res_pres[1]),
            "paired_preservation_wilcoxon_p": round(w_p_pres, 4)
        })

    span_sum_df = pd.DataFrame(summary_span_rows)
    span_sum_df.to_csv(tables_dir / "dependency_pruning_summary.csv", index=False)

    paired_span_df = pd.DataFrame(paired_span_rows)
    paired_span_df.to_csv(tables_dir / "dependency_pruning_paired_tests.csv", index=False)

    # Save audit table of all cases where either word or span pruning caused a polarity reversal
    rev_cases_span = span_df[
        (span_df["word_outcome"] == "POLARITY_REVERSED") | (span_df["span_outcome"] == "POLARITY_REVERSED")
    ].copy()
    rev_cases_span.to_csv(tables_dir / "dependency_pruning_reversals_audit.csv", index=False)

    print(f"[Phase 4] Saved {len(span_df)} paired span-pruning evaluations, summary ({len(span_sum_df)} rows), paired tests ({len(paired_span_df)} rows), and {len(rev_cases_span)} audited reversals.")

    # -------------------------------------------------------------
    # 5. Enhanced Rate-Distortion & Protocol Plots
    # -------------------------------------------------------------
    print("\n[Phase 5] Updating Rate-Distortion & Protocol Plots...")
    fid_path = PROJECT_ROOT / "results" / "metrics" / "fidelity_results.csv"
    if fid_path.exists():
        fid_df = pd.read_csv(fid_path)
        plot_actual_rate_distortion_with_auc(fid_df, str(plots_dir / "actual_rate_distortion_auc.png"))

    pilot_path = PROJECT_ROOT / "results" / "pilot" / "pilot_evaluation_results.csv"
    if pilot_path.exists():
        pilot_df = pd.read_csv(pilot_path)
        plot_protocols_comparison(pilot_df, str(plots_dir / "protocol_comparison.png"))

    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f"ALL VALIDATION EXPERIMENTS COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds!")
    print("=" * 75)


if __name__ == "__main__":
    run_all_enhancements()
