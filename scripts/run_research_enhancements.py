"""Research Enhancement Orchestrator.

Executes the four core scientific enhancements:
1. Semantic Stress Test Suite (7 critical dimensions, 14 minimal pairs)
2. Functional Downstream Task-Success Evaluation (deterministic validators & Groq LLM)
3. Attribution Faithfulness & Importance-Damage Alignment (Spearman & Kendall with CIs)
4. Dependency-Aware Span Pruning Ablation vs Independent Word Pruning
5. Generates enhanced plots and analytical tables
"""

from __future__ import annotations
import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def run_all_enhancements():
    print("=" * 70)
    print("SCULP RESEARCH ENHANCEMENTS: RUNNER")
    print("=" * 70)
    t0 = time.time()

    tables_dir = PROJECT_ROOT / "results" / "tables"
    plots_dir = PROJECT_ROOT / "results" / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Semantic Stress Test Suite
    # -------------------------------------------------------------
    print("\n[Phase 1] Executing Semantic Stress Test Suite...")
    stress_evaluator = StressTestEvaluator()
    stress_df = stress_evaluator.evaluate_all(
        ratios=[0.3, 0.5, 0.7],
        output_csv=str(tables_dir / "stress_test_results.csv")
    )
    print("[Phase 1] Generating stress test outcome plot...")
    plot_stress_test_outcomes(stress_df, str(plots_dir / "stress_test_outcomes.png"))

    # -------------------------------------------------------------
    # 2. Functional Downstream Task Evaluation
    # -------------------------------------------------------------
    print("\n[Phase 2] Executing Functional Downstream Task-Success Evaluation...")
    downstream_evaluator = DownstreamEvaluator()
    downstream_df = downstream_evaluator.run_benchmark(
        scorers=stress_evaluator.scorers,
        baselines=stress_evaluator.baselines,
        pruner=stress_evaluator.pruner,
        tasks=BENCHMARK_TASKS,
        ratios=[0.0, 0.3, 0.5, 0.7],
        output_csv=str(tables_dir / "downstream_results.csv")
    )
    print("[Phase 2] Generating downstream task retention plot...")
    plot_downstream_task_retention(downstream_df, str(plots_dir / "downstream_task_retention.png"))

    # -------------------------------------------------------------
    # 3. Attribution Faithfulness & Importance-Damage Alignment
    # -------------------------------------------------------------
    print("\n[Phase 3] Executing Attribution Faithfulness Evaluation...")
    dm = DatasetManager(str(PROJECT_ROOT / "data" / "processed" / "prompts.json"))
    # Stratified 15-prompt sample (3 from each of the 5 categories)
    all_prompts = dm.all_prompts()
    categories = sorted(list({p.category for p in all_prompts}))
    stratified_sample = []
    for cat in categories:
        cat_prompts = [p for p in all_prompts if p.category == cat]
        stratified_sample.extend(cat_prompts[:3])

    faithfulness_evaluator = AttributionFaithfulnessEvaluator()
    detail_df, summary_df = faithfulness_evaluator.evaluate_faithfulness(
        prompts=stratified_sample,
        n_bootstrap=1000,
        output_dir=str(tables_dir)
    )
    print("[Phase 3] Generating attribution faithfulness plot...")
    plot_attribution_faithfulness_summary(summary_df, str(plots_dir / "attribution_faithfulness.png"))

    # -------------------------------------------------------------
    # 4. Dependency-Aware Span Pruning Ablation
    # -------------------------------------------------------------
    print("\n[Phase 4] Executing Dependency-Aware Span Pruning Ablation...")
    span_pruner = DependencyAwarePruner()
    standard_pruner = stress_evaluator.pruner

    # Benchmark on the 14 stress prompts to quantify polarity reversal mitigation
    stress_prompts_list = load_stress_test_prompts()
    w_rev_count, s_rev_count, total_stress_tests = 0, 0, 0
    span_records = []

    for item in stress_prompts_list:
        sc_res = stress_evaluator.scorers["embedding"].score_prompt(item.prompt, prompt_id=item.id)
        for r in [0.3, 0.5, 0.7]:
            w_res = standard_pruner.prune(sc_res, target_removal=r)
            s_res = span_pruner.prune(sc_res, target_removal=r)
            
            w_outcome, _ = classify_outcome(item, w_res.compressed_prompt)
            s_outcome, _ = classify_outcome(item, s_res.compressed_prompt)

            if w_outcome == "POLARITY_REVERSED":
                w_rev_count += 1
            if s_outcome == "POLARITY_REVERSED":
                s_rev_count += 1
            total_stress_tests += 1

            span_records.append({
                "prompt_id": item.id,
                "category": item.feature_type,
                "target_removal": r,
                "word_pruned_text": w_res.compressed_prompt,
                "word_actual_removal": w_res.actual_removal,
                "word_outcome": w_outcome,
                "span_pruned_text": s_res.compressed_prompt,
                "span_actual_removal": s_res.actual_removal,
                "span_outcome": s_outcome,
                "spans_total": s_res.metadata.get("spans_total", 0),
                "spans_removed": s_res.metadata.get("spans_removed", 0)
            })

    print(f"[Phase 4] Polarity Inversion Mitigation across {total_stress_tests} stress tests:")
    print(f"  - Independent Word Pruning Reversals: {w_rev_count} ({w_rev_count/total_stress_tests*100:.1f}%)")
    print(f"  - Dependency-Aware Span Pruning Reversals: {s_rev_count} ({s_rev_count/total_stress_tests*100:.1f}%)")

    span_df = pd.DataFrame(span_records)
    span_df.to_csv(tables_dir / "dependency_pruning_results.csv", index=False)
    print(f"[Phase 4] Saved {len(span_df)} span pruning ablation records to {tables_dir / 'dependency_pruning_results.csv'}")

    # -------------------------------------------------------------
    # 5. Enhanced Rate-Distortion & Protocol Plots
    # -------------------------------------------------------------
    print("\n[Phase 5] Updating Rate-Distortion & Protocol Plots...")
    # Read full 5,400 fidelity results if available
    fid_path = PROJECT_ROOT / "results" / "metrics" / "fidelity_results.csv"
    if fid_path.exists():
        fid_df = pd.read_csv(fid_path)
        plot_actual_rate_distortion_with_auc(fid_df, str(plots_dir / "actual_rate_distortion_auc.png"))
        print("[Phase 5] Generated actual_rate_distortion_auc.png")

    pilot_path = PROJECT_ROOT / "results" / "pilot" / "pilot_evaluation_results.csv"
    if pilot_path.exists():
        pilot_df = pd.read_csv(pilot_path)
        plot_protocols_comparison(pilot_df, str(plots_dir / "protocol_comparison.png"))
        print("[Phase 5] Generated protocol_comparison.png")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"ALL RESEARCH ENHANCEMENTS COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_enhancements()
