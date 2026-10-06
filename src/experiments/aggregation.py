"""Aggregation utilities for generating experiment tables and failure case reports."""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from src.analysis.rate_distortion import compute_rate_distortion_summary, compute_method_comparisons
from src.analysis.collapse import compute_collapse_thresholds_for_df


def aggregate_experiment_results(
    fidelity_df: pd.DataFrame,
    word_scores_df: pd.DataFrame,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Saves all primary tables required by the specification:
    - results/metrics/fidelity_results.csv
    - results/tables/method_comparison.csv
    - results/tables/category_comparison.csv
    - results/tables/collapse_thresholds.csv
    - results/scores/token_importance.csv
    - results/failure_cases.json
    """
    out = Path(output_dir)
    tables_dir = out / "tables"
    metrics_dir = out / "metrics"
    scores_dir = out / "scores"

    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save core fidelity results
    fidelity_csv = metrics_dir / "fidelity_results.csv"
    fidelity_df.to_csv(fidelity_csv, index=False)

    # 2. Save token importance
    scores_csv = scores_dir / "token_importance.csv"
    word_scores_df.to_csv(scores_csv, index=False)

    # 3. Method comparison table
    method_comp_df = compute_method_comparisons(fidelity_df)
    method_comp_csv = tables_dir / "method_comparison.csv"
    method_comp_df.to_csv(method_comp_csv, index=False)

    # 4. Category comparison table
    rd_summary_df = compute_rate_distortion_summary(fidelity_df)
    category_comp_csv = tables_dir / "category_comparison.csv"
    rd_summary_df.to_csv(category_comp_csv, index=False)

    # 5. Collapse thresholds table
    collapse_df = compute_collapse_thresholds_for_df(fidelity_df)
    collapse_csv = tables_dir / "collapse_thresholds.csv"
    collapse_df.to_csv(collapse_csv, index=False)

    # 6. Failure case extraction:
    # High embedding similarity (> 0.85) BUT low constraint fidelity (< 0.50)
    failure_cases = []
    suspicious = fidelity_df[
        (fidelity_df["embedding_similarity"] > 0.82) &
        (fidelity_df["constraints"] < 0.50) &
        (fidelity_df["category"] == "constraint")
    ]

    for _, row in suspicious.iterrows():
        failure_cases.append({
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "method": row["method"],
            "target_removal": row["target_removal"],
            "compressed_prompt": row.get("compressed_prompt", ""),
            "embedding_similarity": float(row["embedding_similarity"]),
            "constraint_fidelity": float(row["constraints"]),
            "failure_type": "negation_or_constraint_collapse_under_high_embedding_similarity"
        })

    failure_path = out / "failure_cases.json"
    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failure_cases, f, indent=2)

    return {
        "fidelity_results": str(fidelity_csv),
        "token_importance": str(scores_csv),
        "method_comparison": str(method_comp_csv),
        "category_comparison": str(category_comp_csv),
        "collapse_thresholds": str(collapse_csv),
        "failure_cases": str(failure_path),
        "n_failures": len(failure_cases)
    }
