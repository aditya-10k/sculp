"""Semantic Collapse threshold and breakpoint detection module."""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from src.analysis.statistics import bootstrap_ci


def detect_collapse_breakpoint(
    compression_ratios: List[float],
    fidelity_values: List[float]
) -> Dict[str, Any]:
    """
    Detects the semantic collapse breakpoint along the rate-distortion curve.
    Identifies maximum negative delta Delta_i = SF_i - SF_{i-1}.
    """
    if len(fidelity_values) < 2:
        return {
            "collapse_ratio": 0.0,
            "pre_collapse_fidelity": 1.0,
            "post_collapse_fidelity": 1.0,
            "drop_magnitude": 0.0
        }

    deltas = []
    for i in range(1, len(fidelity_values)):
        delta = fidelity_values[i] - fidelity_values[i - 1]
        deltas.append((delta, i))

    # Find largest negative drop (minimum delta)
    min_delta, max_drop_idx = min(deltas, key=lambda x: x[0])

    collapse_ratio = compression_ratios[max_drop_idx]
    pre_fidelity = fidelity_values[max_drop_idx - 1]
    post_fidelity = fidelity_values[max_drop_idx]
    drop_magnitude = abs(min_delta) if min_delta < 0 else 0.0

    return {
        "collapse_ratio": round(float(collapse_ratio), 4),
        "pre_collapse_fidelity": round(float(pre_fidelity), 4),
        "post_collapse_fidelity": round(float(post_fidelity), 4),
        "drop_magnitude": round(float(drop_magnitude), 4)
    }


def compute_collapse_thresholds_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes collapse threshold across methods and categories in the experiment dataframe.
    """
    records = []
    methods = df["method"].unique()
    categories = list(df["category"].unique()) + ["all"]

    for method in methods:
        for cat in categories:
            sub = df[df["method"] == method]
            if cat != "all":
                sub = sub[sub["category"] == cat]

            if sub.empty:
                continue

            # Group by target_removal and compute mean semantic_fidelity
            grouped = sub.groupby("target_removal")["semantic_fidelity"].mean().reset_index()
            grouped = grouped.sort_values("target_removal")

            cr_list = grouped["target_removal"].tolist()
            sf_list = grouped["semantic_fidelity"].tolist()

            bp = detect_collapse_breakpoint(cr_list, sf_list)

            # Prompt-level bootstrap for collapse ratio CI
            prompt_collapses = []
            for pid, pgroup in sub.groupby("prompt_id"):
                pgroup = pgroup.sort_values("target_removal")
                p_bp = detect_collapse_breakpoint(
                    pgroup["target_removal"].tolist(),
                    pgroup["semantic_fidelity"].tolist()
                )
                prompt_collapses.append(p_bp["collapse_ratio"])

            _, ci_low, ci_high = bootstrap_ci(prompt_collapses, n_boot=2000)

            records.append({
                "method": method,
                "category": cat,
                "collapse_ratio": bp["collapse_ratio"],
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "pre_collapse_fidelity": bp["pre_collapse_fidelity"],
                "post_collapse_fidelity": bp["post_collapse_fidelity"],
                "drop_magnitude": bp["drop_magnitude"]
            })

    return pd.DataFrame(records)
