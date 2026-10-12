"""Semantic Collapse threshold and breakpoint detection module.
Implements:
1. Local steepest-drop breakpoint detection (discrete derivative Delta_i = SF_i - SF_{i-1}).
2. Operational threshold metrics (first compression level where composite fidelity drops below 0.8 and 0.7).
3. Explicit handling of uncrossed thresholds and actual achieved compression ratios.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from src.analysis.statistics import bootstrap_ci


def detect_collapse_breakpoint(
    compression_ratios: List[float],
    fidelity_values: List[float]
) -> Dict[str, Any]:
    """
    Detects the semantic collapse breakpoint along the rate-distortion curve.
    Identifies maximum negative marginal gradient Delta_i = SF_i - SF_{i-1}.
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


def detect_operational_threshold(
    compression_ratios: List[float],
    fidelity_values: List[float],
    threshold: float
) -> Optional[float]:
    """
    Identifies the first compression ratio where fidelity falls strictly below threshold.
    Returns None if the threshold is never crossed across the evaluated compression regime.
    These are operational definitions rather than universal semantic-collapse boundaries.
    """
    for cr, fid in zip(compression_ratios, fidelity_values):
        if fid < threshold:
            return round(float(cr), 4)
    return None


def compute_collapse_thresholds_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes both local gradient collapse and operational thresholds (tau_0.8, tau_0.7)
    across methods and categories, incorporating both target and actual compression ratios.
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

            # Group by target_removal and compute mean actual_removal and semantic_fidelity
            grouped = sub.groupby("target_removal").agg({
                "actual_removal": "mean",
                "semantic_fidelity": "mean"
            }).reset_index().sort_values("target_removal")

            cr_list = grouped["target_removal"].tolist()
            actual_cr_list = grouped["actual_removal"].tolist()
            sf_list = grouped["semantic_fidelity"].tolist()

            bp = detect_collapse_breakpoint(cr_list, sf_list)
            tau_80 = detect_operational_threshold(cr_list, sf_list, threshold=0.80)
            tau_70 = detect_operational_threshold(cr_list, sf_list, threshold=0.70)
            actual_tau_80 = detect_operational_threshold(actual_cr_list, sf_list, threshold=0.80)
            actual_tau_70 = detect_operational_threshold(actual_cr_list, sf_list, threshold=0.70)

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
                "drop_magnitude": bp["drop_magnitude"],
                "operational_tau_0_80": tau_80 if tau_80 is not None else "never_crossed",
                "operational_tau_0_70": tau_70 if tau_70 is not None else "never_crossed",
                "actual_tau_0_80": actual_tau_80 if actual_tau_80 is not None else "never_crossed",
                "actual_tau_0_70": actual_tau_70 if actual_tau_70 is not None else "never_crossed",
            })

    return pd.DataFrame(records)
