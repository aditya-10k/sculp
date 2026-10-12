"""Statistical testing, effect size estimation, and bootstrap confidence intervals.
Implements paired repeated-measures analysis, prompt-level cluster bootstrapping,
and Holm-Bonferroni family-wise error rate control.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import scipy.stats as stats
import pandas as pd


def bootstrap_ci(
    data: List[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes empirical bootstrap confidence interval for the mean of 1D independent samples.
    Returns: (mean, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    if n == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    mean_val = float(np.mean(arr))
    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return round(mean_val, 4), round(lower, 4), round(upper, 4)


def prompt_cluster_bootstrap_ci(
    prompt_grouped_diffs: Dict[str, float] | List[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes cluster bootstrap confidence intervals resampling at the PROMPT LEVEL (the independent unit).
    This strictly prevents pseudo-replication when aggregating across repeated measurements.
    """
    if isinstance(prompt_grouped_diffs, dict):
        diffs = list(prompt_grouped_diffs.values())
    else:
        diffs = list(prompt_grouped_diffs)

    return bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate two-sample pooled Cohen's d effect size.
    """
    x1 = np.array(group1, dtype=np.float64)
    x2 = np.array(group2, dtype=np.float64)
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0

    s1_sq = np.var(x1, ddof=1)
    s2_sq = np.var(x2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
    if pooled_sd < 1e-9:
        return 0.0
    return float((np.mean(x1) - np.mean(x2)) / pooled_sd)


def paired_cohens_d(x: List[float], y: List[float]) -> float:
    """
    Calculate paired Cohen's d_z = mean(x - y) / std(x - y).
    Appropriate effect size for matched within-prompt paired observations.
    """
    arr_x = np.array(x, dtype=np.float64)
    arr_y = np.array(y, dtype=np.float64)
    diff = arr_x - arr_y
    if len(diff) < 2:
        return 0.0
    sd_diff = np.std(diff, ddof=1)
    if sd_diff < 1e-9:
        return 0.0
    return float(np.mean(diff) / sd_diff)


def paired_difference_test(x: List[float], y: List[float]) -> Dict[str, Any]:
    """
    Performs Wilcoxon signed-rank test on matched paired observations.
    """
    arr_x = np.array(x, dtype=np.float64)
    arr_y = np.array(y, dtype=np.float64)
    diff = arr_x - arr_y

    mean_diff = float(np.mean(diff))
    if np.allclose(diff, 0.0):
        return {
            "test": "wilcoxon",
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_difference": 0.0,
            "significant_005": False
        }

    try:
        res = stats.wilcoxon(arr_x, arr_y, alternative='two-sided', zero_method='pratt')
        stat = float(res.statistic)
        p_val = float(res.pvalue)
        test_name = "wilcoxon"
    except Exception:
        res = stats.ttest_rel(arr_x, arr_y)
        stat = float(res.statistic)
        p_val = float(res.pvalue)
        test_name = "paired_ttest"

    return {
        "test": test_name,
        "statistic": round(stat, 4),
        "p_value": float(p_val),
        "mean_difference": round(mean_diff, 4),
        "significant_005": bool(p_val < 0.05)
    }


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni step-down correction to control Family-Wise Error Rate (FWER).
    Ensures monotonicity and strictly valid adjusted p-values.
    """
    m = len(p_values)
    if m == 0:
        return []
    if m == 1:
        return [min(1.0, max(0.0, float(p_values[0])))]

    # Pair each p-value with its original index
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_indexed = []

    running_max = 0.0
    for rank_idx, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank_idx
        adj_p = min(1.0, max(0.0, p_val * multiplier))
        running_max = max(running_max, adj_p)
        running_max = min(1.0, running_max)
        adjusted_indexed.append((orig_idx, running_max))

    # Restore original ordering
    adjusted_indexed.sort(key=lambda x: x[0])
    return [round(p, 6) for _, p in adjusted_indexed]


def compute_repeated_measures_comparison(
    fidelity_df: pd.DataFrame,
    baseline_method: str = "random",
    metric_col: str = "semantic_fidelity"
) -> pd.DataFrame:
    """
    Conducts rigorous per-compression-ratio paired comparisons against baseline.
    Avoids pseudo-replication by treating each of the 100 prompts as a matched paired unit.
    Applies Holm-Bonferroni correction across the family of planned comparisons.
    """
    methods = [m for m in fidelity_df["method"].unique() if m != baseline_method]
    ratios = sorted([r for r in fidelity_df["target_removal"].unique() if r > 0.0])

    records = []
    raw_p_values = []

    for m in methods:
        for r in ratios:
            m_sub = fidelity_df[(fidelity_df["method"] == m) & (fidelity_df["target_removal"] == r)]
            b_sub = fidelity_df[(fidelity_df["method"] == baseline_method) & (fidelity_df["target_removal"] == r)]

            # Merge on prompt_id to guarantee exact pairing
            merged = pd.merge(
                m_sub[["prompt_id", metric_col]],
                b_sub[["prompt_id", metric_col]],
                on="prompt_id",
                suffixes=("_method", "_baseline")
            )

            if len(merged) == 0:
                continue

            vals_m = merged[f"{metric_col}_method"].values
            vals_b = merged[f"{metric_col}_baseline"].values
            diffs = vals_m - vals_b

            test_res = paired_difference_test(vals_m.tolist(), vals_b.tolist())
            d_z = paired_cohens_d(vals_m.tolist(), vals_b.tolist())
            mean_diff, ci_l, ci_u = prompt_cluster_bootstrap_ci(diffs.tolist(), n_boot=10000)

            rec = {
                "method": m,
                "target_removal": r,
                "n_prompts": len(merged),
                "mean_delta": mean_diff,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "cohens_d_z": round(d_z, 4),
                "wilcoxon_stat": test_res["statistic"],
                "p_value_raw": test_res["p_value"],
                "win_rate": round(float(np.mean(diffs > 0.0)), 4)
            }
            records.append(rec)
            raw_p_values.append(test_res["p_value"])

    # Apply Holm-Bonferroni correction across the entire planned family
    adj_p = holm_bonferroni_correction(raw_p_values)
    for rec, p_adj in zip(records, adj_p):
        rec["p_value_holm"] = p_adj
        rec["significant_holm_005"] = bool(p_adj < 0.05)

    return pd.DataFrame(records)


def compute_correlations(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Computes Spearman and Kendall rank correlation coefficients with 95% bootstrap CIs.
    """
    if len(x) < 3 or len(y) < 3:
        return {"spearman": 0.0, "spearman_p": 1.0, "kendall": 0.0, "kendall_p": 1.0}

    spearman_res = stats.spearmanr(x, y)
    kendall_res = stats.kendalltau(x, y)

    # Bootstrap CI for Spearman r
    n = len(x)
    rng = np.random.default_rng(42)
    boot_rs = []
    arr_x = np.array(x)
    arr_y = np.array(y)
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = stats.spearmanr(arr_x[idx], arr_y[idx])
        if not np.isnan(r):
            boot_rs.append(r)

    ci_l = float(np.percentile(boot_rs, 2.5)) if boot_rs else 0.0
    ci_u = float(np.percentile(boot_rs, 97.5)) if boot_rs else 0.0

    return {
        "spearman": round(float(spearman_res.statistic if not np.isnan(spearman_res.statistic) else 0.0), 4),
        "spearman_ci_lower": round(ci_l, 4),
        "spearman_ci_upper": round(ci_u, 4),
        "spearman_p": float(spearman_res.pvalue if not np.isnan(spearman_res.pvalue) else 1.0),
        "kendall": round(float(kendall_res.statistic if not np.isnan(kendall_res.statistic) else 0.0), 4),
        "kendall_p": float(kendall_res.pvalue if not np.isnan(kendall_res.pvalue) else 1.0),
    }
