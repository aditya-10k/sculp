"""Statistical testing, effect size estimation, and bootstrap confidence intervals."""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import scipy.stats as stats


def bootstrap_ci(
    data: List[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes empirical bootstrap confidence interval for the mean.
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


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size between two groups.
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


def paired_difference_test(x: List[float], y: List[float]) -> Dict[str, Any]:
    """
    Performs Wilcoxon signed-rank test (or paired t-test if diffs are degenerate/tied)
    on paired observations.
    """
    arr_x = np.array(x, dtype=np.float64)
    arr_y = np.array(y, dtype=np.float64)
    diff = arr_x - arr_y
    n = len(diff)

    mean_diff = float(np.mean(diff))
    # If all differences are zero
    if np.allclose(diff, 0.0):
        return {
            "test": "wilcoxon",
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_difference": 0.0,
            "significant_005": False
        }

    try:
        # Wilcoxon signed-rank test
        res = stats.wilcoxon(arr_x, arr_y, alternative='two-sided', zero_method='pratt')
        stat = float(res.statistic)
        p_val = float(res.pvalue)
        test_name = "wilcoxon"
    except Exception:
        # Fallback to paired t-test
        res = stats.ttest_rel(arr_x, arr_y)
        stat = float(res.statistic)
        p_val = float(res.pvalue)
        test_name = "paired_ttest"

    return {
        "test": test_name,
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "mean_difference": round(mean_diff, 4),
        "significant_005": bool(p_val < 0.05)
    }


def compute_correlations(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Computes Spearman and Kendall rank correlation coefficients.
    """
    if len(x) < 3 or len(y) < 3:
        return {"spearman": 0.0, "spearman_p": 1.0, "kendall": 0.0, "kendall_p": 1.0}

    spearman_res = stats.spearmanr(x, y)
    kendall_res = stats.kendalltau(x, y)

    return {
        "spearman": round(float(spearman_res.statistic if not np.isnan(spearman_res.statistic) else 0.0), 4),
        "spearman_p": round(float(spearman_res.pvalue if not np.isnan(spearman_res.pvalue) else 1.0), 6),
        "kendall": round(float(kendall_res.statistic if not np.isnan(kendall_res.statistic) else 0.0), 4),
        "kendall_p": round(float(kendall_res.pvalue if not np.isnan(kendall_res.pvalue) else 1.0), 6),
    }
