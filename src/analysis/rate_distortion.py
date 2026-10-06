"""Rate-distortion aggregation and method comparison analysis."""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from src.analysis.statistics import bootstrap_ci, cohens_d, paired_difference_test


def compute_rate_distortion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary statistics (mean, std, 95% bootstrap CI) across
    method, category, and compression ratio.
    """
    records = []
    metric_cols = [
        "semantic_fidelity", "intent", "constraints", "tone",
        "context", "dependencies", "embedding_similarity", "llm_judge"
    ]

    grouped = df.groupby(["method", "category", "target_removal"])

    for (method, category, cr), group in grouped:
        rec = {
            "method": method,
            "category": category,
            "target_removal": cr,
            "n_prompts": len(group),
            "actual_removal_mean": round(group["actual_removal"].mean(), 4),
            "retention_mean": round(group["retention"].mean(), 4)
        }

        for m in metric_cols:
            vals = group[m].dropna().tolist()
            mean_v, ci_l, ci_u = bootstrap_ci(vals, n_boot=2000)
            rec[f"{m}_mean"] = mean_v
            rec[f"{m}_std"] = round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0
            rec[f"{m}_ci_lower"] = ci_l
            rec[f"{m}_ci_upper"] = ci_u

        records.append(rec)

    return pd.DataFrame(records)


def compute_method_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates paired differences, Cohen's d, and Wilcoxon significance
    between importance methods and baselines (Random, LLMLingua, LLMLingua-2)
    at equivalent compression ratios.
    """
    records = []
    methods = [m for m in df["method"].unique() if m != "random"]
    crs = sorted(df["target_removal"].unique())

    for cr in crs:
        sub = df[df["target_removal"] == cr]
        random_sub = sub[sub["method"] == "random"].set_index("prompt_id")["semantic_fidelity"]
        llmlingua_sub = sub[sub["method"] == "llmlingua"].set_index("prompt_id")["semantic_fidelity"] if "llmlingua" in sub["method"].values else None
        llmlingua2_sub = sub[sub["method"] == "llmlingua2"].set_index("prompt_id")["semantic_fidelity"] if "llmlingua2" in sub["method"].values else None

        for method in methods:
            m_sub = sub[sub["method"] == method].set_index("prompt_id")["semantic_fidelity"]
            common_prompts = list(set(m_sub.index) & set(random_sub.index))
            if not common_prompts:
                continue

            vals_m = [m_sub[p] for p in common_prompts]
            vals_rand = [random_sub[p] for p in common_prompts]

            # Delta vs Random
            delta_rand = float(np.mean(vals_m) - np.mean(vals_rand))
            d_rand = cohens_d(vals_m, vals_rand)
            test_rand = paired_difference_test(vals_m, vals_rand)

            rec = {
                "target_removal": cr,
                "method": method,
                "mean_fidelity": round(float(np.mean(vals_m)), 4),
                "delta_vs_random": round(delta_rand, 4),
                "cohens_d_vs_random": round(d_rand, 4),
                "p_value_vs_random": test_rand["p_value"],
                "sig_vs_random": test_rand["significant_005"],
            }

            # Delta vs LLMLingua
            if llmlingua_sub is not None and method != "llmlingua":
                common_lingua = list(set(m_sub.index) & set(llmlingua_sub.index))
                if common_lingua:
                    vm = [m_sub[p] for p in common_lingua]
                    vl = [llmlingua_sub[p] for p in common_lingua]
                    rec["delta_vs_llmlingua"] = round(float(np.mean(vm) - np.mean(vl)), 4)
                    rec["p_value_vs_llmlingua"] = paired_difference_test(vm, vl)["p_value"]

            # Delta vs LLMLingua-2
            if llmlingua2_sub is not None and method != "llmlingua2":
                common_lingua2 = list(set(m_sub.index) & set(llmlingua2_sub.index))
                if common_lingua2:
                    vm = [m_sub[p] for p in common_lingua2]
                    vl2 = [llmlingua2_sub[p] for p in common_lingua2]
                    rec["delta_vs_llmlingua2"] = round(float(np.mean(vm) - np.mean(vl2)), 4)
                    rec["p_value_vs_llmlingua2"] = paired_difference_test(vm, vl2)["p_value"]

            records.append(rec)

    return pd.DataFrame(records)
