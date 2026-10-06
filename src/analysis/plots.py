"""Plotting and publication-ready visualization generation using Matplotlib and Seaborn."""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Publication styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

METHOD_COLORS = {
    "random": "#95a5a6",
    "embedding": "#2980b9",
    "gradient": "#e67e22",
    "integrated_gradients": "#27ae60",
    "llmlingua": "#8e44ad",
    "llmlingua2": "#d35400"
}


def plot_overall_rate_distortion(df: pd.DataFrame, out_path: str = "results/plots/overall_rate_distortion.png"):
    """Plot semantic fidelity vs compression ratio across all methods."""
    plt.figure(figsize=(9, 6))
    methods = df["method"].unique()

    for m in methods:
        sub = df[df["method"] == m]
        grouped = sub.groupby("target_removal")["semantic_fidelity"].agg(["mean", "std"]).reset_index()
        color = METHOD_COLORS.get(m, "#34495e")
        label = m.replace("_", " ").title()
        plt.plot(grouped["target_removal"], grouped["mean"], marker="o", linewidth=2.2, label=label, color=color)
        plt.fill_between(
            grouped["target_removal"],
            (grouped["mean"] - grouped["std"]).clip(lower=0.0),
            (grouped["mean"] + grouped["std"]).clip(upper=1.0),
            color=color, alpha=0.15
        )

    plt.title("Semantic Rate-Distortion: Fidelity vs Compression Ratio", weight="bold")
    plt.xlabel("Target Lexical Removal Ratio (Compression Rate)")
    plt.ylabel("Semantic Fidelity Score (SF in [0, 1])")
    plt.ylim(-0.02, 1.05)
    plt.xlim(-0.02, 0.85)
    plt.legend(title="Method", loc="lower left", frameon=True)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_category_rate_distortion(df: pd.DataFrame, out_path: str = "results/plots/category_rate_distortion.png"):
    """Plot rate-distortion curves faceted by prompt category."""
    categories = sorted(df["category"].unique())
    n_cats = max(1, len(categories))
    fig, axes = plt.subplots(1, n_cats, figsize=(4 * n_cats, 4.5), sharey=True)
    if n_cats == 1:
        axes = [axes]

    for idx, (cat, ax) in enumerate(zip(categories, axes)):
        cat_df = df[df["category"] == cat]
        for m in cat_df["method"].unique():
            sub = cat_df[cat_df["method"] == m]
            grouped = sub.groupby("target_removal")["semantic_fidelity"].mean().reset_index()
            color = METHOD_COLORS.get(m, "#34495e")
            ax.plot(grouped["target_removal"], grouped["semantic_fidelity"], marker="s", label=m, color=color)

        ax.set_title(cat.replace("_", " ").title(), weight="bold")
        ax.set_xlabel("Compression Ratio")
        if idx == 0:
            ax.set_ylabel("Semantic Fidelity")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-0.02, 0.85)

    axes[-1].legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.suptitle("Semantic Collapse by Prompt Component Category", weight="bold", y=1.02)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_method_comparison(df: pd.DataFrame, out_path: str = "results/plots/method_comparison.png"):
    """Bar chart comparing methods at 30%, 50%, and 70% compression."""
    selected_crs = [0.3, 0.5, 0.7]
    sub = df[df["target_removal"].isin(selected_crs)]

    plt.figure(figsize=(10, 5.5))
    sns.barplot(
        data=sub,
        x="target_removal",
        y="semantic_fidelity",
        hue="method",
        palette=METHOD_COLORS,
        errorbar="ci"
    )

    plt.title("Method Comparison at Critical Compression Regimes (30%, 50%, 70%)", weight="bold")
    plt.xlabel("Target Removal Ratio")
    plt.ylabel("Mean Semantic Fidelity")
    plt.ylim(0.0, 1.05)
    plt.legend(title="Method", loc="lower left")
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_importance_distribution(word_scores_df: pd.DataFrame, out_path: str = "results/plots/importance_distribution.png"):
    """Distribution of normalized importance scores across methods."""
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=word_scores_df,
        x="method",
        y="normalized_score",
        palette=METHOD_COLORS
    )

    plt.title("Distribution of Prompt-Normalized Importance Scores", weight="bold")
    plt.xlabel("Scoring Method")
    plt.ylabel("Normalized Importance Score")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_collapse_thresholds(collapse_df: pd.DataFrame, out_path: str = "results/plots/collapse_thresholds.png"):
    """Horizontal bar chart showing collapse ratio breakpoints."""
    plt.figure(figsize=(10, 6))
    sub = collapse_df[collapse_df["category"] != "all"]

    sns.barplot(
        data=sub,
        y="category",
        x="collapse_ratio",
        hue="method",
        palette=METHOD_COLORS
    )

    plt.title("Empirical Semantic Collapse Thresholds by Category", weight="bold")
    plt.xlabel("Breakpoint Removal Ratio (Where Largest Semantic Drop Occurs)")
    plt.ylabel("Prompt Category")
    plt.xlim(0.0, 0.85)
    plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_component_survival(df: pd.DataFrame, out_path: str = "results/plots/component_survival.png"):
    """Survival curves of individual semantic components vs compression ratio."""
    plt.figure(figsize=(9, 6))
    components = ["intent", "constraints", "tone", "context", "dependencies"]
    comp_colors = {
        "intent": "#2c3e50",
        "constraints": "#e74c3c",
        "tone": "#f39c12",
        "context": "#16a085",
        "dependencies": "#8e44ad"
    }

    for comp in components:
        grouped = df.groupby("target_removal")[comp].mean().reset_index()
        plt.plot(
            grouped["target_removal"],
            grouped[comp],
            marker="o",
            linewidth=2.2,
            label=comp.title(),
            color=comp_colors.get(comp, "#333333")
        )

    plt.title("Component-Wise Semantic Survival Under Lexical Pruning", weight="bold")
    plt.xlabel("Compression Ratio (Lexical Removal)")
    plt.ylabel("Component Fidelity Score in [0, 1]")
    plt.ylim(-0.02, 1.05)
    plt.xlim(-0.02, 0.85)
    plt.legend(title="Semantic Component", loc="lower left")
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_token_heatmap(
    words: List[str],
    scores: List[float],
    title: str = "Token Importance Heatmap",
    out_path: str = "results/plots/token_heatmap_smoke.png"
):
    """Visual heatmap of word importance scores for a sample prompt."""
    plt.figure(figsize=(max(8, len(words) * 1.1), 3))
    data = np.array([scores])

    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar=True,
        xticklabels=words,
        yticklabels=["Score"],
        vmin=0.0,
        vmax=1.0,
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title(title, weight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
