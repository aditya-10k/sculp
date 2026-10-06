"""Script to generate all publication-ready graphs from experiment results."""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.analysis.plots import (
    plot_overall_rate_distortion,
    plot_category_rate_distortion,
    plot_method_comparison,
    plot_importance_distribution,
    plot_collapse_thresholds,
    plot_component_survival
)


def main():
    parser = argparse.ArgumentParser(description="Generate all research plots.")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    fidelity_path = res_dir / "metrics" / "fidelity_results.csv"
    scores_path = res_dir / "scores" / "token_importance.csv"
    collapse_path = res_dir / "tables" / "collapse_thresholds.csv"
    plots_dir = res_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not fidelity_path.exists():
        print(f"Error: {fidelity_path} not found. Run experiment runner first.")
        return

    print("Loading data for plotting...")
    fidelity_df = pd.read_csv(fidelity_path)

    print("Generating overall rate-distortion curve...")
    plot_overall_rate_distortion(fidelity_df, str(plots_dir / "overall_rate_distortion.png"))

    print("Generating category-wise rate-distortion curves...")
    plot_category_rate_distortion(fidelity_df, str(plots_dir / "category_rate_distortion.png"))

    print("Generating method comparison bar chart...")
    plot_method_comparison(fidelity_df, str(plots_dir / "method_comparison.png"))

    print("Generating component survival curves...")
    plot_component_survival(fidelity_df, str(plots_dir / "component_survival.png"))

    if scores_path.exists():
        print("Generating importance score distribution plot...")
        scores_df = pd.read_csv(scores_path)
        plot_importance_distribution(scores_df, str(plots_dir / "importance_distribution.png"))

    if collapse_path.exists():
        print("Generating collapse threshold chart...")
        collapse_df = pd.read_csv(collapse_path)
        plot_collapse_thresholds(collapse_df, str(plots_dir / "collapse_thresholds.png"))

    print(f"All plots successfully generated and saved to {plots_dir}")


if __name__ == "__main__":
    main()
