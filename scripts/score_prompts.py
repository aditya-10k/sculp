"""Script to score benchmark prompts across all attribution methods."""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm
from src.data.dataset import DatasetManager
from src.experiments.experiment_config import ExperimentConfig
from src.scoring.random import RandomScorer
from src.scoring.embedding import EmbeddingLeaveOneOutScorer
from src.scoring.gradient import GradientScorer
from src.scoring.integrated_gradients import IntegratedGradientScorer


def main():
    parser = argparse.ArgumentParser(description="Score prompts using all attribution methods.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts")
    args = parser.parse_args()

    cfg = ExperimentConfig.load_from_yaml(args.config)
    dataset = DatasetManager(cfg.prompts_path)
    prompts = dataset.all_prompts()
    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Scoring {len(prompts)} prompts...")
    scorers = {
        "random": RandomScorer(seed=cfg.seed),
        "embedding": EmbeddingLeaveOneOutScorer(model_name=cfg.embedding_model, seed=cfg.seed),
        "gradient": GradientScorer(model_name=cfg.attribution_model, seed=cfg.seed),
        "integrated_gradients": IntegratedGradientScorer(model_name=cfg.attribution_model, seed=cfg.seed)
    }

    scores_rows = []
    for p in tqdm(prompts, desc="Scoring"):
        for m_name, scorer in scorers.items():
            res = scorer.score_prompt(p.prompt, prompt_id=p.id)
            for ws in res.word_scores:
                scores_rows.append({
                    "prompt_id": p.id,
                    "category": p.category,
                    "method": m_name,
                    "word_id": ws.word_id,
                    "word": ws.word,
                    "raw_score": ws.raw_score,
                    "normalized_score": ws.normalized_score,
                    "rank": ws.rank
                })

    df = pd.DataFrame(scores_rows)
    out_file = Path(cfg.results_dir) / "scores" / "token_importance.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} scored token records to {out_file}")


if __name__ == "__main__":
    main()
