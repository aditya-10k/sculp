"""Script to run progressive lexical pruning across all methods and ratios."""

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
from src.pruning.pruner import PromptPruner
from src.baselines.llmlingua import LLMLinguaCompressor
from src.baselines.llmlingua2 import LLMLingua2Compressor


def main():
    parser = argparse.ArgumentParser(description="Prune prompts across progressive ratios.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig.load_from_yaml(args.config)
    dataset = DatasetManager(cfg.prompts_path)
    prompts = dataset.all_prompts()
    if args.limit:
        prompts = prompts[:args.limit]

    pruner = PromptPruner(tokenizer_model=cfg.attribution_model)
    scorers = {
        "random": RandomScorer(seed=cfg.seed),
        "embedding": EmbeddingLeaveOneOutScorer(model_name=cfg.embedding_model, seed=cfg.seed)
    }
    baselines = {
        "llmlingua": LLMLinguaCompressor(fallback_tokenizer=cfg.attribution_model),
        "llmlingua2": LLMLingua2Compressor(fallback_tokenizer=cfg.attribution_model)
    }

    compressed_rows = []
    print(f"Pruning {len(prompts)} prompts across ratios {cfg.compression_ratios}...")

    for p in tqdm(prompts, desc="Pruning"):
        for m_name, scorer in scorers.items():
            score_res = scorer.score_prompt(p.prompt, prompt_id=p.id)
            for cr in cfg.compression_ratios:
                res = pruner.prune(score_res, target_removal=cr)
                compressed_rows.append(res.to_dict())

        for b_name, baseline in baselines.items():
            for cr in cfg.compression_ratios:
                res = baseline.compress(p.prompt, prompt_id=p.id, target_removal=cr)
                compressed_rows.append(res.to_dict())

    df = pd.DataFrame(compressed_rows)
    out_file = Path(cfg.results_dir) / "compressed" / "pruned_prompts.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} compressed prompt instances to {out_file}")


if __name__ == "__main__":
    main()
