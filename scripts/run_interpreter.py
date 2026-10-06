"""Script to run blind interpretation on compressed prompts."""

import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm
from src.experiments.experiment_config import ExperimentConfig
from src.interpretation.interpreter import BlindInterpreter


def main():
    parser = argparse.ArgumentParser(description="Run blind interpreter on compressed prompts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default="results/compressed/pruned_prompts.csv")
    args = parser.parse_args()

    cfg = ExperimentConfig.load_from_yaml(args.config)
    inp_path = Path(args.input)
    if not inp_path.exists():
        print(f"Input file {inp_path} not found. Run scripts/run_pruning.py first.")
        return

    df = pd.read_csv(inp_path)
    interpreter = BlindInterpreter(provider=cfg.interpreter_provider, model_name=cfg.interpreter_model)

    results = []
    print(f"Interpreting {len(df)} compressed prompt instances...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Interpreting"):
        prompt_str = str(row["compressed_prompt"])
        recon = interpreter.interpret(prompt_str)
        results.append({
            "prompt_id": row["prompt_id"],
            "method": row["method"],
            "target_removal": row["target_removal"],
            "compressed_prompt": prompt_str,
            "reconstruction": recon.model_dump()
        })

    out_file = Path(cfg.results_dir) / "interpreter" / "reconstructions.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} interpretations to {out_file}")


if __name__ == "__main__":
    main()
