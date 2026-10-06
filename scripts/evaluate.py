"""Script to evaluate ground-truth semantic fidelity on interpretations."""

import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm
from src.data.dataset import DatasetManager
from src.experiments.experiment_config import ExperimentConfig
from src.interpretation.prompts import ReconstructedSemantics
from src.evaluation.fidelity import FidelityEvaluatorSuite


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic fidelity of interpretations.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default="results/interpreter/reconstructions.json")
    args = parser.parse_args()

    cfg = ExperimentConfig.load_from_yaml(args.config)
    dataset = DatasetManager(cfg.prompts_path)

    inp_path = Path(args.input)
    if not inp_path.exists():
        print(f"File {inp_path} not found. Run scripts/run_interpreter.py first.")
        return

    with open(inp_path, "r", encoding="utf-8") as f:
        reconstructions = json.load(f)

    evaluator = FidelityEvaluatorSuite(embedding_model_name=cfg.embedding_model, judge_provider=cfg.interpreter_provider)

    records = []
    print(f"Evaluating fidelity for {len(reconstructions)} instances...")
    for item in tqdm(reconstructions, desc="Evaluating"):
        pid = item["prompt_id"]
        p_item = dataset.get_by_id(pid)
        if not p_item:
            continue

        comp_prompt = item["compressed_prompt"]
        recon = ReconstructedSemantics(**item["reconstruction"])

        fid = evaluator.evaluate(p_item.prompt, comp_prompt, p_item.ground_truth, recon)

        rec = {
            "prompt_id": pid,
            "category": p_item.category,
            "method": item["method"],
            "target_removal": item["target_removal"],
            "compressed_prompt": comp_prompt
        }
        rec.update(fid.to_dict())
        records.append(rec)

    df = pd.DataFrame(records)
    out_file = Path(cfg.results_dir) / "metrics" / "fidelity_results.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved fidelity evaluations to {out_file}")


if __name__ == "__main__":
    main()
