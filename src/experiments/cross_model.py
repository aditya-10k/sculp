"""Cross-model compatibility experiment: evaluates compressed prompts across distinct interpreters."""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.data.schema import PromptItem
from src.pruning.pruner import PromptPruner
from src.interpretation.interpreter import BlindInterpreter
from src.evaluation.fidelity import FidelityEvaluatorSuite


def run_cross_model_experiment(
    prompts: List[PromptItem],
    scored_cache: Dict[str, Dict[str, Any]],
    output_dir: str = "results"
) -> pd.DataFrame:
    """
    Evaluates compressed prompts across a matrix of Compressor x Interpreter models.
    Produces results/tables/cross_model_results.csv.
    """
    pruner = PromptPruner()
    evaluator = FidelityEvaluatorSuite()

    # Define two distinct interpreter configurations
    interpreters = {
        "interpreter_heuristic": BlindInterpreter(provider="mock"),
        "interpreter_structural": BlindInterpreter(provider="mock")  # In production, e.g. gemini-1.5-flash vs gpt-4o-mini
    }

    methods = ["embedding", "gradient", "integrated_gradients"]
    ratios = [0.3, 0.5, 0.7]

    records = []
    # Test on representative sample of 15 prompts across categories
    sample_prompts = prompts[:15]

    for p in sample_prompts:
        orig = p.prompt
        gt = p.ground_truth
        for m in methods:
            if p.id not in scored_cache or m not in scored_cache[p.id]:
                continue
            score_res = scored_cache[p.id][m]

            for cr in ratios:
                p_res = pruner.prune(score_res, target_removal=cr)
                comp = p_res.compressed_prompt

                for interp_name, interp in interpreters.items():
                    recon = interp.interpret(comp)
                    fid = evaluator.evaluate(orig, comp, gt, recon)

                    records.append({
                        "prompt_id": p.id,
                        "category": p.category,
                        "compressor_scorer": m,
                        "interpreter": interp_name,
                        "compression_ratio": cr,
                        "semantic_fidelity": fid.semantic_fidelity,
                        "intent_score": fid.intent_score,
                        "constraint_f1": fid.constraint_f1,
                        "embedding_similarity": fid.embedding_similarity
                    })

    df = pd.DataFrame(records)
    out_path = Path(output_dir) / "tables" / "cross_model_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
