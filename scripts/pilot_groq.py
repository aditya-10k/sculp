"""Pilot execution script testing Groq API integration across 10 representative prompts.
Validates Protocol A (Blind Interpretability) and Protocol B (Reference-Guided Preservation),
measures request latencies, failure rates, and checks inter-model agreement.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv
from src.data.dataset import DatasetManager
from src.pruning.pruner import PromptPruner
from src.scoring.embedding import EmbeddingLeaveOneOutScorer
from src.scoring.gradient import GradientScorer
from src.scoring.random import RandomScorer
from src.interpretation.groq_client import GroqEvaluatorClient, InterpreterAPIError
from src.interpretation.evaluator_protocols import (
    ProtocolAEvaluator,
    ProtocolBEvaluator,
    EvaluatorAgreementSuite
)

load_dotenv()


def run_pilot():
    print("=" * 80)
    print("SCULP PILOT: GROQ API EVALUATION (10 PROMPTS)")
    print("=" * 80)

    # 1. Initialize Dataset & Scorers
    dataset_path = "data/processed/prompts.json"
    mgr = DatasetManager(dataset_path)
    all_prompts = mgr.all_prompts()

    # Pick 2 prompts from each of the 5 categories (Total = 10 prompts)
    categories = ["instruction", "constraint", "context", "tone", "multi_step"]
    pilot_prompts = []
    for cat in categories:
        cat_p = [p for p in all_prompts if p.category == cat]
        pilot_prompts.extend(cat_p[:2])

    print(f"[Pilot] Selected {len(pilot_prompts)} prompts across 5 categories:")
    for p in pilot_prompts:
        print(f"  - [{p.id}] ({p.category}): \"{p.prompt[:60]}...\"")

    # 2. Initialize Groq Client
    api_key = os.getenv("GROQ_API_KEY")
    primary_model = os.getenv("INTERPRETER_MODEL", "openai/gpt-oss-20b")
    secondary_model = os.getenv("RELIABILITY_MODEL", "qwen/qwen3.8-27b")

    print(f"\n[Pilot] Primary Evaluator Model:   {primary_model}")
    print(f"[Pilot] Secondary Agreement Model: {secondary_model}")
    print(f"[Pilot] API Key Present:           {bool(api_key)}")

    client = GroqEvaluatorClient(
        api_key=api_key,
        model_name=primary_model,
        cache_dir="results/interpreter_cache"
    )
    proto_a = ProtocolAEvaluator(client)
    proto_b = ProtocolBEvaluator(client)
    agreement_suite = EvaluatorAgreementSuite(
        client,
        primary_model=primary_model,
        secondary_model=secondary_model
    )

    # 3. Setup Pruning
    pruner = PromptPruner(tokenizer_model="bert-base-uncased")
    scorers = {
        "embedding": EmbeddingLeaveOneOutScorer(seed=42),
        "gradient": GradientScorer(seed=42),
        "random": RandomScorer(seed=42)
    }

    test_ratios = [0.3, 0.5, 0.7]

    records = []
    agreement_records = []
    t_start = time.time()
    call_count = 0
    failures = 0

    print(f"\n[Pilot] Running progressive pruning, Protocol A, and Protocol B...")

    for p_item in pilot_prompts:
        pid = p_item.id
        orig_text = p_item.prompt
        gt = p_item.ground_truth

        # Score prompt with embedding scorer
        score_res = scorers["embedding"].score_prompt(orig_text, prompt_id=pid)

        for cr in test_ratios:
            prune_res = pruner.prune(score_res, target_removal=cr)
            comp_text = prune_res.compressed_prompt

            print(f"\n  Evaluating {pid} (Target CR: {cr*100:.0f}%, Actual CR: {prune_res.actual_removal*100:.1f}%):")
            print(f"    Compressed: \"{comp_text}\"")

            # --- Protocol A: Blind Interpretability (Input isolated: ONLY compressed text) ---
            try:
                t0 = time.time()
                res_a = proto_a.evaluate_standalone(comp_text)
                lat_a = round((time.time() - t0) * 1000, 1)
                call_count += 1
                print(f"    [Protocol A Blind] Interpretability={res_a.standalone_interpretability:.2f}, Clarity={res_a.clarity:.2f}, Actionability={res_a.actionability:.2f} ({lat_a}ms)")
            except Exception as e:
                print(f"    [Protocol A ERROR] {e}")
                failures += 1
                res_a = None

            # --- Protocol B: Reference-Guided Semantic Preservation ---
            try:
                t0 = time.time()
                res_b = proto_b.evaluate_preservation(comp_text, gt)
                lat_b = round((time.time() - t0) * 1000, 1)
                call_count += 1
                print(f"    [Protocol B Guided] Intent={res_b.intent_preservation:.2f}, Constraints={res_b.constraint_preservation:.2f}, Context={res_b.context_preservation:.2f}, Comp={res_b.composite_preservation:.2f} ({lat_b}ms)")
            except Exception as e:
                print(f"    [Protocol B ERROR] {e}")
                failures += 1
                res_b = None

            # Record sample
            records.append({
                "prompt_id": pid,
                "category": p_item.category,
                "target_removal": cr,
                "actual_removal": prune_res.actual_removal,
                "compressed_prompt": comp_text,
                "protocol_a_interpretability": res_a.standalone_interpretability if res_a else None,
                "protocol_a_clarity": res_a.clarity if res_a else None,
                "protocol_a_actionability": res_a.actionability if res_a else None,
                "protocol_b_intent": res_b.intent_preservation if res_b else None,
                "protocol_b_constraints": res_b.constraint_preservation if res_b else None,
                "protocol_b_context": res_b.context_preservation if res_b else None,
                "protocol_b_tone": res_b.tone_preservation if res_b else None,
                "protocol_b_dependencies": res_b.dependency_preservation if res_b else None,
                "protocol_b_composite": res_b.composite_preservation if res_b else None,
            })

            # Check evaluator agreement on a subset (at CR=0.5)
            if cr == 0.5:
                try:
                    agree_res = agreement_suite.evaluate_agreement_sample(comp_text, gt)
                    agree_res["prompt_id"] = pid
                    agree_res["category"] = p_item.category
                    agreement_records.append(agree_res)
                    print(f"    [Agreement @ 50%] M1 ProtoA={agree_res['primary_protocol_a']:.2f} vs M2 ProtoA={agree_res['secondary_protocol_a']:.2f} (Delta={agree_res['delta_protocol_a']:.2f})")
                except Exception as e:
                    print(f"    [Agreement ERROR] {e}")

    total_time = time.time() - t_start
    pilot_df = pd.DataFrame(records)
    agree_df = pd.DataFrame(agreement_records)

    # Save pilot outputs
    out_dir = Path("results/pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_df.to_csv(out_dir / "pilot_evaluation_results.csv", index=False)
    agree_df.to_csv(out_dir / "pilot_evaluator_agreement.csv", index=False)

    print("\n" + "=" * 80)
    print("PILOT EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Prompts Evaluated:     {len(pilot_prompts)}")
    print(f"Total Evaluations Conducted: {len(records)}")
    print(f"Total API / Cached Calls:    {call_count}")
    print(f"Failure Count:               {failures} ({failures/max(1, call_count)*100:.1f}%)")
    print(f"Total Elapsed Time:          {total_time:.2f} seconds ({total_time/max(1, call_count):.2f}s per call)")
    print(f"\nProtocol A (Standalone Interpretability) Mean: {pilot_df['protocol_a_interpretability'].mean():.4f}")
    print(f"Protocol B (Reference-Guided Preservation) Mean: {pilot_df['protocol_b_composite'].mean():.4f}")
    if not agree_df.empty:
        print(f"Inter-Model Mean Absolute Difference (Protocol A): {agree_df['delta_protocol_a'].mean():.4f}")
        print(f"Inter-Model Mean Absolute Difference (Protocol B): {agree_df['delta_protocol_b'].mean():.4f}")
    print(f"\nPilot results saved to: {out_dir / 'pilot_evaluation_results.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    run_pilot()
