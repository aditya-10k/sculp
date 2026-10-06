"""Scientific and Implementation Audit Script for SCULP.
Extracts empirical evidence, runs verification checks, and gathers numerical data for all 20 audit points.
"""

import sys
import json
import math
from pathlib import Path
from collections import Counter
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DatasetManager
from src.scoring.random import RandomScorer
from src.scoring.embedding import EmbeddingLeaveOneOutScorer
from src.scoring.gradient import GradientScorer
from src.scoring.integrated_gradients import IntegratedGradientScorer
from src.pruning.pruner import PromptPruner
from src.baselines.llmlingua import LLMLinguaCompressor
from src.baselines.llmlingua2 import LLMLingua2Compressor
from src.interpretation.interpreter import BlindInterpreter
from src.evaluation.fidelity import FidelityEvaluatorSuite
from src.analysis.statistics import bootstrap_ci, paired_difference_test, cohens_d, compute_correlations


def audit_dataset():
    print("=" * 60)
    print("AUDIT ITEMS 2 & 3: DATASET COUNTS & REPRESENTATIVE PROMPTS")
    print("=" * 60)
    mgr = DatasetManager("data/processed/prompts.json")
    all_p = mgr.all_prompts()
    print(f"Total prompts loaded: {len(all_p)}")
    counts = Counter(p.category for p in all_p)
    for cat, c in sorted(counts.items()):
        print(f"  Category '{cat}': {c} prompts")

    samples_by_cat = {}
    for cat in ["instruction", "constraint", "tone", "context", "multi_step"]:
        cat_prompts = mgr.get_by_category(cat)
        samples_by_cat[cat] = cat_prompts[:5]
        print(f"\n--- Category: {cat} (Showing {len(samples_by_cat[cat])} prompts) ---")
        for i, p in enumerate(samples_by_cat[cat], 1):
            print(f"  [{p.id}] Prompt: \"{p.prompt}\"")
            print(f"       Ground Truth Intent: {p.ground_truth.intent}")
            print(f"       Constraints: {p.ground_truth.constraints}")
            print(f"       Tone: {p.ground_truth.tone}")
            print(f"       Context: {p.ground_truth.context}")
            print(f"       Steps: {p.ground_truth.steps}")
    return counts, samples_by_cat


def audit_gradient_scorer():
    print("\n" + "=" * 60)
    print("AUDIT ITEM 4: INSPECT GRADIENT SCORER")
    print("=" * 60)
    sentence = "That cat chased the mouse across the garden."
    scorer = GradientScorer(model_name="bert-base-uncased", seed=42)
    res = scorer.score_prompt(sentence, "smoke_gradient")

    raw_scores = [w.raw_score for w in res.word_scores]
    words = [w.word for w in res.word_scores]

    print(f"Test Sentence: \"{sentence}\"")
    print(f"Number of words: {len(words)}")
    print("\nRaw attribution values with 12 significant digits (scientific):")
    for w, r in zip(words, raw_scores):
        print(f"  {w:<10}: {r:.12e}")

    print(f"\nSummary Statistics for Raw Gradients:")
    print(f"  Min gradient:    {min(raw_scores):.12e}")
    print(f"  Max gradient:    {max(raw_scores):.12e}")
    print(f"  Mean gradient:   {np.mean(raw_scores):.12e}")
    print(f"  Non-zero count:  {sum(1 for r in raw_scores if abs(r) > 1e-12)} / {len(raw_scores)}")

    print("\nRanking by Gradient Attribution:")
    for ws in sorted(res.word_scores, key=lambda x: x.raw_score, reverse=True):
        print(f"  Rank {ws.rank}: {ws.word:<10} Raw: {ws.raw_score:.6f} Norm: {ws.normalized_score:.4f}")

    return res


def audit_integrated_gradients():
    print("\n" + "=" * 60)
    print("AUDIT ITEM 5: INSPECT INTEGRATED GRADIENTS (PUNCTUATION SINK TEST)")
    print("=" * 60)
    sentence_with_punc = "That cat chased the mouse across the garden."
    sentence_no_punc = "That cat chased the mouse across the garden"

    ig_scorer = IntegratedGradientScorer(model_name="bert-base-uncased", n_steps=25, seed=42)

    res_with = ig_scorer.score_prompt(sentence_with_punc, "smoke_ig_with")
    res_no = ig_scorer.score_prompt(sentence_no_punc, "smoke_ig_no")

    print("\n--- Integrated Gradients WITH Punctuation ---")
    for ws in sorted(res_with.word_scores, key=lambda x: x.raw_score, reverse=True):
        print(f"  Rank {ws.rank}: {ws.word:<10} Raw: {ws.raw_score:.6f} Norm: {ws.normalized_score:.4f}")

    print("\n--- Integrated Gradients WITHOUT Punctuation ---")
    for ws in sorted(res_no.word_scores, key=lambda x: x.raw_score, reverse=True):
        print(f"  Rank {ws.rank}: {ws.word:<10} Raw: {ws.raw_score:.6f} Norm: {ws.normalized_score:.4f}")

    return res_with, res_no


def audit_subword_alignment():
    print("\n" + "=" * 60)
    print("AUDIT ITEM 6: TOKEN / SUBWORD ALIGNMENT VERIFICATION")
    print("=" * 60)
    sentence = "Explain unsupervised machine learning algorithms."
    pruner = PromptPruner(tokenizer_model="bert-base-uncased")
    tokens, words, subword_map = pruner.tokenizer_wrapper.tokenize_and_align(sentence)

    print(f"Sentence: \"{sentence}\"")
    print(f"Total Bert tokens (including specials): {len(tokens)}")
    for t in tokens:
        print(f"  Token Index {t.token_index}: '{t.token}' (ID={t.token_id}, Span: {t.start_char}-{t.end_char}, Special={t.is_special})")

    print("\nWord Spans & Subwords mapped:")
    for w in words:
        sub_texts = [tokens[tid].token for tid in w.subtoken_indices]
        print(f"  Word {w.word_id}: '{w.word}' -> Subwords: {sub_texts} (Indices: {w.subtoken_indices})")

    scorer = GradientScorer(model_name="bert-base-uncased", seed=42)
    res = scorer.score_prompt(sentence, "subword_test")
    print("\nWord-level Aggregation from Subwords (sum of subword attribution L2 norms):")
    for ws in res.word_scores:
        print(f"  Word: '{ws.word:<14}' -> Aggregated Raw Score: {ws.raw_score:.6f}")


def audit_pruning_discrete_rounding():
    print("\n" + "=" * 60)
    print("AUDIT ITEM 7: PRUNING TARGET VS ACTUAL TOKEN REMOVAL")
    print("=" * 60)
    sentence = "That cat chased the mouse across the garden."
    scorer = EmbeddingLeaveOneOutScorer(seed=42)
    score_res = scorer.score_prompt(sentence, "smoke_round")
    pruner = PromptPruner()

    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"{'Target CR':<12} {'Orig Tokens':<14} {'Rem Tokens':<14} {'Actual CR':<12} {'Retention':<12} {'Compressed Prompt'}")
    print("-" * 90)
    for cr in ratios:
        p_res = pruner.prune(score_res, target_removal=cr)
        print(f"{cr:<12.2f} {p_res.original_tokens:<14} {p_res.remaining_tokens:<14} {p_res.actual_removal:<12.4f} {p_res.retention:<12.4f} \"{p_res.compressed_prompt}\"")


def audit_baselines():
    print("\n" + "=" * 60)
    print("AUDIT ITEMS 8 & 9: BASELINE COMPRESSORS (LLMLingua & LLMLingua-2)")
    print("=" * 60)
    c1 = LLMLinguaCompressor()
    c2 = LLMLingua2Compressor()

    test_prompts = [
        "Explain supervised vs unsupervised machine learning in simple terms.",
        "Summarize the attached quarterly earnings report focusing on revenue and EBITDA.",
        "Write a polite job rejection email to a qualified candidate.",
        "Respond to the customer who contacted us for the third time regarding delayed order #9482.",
        "1. Identify the system failure. 2. Explain the root cause. 3. Suggest two mitigation steps."
    ]

    print("\n>>> LLMLingua (GPT-2 Perplexity Baseline):")
    for i, p in enumerate(test_prompts, 1):
        r = c1.compress(p, f"llmlingua_p{i}", target_removal=0.5)
        print(f"  Prompt {i} ({r.original_tokens} -> {r.remaining_tokens} tokens, Actual Removal={r.actual_removal*100:.1f}%, Official={r.metadata.get('official_llmlingua')}):")
        print(f"    Orig: \"{p}\"")
        print(f"    Comp: \"{r.compressed_prompt}\"")

    print("\n>>> LLMLingua-2 (Token Classification Baseline):")
    for i, p in enumerate(test_prompts, 1):
        r = c2.compress(p, f"llmlingua2_p{i}", target_removal=0.5)
        print(f"  Prompt {i} ({r.original_tokens} -> {r.remaining_tokens} tokens, Actual Removal={r.actual_removal*100:.1f}%, Official={r.metadata.get('official_llmlingua2')}):")
        print(f"    Orig: \"{p}\"")
        print(f"    Comp: \"{r.compressed_prompt}\"")


def audit_blind_interpreter():
    print("\n" + "=" * 60)
    print("AUDIT ITEM 10: BLIND INTERPRETER ISOLATION & SAMPLE")
    print("=" * 60)
    interp = BlindInterpreter(provider="mock")
    comp_input = "Identify system failure root cause two mitigation steps"
    recon = interp.interpret(comp_input)

    print(f"Input to Interpreter (ONLY compressed text, NO metadata):")
    print(f"  \"{comp_input}\"")
    print("\nParsed Semantic Reconstruction:")
    print(f"  Task:                  {recon.task}")
    print(f"  Identified Constraints: {recon.constraints}")
    print(f"  Inferred Tone:         {recon.tone}")
    print(f"  Entities / Context:    {recon.context}")
    print(f"  Identified Steps:      {recon.ordering_dependencies}")
    print(f"  Required Actions:      {recon.required_actions}")


def audit_fidelity_metrics():
    print("\n" + "=" * 60)
    print("AUDIT ITEMS 11 & 12: SEMANTIC FIDELITY FORMULAS & LLM JUDGE")
    print("=" * 60)
    suite = FidelityEvaluatorSuite(judge_provider="mock")
    mgr = DatasetManager("data/processed/prompts.json")
    item = mgr.get_by_id("p081")  # multi-step prompt

    orig = item.prompt
    comp = "Identify failure cause two solutions safer option"
    recon = BlindInterpreter(provider="mock").interpret(comp)
    fid = suite.evaluate(orig, comp, item.ground_truth, recon)

    print(f"Prompt ID: {item.id}")
    print(f"Original:   \"{orig}\"")
    print(f"Compressed: \"{comp}\"")
    print("\nComponent Fidelity Scores:")
    print(f"  1. Intent Score (Cosine Similarity):      {fid.intent_score:.4f}")
    print(f"  2. Constraint F1 (Harmonic Mean P&R):     {fid.constraint_f1:.4f} (P={fid.constraint_precision:.4f}, R={fid.constraint_recall:.4f})")
    print(f"  3. Tone F1 (Descriptor Overlap):          {fid.tone_f1:.4f}")
    print(f"  4. Context F1 (Entity/Keyword Overlap):   {fid.context_f1:.4f}")
    print(f"  5. Dependency Preservation (Step Order):  {fid.dependency_preservation:.4f}")
    print(f"  -------------------------------------------------------------")
    print(f"  Composite Semantic Fidelity (SF):        {fid.semantic_fidelity:.4f}")
    print(f"  Sentence-BERT Embedding Cosine:          {fid.embedding_similarity:.4f}")
    print(f"  LLM Judge Score (Secondary Metric):       {fid.llm_judge_score:.4f}")


def audit_human_baseline():
    print("\n" + "=" * 60)
    print("AUDIT ITEM 13: HUMAN BASELINE PROVENANCE & ALL 20 EXAMPLES")
    print("=" * 60)
    csv_path = Path("results/tables/human_baseline_comparison.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Total human compressed prompts evaluated: {len(df)}")
        print(f"Mean Retention:         {df['retention'].mean():.4f}")
        print(f"Mean Semantic Fidelity: {df['semantic_fidelity'].mean():.4f}")
        print(f"Mean Intent Score:      {df['intent'].mean():.4f}")
        print(f"Mean Constraint F1:     {df['constraints'].mean():.4f}")
        print("\nAll 20 Examples in Human Baseline Comparison:")
        for _, r in df.iterrows():
            print(f"  [{r['prompt_id']}] ({r['category']}): Orig={r['original_tokens']} -> Comp={r['compressed_tokens']} tokens (CR={r['actual_removal']*100:.1f}%), SF={r['semantic_fidelity']:.4f}")
    else:
        print("human_baseline_comparison.csv not found!")


def audit_full_results():
    print("\n" + "=" * 60)
    print("AUDIT ITEMS 14, 15, 16, 17, 18: FULL DATASET RESULTS & STATISTICS")
    print("=" * 60)
    df = pd.read_csv("results/metrics/fidelity_results.csv")
    print(f"Total rows in fidelity_results.csv: {len(df)}")
    print(f"Unique prompts: {df['prompt_id'].nunique()}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Ratios: {sorted(df['target_removal'].unique().tolist())}")

    print("\n--- Mean Semantic Fidelity by Method across 100 Prompts & 9 Ratios ---")
    method_means = df.groupby("method")["semantic_fidelity"].agg(["mean", "std", "count"]).reset_index()
    print(method_means.to_string(index=False))

    print("\n--- Paired Wilcoxon Signed-Rank Tests vs Random Baseline ---")
    rand_scores = df[df["method"] == "random"].sort_values(["prompt_id", "target_removal"])["semantic_fidelity"].values
    for m in ["embedding", "gradient", "integrated_gradients", "llmlingua", "llmlingua2"]:
        m_scores = df[df["method"] == m].sort_values(["prompt_id", "target_removal"])["semantic_fidelity"].values
        diff = m_scores - rand_scores
        test_res = paired_difference_test(m_scores, rand_scores)
        d = cohens_d(m_scores, rand_scores)
        mean_val, lower, upper = bootstrap_ci(diff, n_boot=10000)
        print(f"  Method '{m:<20}' vs Random: Delta={mean_val:+.4f}, 95% CI=[{lower:+.4f}, {upper:+.4f}], p={test_res['p_value']:.2e}, Cohen's d={d:.4f}")

    print("\n--- Collapse Thresholds by Method & Category ---")
    coll_df = pd.read_csv("results/tables/collapse_thresholds.csv")
    print(coll_df.to_string(index=False))

    print("\n--- Failure Cases Count & Structure ---")
    with open("results/failure_cases.json", "r", encoding="utf-8") as f:
        fc = json.load(f)
    print(f"Total failure cases detected: {len(fc)}")
    if fc:
        print(f"Sample failure case 1: Prompt {fc[0]['prompt_id']}, Method {fc[0]['method']}, EmbSim={fc[0]['embedding_similarity']:.4f}, ConstraintF1={fc[0]['constraint_fidelity']:.4f}")


if __name__ == "__main__":
    audit_dataset()
    audit_gradient_scorer()
    audit_integrated_gradients()
    audit_subword_alignment()
    audit_pruning_discrete_rounding()
    audit_baselines()
    audit_blind_interpreter()
    audit_fidelity_metrics()
    audit_human_baseline()
    audit_full_results()
