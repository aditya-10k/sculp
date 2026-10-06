"""Core Experiment Runner executing multi-method progressive pruning, interpretation, and evaluation."""

from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from src.data.dataset import DatasetManager
from src.data.schema import PromptItem
from src.scoring.random import RandomScorer
from src.scoring.embedding import EmbeddingLeaveOneOutScorer
from src.scoring.gradient import GradientScorer
from src.scoring.integrated_gradients import IntegratedGradientScorer
from src.pruning.pruner import PromptPruner
from src.pruning.protection import ProtectionConfig
from src.baselines.llmlingua import LLMLinguaCompressor
from src.baselines.llmlingua2 import LLMLingua2Compressor
from src.interpretation.interpreter import BlindInterpreter
from src.evaluation.fidelity import FidelityEvaluatorSuite
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.aggregation import aggregate_experiment_results
from src.analysis.statistics import compute_correlations


class ExperimentRunner:
    """
    Orchestrates the entire Semantic Collapse experiment pipeline.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset_manager = DatasetManager(config.prompts_path)

        # Output paths
        self.results_dir = Path(config.results_dir)
        self.scores_dir = self.results_dir / "scores"
        self.comp_dir = self.results_dir / "compressed"
        self.interp_dir = self.results_dir / "interpreter"
        for d in [self.scores_dir, self.comp_dir, self.interp_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print("[Runner] Initializing scorers and evaluators...")
        self.scorers = {
            "random": RandomScorer(tokenizer_model=config.attribution_model, seed=config.seed),
            "embedding": EmbeddingLeaveOneOutScorer(model_name=config.embedding_model, tokenizer_model=config.attribution_model, seed=config.seed),
            "gradient": GradientScorer(model_name=config.attribution_model, seed=config.seed),
            "integrated_gradients": IntegratedGradientScorer(model_name=config.attribution_model, seed=config.seed)
        }

        self.baselines = {
            "llmlingua": LLMLinguaCompressor(fallback_tokenizer=config.attribution_model),
            "llmlingua2": LLMLingua2Compressor(fallback_tokenizer=config.attribution_model)
        }

        self.pruner = PromptPruner(tokenizer_model=config.attribution_model)
        self.interpreter = BlindInterpreter(
            provider=config.interpreter_provider,
            model_name=config.interpreter_model,
            cache_dir=str(self.interp_dir)
        )
        self.evaluator = FidelityEvaluatorSuite(
            embedding_model_name=config.embedding_model,
            judge_provider=config.interpreter_provider
        )

    def run_full_experiment(self, max_prompts: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes the main experiment across all prompts, methods, and compression ratios.
        """
        all_prompts = self.dataset_manager.all_prompts()
        if max_prompts and max_prompts < len(all_prompts):
            prompts = all_prompts[:max_prompts]
        else:
            prompts = all_prompts

        print(f"[Runner] Starting evaluation on {len(prompts)} prompts across {len(self.config.compression_ratios)} compression ratios...")

        fidelity_rows = []
        word_scores_rows = []

        # Step 1: Score all prompts with the 4 attribution methods
        scored_cache: Dict[str, Dict[str, Any]] = {}  # prompt_id -> {method: PromptScoringResult}

        for prompt_item in tqdm(prompts, desc="Scoring prompts"):
            pid = prompt_item.id
            p_text = prompt_item.prompt
            scored_cache[pid] = {}

            for m_name, scorer in self.scorers.items():
                res = scorer.score_prompt(p_text, prompt_id=pid)
                scored_cache[pid][m_name] = res

                # Record word-level scores
                for ws in res.word_scores:
                    word_scores_rows.append({
                        "prompt_id": pid,
                        "category": prompt_item.category,
                        "method": m_name,
                        "word_id": ws.word_id,
                        "word": ws.word,
                        "raw_score": ws.raw_score,
                        "normalized_score": ws.normalized_score,
                        "rank": ws.rank
                    })

        # Step 2: Progressive pruning, blind interpretation, and semantic fidelity calculation
        total_steps = len(prompts) * (len(self.scorers) + len(self.baselines)) * len(self.config.compression_ratios)
        pbar = tqdm(total=total_steps, desc="Pruning & Evaluating")

        for prompt_item in prompts:
            pid = prompt_item.id
            category = prompt_item.category
            orig_text = prompt_item.prompt
            gt = prompt_item.ground_truth

            # Evaluate 4 attribution pruning methods
            for m_name in self.scorers.keys():
                score_res = scored_cache[pid][m_name]
                for cr in self.config.compression_ratios:
                    # Prune
                    prune_res = self.pruner.prune(score_res, target_removal=cr)
                    comp_text = prune_res.compressed_prompt

                    # Blind interpretation
                    recon = self.interpreter.interpret(comp_text)

                    # Semantic fidelity evaluation
                    fid = self.evaluator.evaluate(orig_text, comp_text, gt, recon)

                    fidelity_rows.append({
                        "prompt_id": pid,
                        "category": category,
                        "method": m_name,
                        "target_removal": cr,
                        "original_tokens": prune_res.original_tokens,
                        "compressed_tokens": prune_res.remaining_tokens,
                        "actual_removal": prune_res.actual_removal,
                        "retention": prune_res.retention,
                        "intent": fid.intent_score,
                        "constraints": fid.constraint_f1,
                        "constraint_recall": fid.constraint_recall,
                        "constraint_precision": fid.constraint_precision,
                        "tone": fid.tone_f1,
                        "context": fid.context_f1,
                        "dependencies": fid.dependency_preservation,
                        "semantic_fidelity": fid.semantic_fidelity,
                        "embedding_similarity": fid.embedding_similarity,
                        "llm_judge": fid.llm_judge_score,
                        "compressed_prompt": comp_text
                    })
                    pbar.update(1)

            # Evaluate baselines (LLMLingua & LLMLingua-2)
            for b_name, baseline in self.baselines.items():
                for cr in self.config.compression_ratios:
                    prune_res = baseline.compress(orig_text, prompt_id=pid, target_removal=cr)
                    comp_text = prune_res.compressed_prompt

                    recon = self.interpreter.interpret(comp_text)
                    fid = self.evaluator.evaluate(orig_text, comp_text, gt, recon)

                    fidelity_rows.append({
                        "prompt_id": pid,
                        "category": category,
                        "method": b_name,
                        "target_removal": cr,
                        "original_tokens": prune_res.original_tokens,
                        "compressed_tokens": prune_res.remaining_tokens,
                        "actual_removal": prune_res.actual_removal,
                        "retention": prune_res.retention,
                        "intent": fid.intent_score,
                        "constraints": fid.constraint_f1,
                        "constraint_recall": fid.constraint_recall,
                        "constraint_precision": fid.constraint_precision,
                        "tone": fid.tone_f1,
                        "context": fid.context_f1,
                        "dependencies": fid.dependency_preservation,
                        "semantic_fidelity": fid.semantic_fidelity,
                        "embedding_similarity": fid.embedding_similarity,
                        "llm_judge": fid.llm_judge_score,
                        "compressed_prompt": comp_text
                    })
                    pbar.update(1)

        pbar.close()

        fidelity_df = pd.DataFrame(fidelity_rows)
        word_scores_df = pd.DataFrame(word_scores_rows)

        # Aggregation of tables and failure cases
        summary_tables = aggregate_experiment_results(fidelity_df, word_scores_df, output_dir=str(self.results_dir))

        # Run specialized validation experiments:
        extra_results = self._run_specialized_experiments(prompts, scored_cache)
        summary_tables.update(extra_results)

        print("[Runner] Main experiment execution complete!")
        return fidelity_df, word_scores_df, summary_tables

    def _run_specialized_experiments(
        self,
        prompts: List[PromptItem],
        scored_cache: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes:
        1. Importance-score damage correlation (Spearman & Kendall)
        2. Paraphrase ranking stability test
        3. Human-compression baseline comparison (on 20 prompts)
        4. Protected vs Standard pruning ablation
        """
        out_dict = {}

        # 1. Importance vs Semantic Damage Correlation (on subset of prompts)
        print("[Runner] Running Importance-Score Damage Validation...")
        damage_correlations = []
        for p in prompts[:15]:
            orig = p.prompt
            for m in ["embedding", "gradient", "integrated_gradients"]:
                score_res = scored_cache[p.id][m]
                word_scores = score_res.word_scores
                if len(word_scores) < 3:
                    continue

                # Measure semantic damage of leaving out each word
                importances = [w.normalized_score for w in word_scores]
                damages = []
                e_orig = self.evaluator.embedding_model.encode(orig, convert_to_tensor=True, show_progress_bar=False)
                for w in word_scores:
                    sub_text = self.pruner.tokenizer_wrapper.remove_word_span(orig, w)
                    e_sub = self.evaluator.embedding_model.encode(sub_text, convert_to_tensor=True, show_progress_bar=False)
                    from sentence_transformers.util import cos_sim
                    damage = 1.0 - float(cos_sim(e_orig, e_sub)[0][0].detach().cpu().item())
                    damages.append(max(0.0, damage))

                corr = compute_correlations(importances, damages)
                damage_correlations.append({
                    "prompt_id": p.id,
                    "method": m,
                    "spearman": corr["spearman"],
                    "kendall": corr["kendall"]
                })

        damage_df = pd.DataFrame(damage_correlations)
        damage_csv = self.results_dir / "tables" / "importance_damage_correlation.csv"
        damage_cols = ["prompt_id", "method", "spearman", "kendall"]
        damage_df = pd.DataFrame(damage_correlations, columns=damage_cols)
        damage_csv = self.results_dir / "tables" / "importance_damage_correlation.csv"
        damage_df.to_csv(damage_csv, index=False)
        out_dict["importance_damage_correlation"] = str(damage_csv)

        # 2. Ranking Stability Across Paraphrases (Smoke Test Sentence)
        print("[Runner] Running Paraphrase Stability Test...")
        pair_a = "That cat chased the mouse across the garden."
        pair_b = "The cat ran after the mouse through the garden."

        res_a = self.scorers["embedding"].score_prompt(pair_a, "smoke_a")
        res_b = self.scorers["embedding"].score_prompt(pair_b, "smoke_b")

        # Key semantic concept alignments: (cat, cat), (chased, ran), (mouse, mouse), (garden, garden)
        concepts = ["cat", "mouse", "garden"]
        ranks_a = [next(w.rank for w in res_a.word_scores if w.word.lower() == c) for c in concepts]
        ranks_b = [next(w.rank for w in res_b.word_scores if w.word.lower() == c) for c in concepts]
        stability_corr = compute_correlations(ranks_a, ranks_b)

        out_dict["paraphrase_stability"] = stability_corr

        # 3. Human Compression Baseline on 20 prompts
        print("[Runner] Evaluating Human Compression Baseline (20 prompts)...")
        human_compressed_examples = {
            "p001": "Cat chased mouse across garden.",
            "p002": "Cat chased mouse across garden.",
            "p003": "Summarize attached quarterly earnings report findings.",
            "p004": "Translate paragraph to French.",
            "p005": "Extract meeting minutes action items and assignees.",
            "p006": "Generate five renewable energy blog headlines.",
            "p007": "Explain supervised vs unsupervised machine learning.",
            "p008": "Convert CSV table to JSON format.",
            "p009": "Proofread email draft for grammar and spelling.",
            "p010": "Draft polite job rejection reply.",
            "p021": "Summarize article in 5 sentences. No technical jargon. No author name.",
            "p022": "Explain quantum computing simply under 100 words without 'qubits' or 'superposition'.",
            "p023": "Write running shoes copy without 'comfortable', 'fast', or 'lightweight'.",
            "p024": "Provide Wi-Fi fix in valid JSON only.",
            "p025": "Describe French Revolution in three bullets of 15-20 words each.",
            "p041": "Respond to customer contacting third time on delayed order #9482 stuck 6 days.",
            "p061": "Explain service outage to frustrated customer empathetically, professionally, and concisely.",
            "p081": "1. Identify problem. 2. Explain cause. 3. Give two solutions. 4. Recommend safer option.",
            "p082": "1. Extract numbers. 2. Calculate mean and variance. 3. Interpret variance.",
            "p083": "Parse JSON, filter expired sessions, group by user ID, output counts."
        }

        human_cols = ["prompt_id", "category", "method", "original_tokens", "compressed_tokens", "retention", "actual_removal", "semantic_fidelity", "intent", "constraints", "embedding_similarity"]
        human_rows = []
        for pid, h_text in human_compressed_examples.items():
            p_item = self.dataset_manager.get_by_id(pid)
            if not p_item:
                continue

            recon = self.interpreter.interpret(h_text)
            fid = self.evaluator.evaluate(p_item.prompt, h_text, p_item.ground_truth, recon)

            tokens_orig, _, _ = self.pruner.tokenizer_wrapper.tokenize_and_align(p_item.prompt)
            tokens_comp, _, _ = self.pruner.tokenizer_wrapper.tokenize_and_align(h_text)
            n_orig = len([t for t in tokens_orig if not t.is_special])
            n_comp = len([t for t in tokens_comp if not t.is_special])

            human_rows.append({
                "prompt_id": pid,
                "category": p_item.category,
                "method": "human_compression",
                "original_tokens": n_orig,
                "compressed_tokens": n_comp,
                "retention": round(n_comp / max(1, n_orig), 4),
                "actual_removal": round(1.0 - (n_comp / max(1, n_orig)), 4),
                "semantic_fidelity": fid.semantic_fidelity,
                "intent": fid.intent_score,
                "constraints": fid.constraint_f1,
                "embedding_similarity": fid.embedding_similarity
            })

        human_df = pd.DataFrame(human_rows, columns=human_cols)
        human_csv = self.results_dir / "tables" / "human_baseline_comparison.csv"
        human_df.to_csv(human_csv, index=False)
        out_dict["human_baseline"] = str(human_csv)

        # 4. Protected vs Standard Pruning (Negation & Constraint Survival)
        print("[Runner] Running Protected vs Standard Pruning Experiment...")
        prot_config = ProtectionConfig(protect_negation=True, protect_numbers=True)
        constraint_prompts = self.dataset_manager.get_by_category("constraint")[:10]

        prot_cols = [
            "prompt_id", "target_removal", "standard_constraint_f1",
            "protected_constraint_f1", "standard_semantic_fidelity",
            "protected_semantic_fidelity", "delta_constraint_f1"
        ]
        prot_rows = []
        for p in constraint_prompts:
            if p.id in scored_cache and "embedding" in scored_cache[p.id]:
                score_res = scored_cache[p.id]["embedding"]
            else:
                score_res = self.scorers["embedding"].score_prompt(p.prompt, p.id)

            for cr in [0.3, 0.5, 0.7]:
                # Standard
                res_std = self.pruner.prune(score_res, target_removal=cr, protection_config=None)
                fid_std = self.evaluator.evaluate(p.prompt, res_std.compressed_prompt, p.ground_truth, self.interpreter.interpret(res_std.compressed_prompt))

                # Protected
                res_prot = self.pruner.prune(score_res, target_removal=cr, protection_config=prot_config)
                fid_prot = self.evaluator.evaluate(p.prompt, res_prot.compressed_prompt, p.ground_truth, self.interpreter.interpret(res_prot.compressed_prompt))

                prot_rows.append({
                    "prompt_id": p.id,
                    "target_removal": cr,
                    "standard_constraint_f1": fid_std.constraint_f1,
                    "protected_constraint_f1": fid_prot.constraint_f1,
                    "standard_semantic_fidelity": fid_std.semantic_fidelity,
                    "protected_semantic_fidelity": fid_prot.semantic_fidelity,
                    "delta_constraint_f1": round(fid_prot.constraint_f1 - fid_std.constraint_f1, 4)
                })

        prot_df = pd.DataFrame(prot_rows, columns=prot_cols)
        prot_csv = self.results_dir / "tables" / "protected_pruning_ablation.csv"
        prot_df.to_csv(prot_csv, index=False)
        out_dict["protected_pruning_ablation"] = str(prot_csv)

        # 5. Cross-Model Evaluation
        from src.experiments.cross_model import run_cross_model_experiment
        cross_df = run_cross_model_experiment(prompts, scored_cache, output_dir=str(self.results_dir))
        out_dict["cross_model"] = str(self.results_dir / "tables" / "cross_model_results.csv")

        return out_dict

