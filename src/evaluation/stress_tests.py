"""Semantic Stress-Test Suite for Critical Failure Mode Evaluation.

Evaluates prompt compression algorithms against 7 critical semantic dimensions:
1. Negation / Prohibition
2. Numerical Bounds & Quantifiers
3. Entities & Identifiers
4. Conditional Exceptions
5. Output Formatting Constraints
6. Step Ordering Dependencies
7. Priority Conflicts

Classifies post-compression outcomes into:
- PRESERVED: Critical condition and semantics fully intact
- REMOVED: Critical condition or tokens deleted
- POLARITY_REVERSED: Negation/condition inverted, changing the imperative directive
- AMBIGUOUS: Fragmented or syntactically mangled text obscuring original intent
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from src.data.schema import PromptItem
from src.scoring.random import RandomScorer
from src.scoring.embedding import EmbeddingLeaveOneOutScorer
from src.scoring.gradient import GradientScorer
from src.scoring.integrated_gradients import IntegratedGradientScorer
from src.baselines.llmlingua import LLMLinguaCompressor
from src.baselines.llmlingua2 import LLMLingua2Compressor
from src.pruning.pruner import PromptPruner, PruningResult


@dataclass
class StressTestItem:
    id: str
    pair_id: str
    feature_type: str
    variant: str
    prompt: str
    critical_requirement: str
    critical_tokens: List[str]


def load_stress_test_prompts(path: str = "data/processed/stress_test_prompts.json") -> List[StressTestItem]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stress test prompts file not found at {path}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [StressTestItem(**item) for item in data]


def classify_outcome(
    item: StressTestItem,
    compressed_text: str
) -> Tuple[str, str]:
    """Classifies the compression outcome into PRESERVED, REMOVED, POLARITY_REVERSED, or AMBIGUOUS.
    
    Returns:
        (classification, rationale)
    """
    text_lower = compressed_text.lower().strip()
    words = set(re.findall(r"\b\w+\b", text_lower))
    ft = item.feature_type
    
    if ft == "negation_prohibition":
        # Check for negation words
        has_negation = bool(re.search(r"\b(not|no|never|don't|dont|cannot)\b", text_lower))
        has_mention = bool(re.search(r"\b(mention|include|state|note)\b", text_lower))
        has_target = bool(re.search(r"\b(refund|refunds|compensation)\b", text_lower))
        
        if item.variant == "negative":
            # Original: "Do NOT mention refunds or compensation"
            if not has_negation and has_mention and has_target:
                return "POLARITY_REVERSED", "Negation dropped while mention and target retained; prompt now mandates mentioning refunds"
            elif has_negation and has_target:
                return "PRESERVED", "Negation and prohibited target both retained"
            elif not has_target:
                return "REMOVED", "Target prohibition completely omitted"
            else:
                return "AMBIGUOUS", "Ambiguous negation fragments without clear imperative direction"
        else:
            # Original: "Explicitly mention refunds and compensation"
            has_explicit = bool(re.search(r"\b(explicitly|must|always)\b", text_lower))
            if has_negation and (has_mention or has_target):
                return "POLARITY_REVERSED", "Spurious negation introduced or aligned against mandatory directive"
            elif has_mention and has_target:
                return "PRESERVED", "Affirmative mention directive and target retained"
            elif not has_target:
                return "REMOVED", "Affirmative requirement targets removed"
            else:
                return "AMBIGUOUS", "Incomplete affirmative specification"

    elif ft == "numbers_bounds":
        has_50 = "50" in compressed_text
        has_10k = bool(re.search(r"10,?000", compressed_text))
        has_500 = "500" in compressed_text
        has_10m = bool(re.search(r"10,?000,?000", compressed_text))
        has_at_most = bool(re.search(r"\b(at most|maximum|max|under|below|not exceed)\b", text_lower))
        has_at_least = bool(re.search(r"\b(at least|minimum|min|above|over|exceeding)\b", text_lower))
        
        if item.variant == "bound_lower":
            # bound: at most 50 words, <= $10,000
            if has_50 and has_10k and has_at_most:
                return "PRESERVED", "Both upper bounds (50 words, $10k) and ceiling qualifiers preserved"
            elif has_50 and has_at_least:
                return "POLARITY_REVERSED", "Bound modifier inverted from 'at most' to 'at least'"
            elif not has_50 and not has_10k:
                return "REMOVED", "Both numerical bounds stripped completely"
            else:
                return "AMBIGUOUS", "Numerical bounds partially lost or quantifier missing"
        else:
            # bound: at least 500 words, <= $10,000,000
            if has_500 and has_10m and has_at_least:
                return "PRESERVED", "Both lower bound (500 words) and ceiling ($10m) preserved"
            elif has_500 and has_at_most:
                return "POLARITY_REVERSED", "Bound modifier inverted from 'at least' to 'at most'"
            elif not has_500 and not has_10m:
                return "REMOVED", "Both numerical boundaries stripped"
            else:
                return "AMBIGUOUS", "Numerical thresholds fragmented or quantifier missing"

    elif ft == "entities_identifiers":
        if item.variant == "entity_london":
            has_ticket = "9482" in compressed_text
            has_person = "sarah" in text_lower or "jenkins" in text_lower
            has_loc = "london" in text_lower
            if has_ticket and has_person and has_loc:
                return "PRESERVED", "Ticket #9482, engineer Sarah Jenkins, and London datacenter all intact"
            elif not has_ticket and not has_person and not has_loc:
                return "REMOVED", "All routing identifiers eliminated"
            else:
                return "AMBIGUOUS", "Partial identifier retention causing routing ambiguity"
        else:
            has_ticket = "2849" in compressed_text
            has_person = "mark" in text_lower or "davies" in text_lower
            has_loc = "tokyo" in text_lower
            if has_ticket and has_person and has_loc:
                return "PRESERVED", "Ticket #2849, engineer Mark Davies, and Tokyo datacenter all intact"
            elif not has_ticket and not has_person and not has_loc:
                return "REMOVED", "All routing identifiers eliminated"
            else:
                return "AMBIGUOUS", "Partial identifier retention causing routing ambiguity"

    elif ft == "conditional_exceptions":
        has_unless = bool(re.search(r"\b(unless|except|if|provided)\b", text_lower))
        has_500 = "500" in compressed_text
        if item.variant == "allow_unless":
            has_auth = "authorize" in text_lower
            if has_auth and not has_unless and not has_500:
                return "POLARITY_REVERSED", "Exception clause dropped; conditional authorization transformed into unconditional authorization"
            elif has_auth and has_unless and has_500:
                return "PRESERVED", "Authorization action and UNLESS boundary condition preserved"
            elif not has_auth and not has_unless:
                return "REMOVED", "Core transaction action and conditional logic removed"
            else:
                return "AMBIGUOUS", "Fragmented conditional statement with unclear boundary logic"
        else:
            has_reject = "reject" in text_lower
            if has_reject and not has_unless and not has_500:
                return "POLARITY_REVERSED", "Exception clause dropped; conditional rejection transformed into unconditional rejection"
            elif has_reject and has_unless and has_500:
                return "PRESERVED", "Rejection action and UNLESS boundary condition preserved"
            elif not has_reject and not has_unless:
                return "REMOVED", "Core action and exception removed"
            else:
                return "AMBIGUOUS", "Fragmented conditional statement with unclear logic"

    elif ft == "output_format":
        has_json = "json" in text_lower
        has_html = "html" in text_lower or "table" in text_lower
        has_no_md = bool(re.search(r"\b(no markdown|raw|rfc8259)\b", text_lower))
        if item.variant == "json_strict":
            if has_html and not has_json:
                return "POLARITY_REVERSED", "Format altered to conflicting format (HTML)"
            elif has_json and has_no_md:
                return "PRESERVED", "JSON specification and formatting restrictions preserved"
            elif has_json and not has_no_md:
                return "AMBIGUOUS", "JSON mentioned but raw formatting / no-markdown constraint dropped"
            else:
                return "REMOVED", "Formatting directives completely removed"
        else:
            if has_json and not has_html:
                return "POLARITY_REVERSED", "Format altered to conflicting format (JSON)"
            elif has_html:
                return "PRESERVED", "HTML table directive preserved"
            else:
                return "REMOVED", "Format specifications removed"

    elif ft == "ordering_dependencies":
        # Check ordering markers and sequence
        has_first = "first" in text_lower
        has_second = "second" in text_lower
        has_then = bool(re.search(r"\b(then|after|finally)\b", text_lower))
        has_isolate = "isolate" in text_lower
        has_patch = "patch" in text_lower
        has_restart = "restart" in text_lower
        
        if item.variant == "isolate_first":
            # Expect isolate before patch before restart
            pos_iso = text_lower.find("isolate")
            pos_patch = text_lower.find("patch")
            pos_restart = text_lower.find("restart")
            if pos_iso != -1 and pos_patch != -1 and pos_restart != -1:
                if pos_iso < pos_patch < pos_restart and (has_first or has_second or has_then):
                    return "PRESERVED", "Strict sequence isolate -> patch -> restart and ordering markers preserved"
                elif pos_restart < pos_iso or pos_patch < pos_iso:
                    return "POLARITY_REVERSED", "Action sequence reversed or permuted, violating causal safety"
                else:
                    return "AMBIGUOUS", "Actions preserved but ordering qualifiers weakened"
            elif not has_isolate and not has_patch and not has_restart:
                return "REMOVED", "Sequential workflow steps eliminated"
            else:
                return "AMBIGUOUS", "Partial workflow steps retained without clear sequence"
        else:
            pos_restart = text_lower.find("restart")
            pos_iso = text_lower.find("isolate")
            pos_patch = text_lower.find("patch")
            if pos_restart != -1 and pos_iso != -1 and pos_patch != -1:
                if pos_restart < pos_iso < pos_patch and (has_first or has_second or has_then):
                    return "PRESERVED", "Strict sequence restart -> isolate -> patch preserved"
                elif pos_iso < pos_restart:
                    return "POLARITY_REVERSED", "Action sequence inverted"
                else:
                    return "AMBIGUOUS", "Actions preserved but ordering qualifiers weakened"
            elif not has_isolate and not has_patch and not has_restart:
                return "REMOVED", "Sequential workflow steps eliminated"
            else:
                return "AMBIGUOUS", "Partial workflow steps retained"

    elif ft == "priority_conflicts":
        has_brevity = "brevity" in text_lower
        has_precision = "precision" in text_lower or "technical" in text_lower
        has_prioritize = bool(re.search(r"\b(prioritize|priority|above all|favor)\b", text_lower))
        
        if item.variant == "prioritize_brevity":
            pos_prio = text_lower.find("prioritize")
            pos_brev = text_lower.find("brevity")
            pos_prec = text_lower.find("precision")
            if has_prioritize and has_brevity and (pos_prec == -1 or (pos_brev != -1 and abs(pos_prio - pos_brev) < 30)):
                return "PRESERVED", "Explicit resolution prioritizing brevity over precision preserved"
            elif has_prioritize and has_precision and not has_brevity:
                return "POLARITY_REVERSED", "Priority inverted to technical precision due to brevity deletion"
            elif not has_brevity and not has_precision:
                return "REMOVED", "Conflict resolution directives removed"
            else:
                return "AMBIGUOUS", "Both criteria present without unambiguous priority winner"
        else:
            pos_prio = text_lower.find("prioritize")
            pos_prec = text_lower.find("precision")
            if has_prioritize and has_precision and ("brevity" not in text_lower or (pos_prec != -1 and abs(pos_prio - pos_prec) < 30)):
                return "PRESERVED", "Explicit resolution prioritizing precision over brevity preserved"
            elif has_prioritize and has_brevity and not has_precision:
                return "POLARITY_REVERSED", "Priority inverted to brevity due to precision deletion"
            elif not has_brevity and not has_precision:
                return "REMOVED", "Conflict resolution directives removed"
            else:
                return "AMBIGUOUS", "Both criteria present without unambiguous priority winner"

    # Default fallback
    return "AMBIGUOUS", "Unclassified compression outcome"


class StressTestEvaluator:
    """Executes the Semantic Stress Test Suite across multiple compression methods."""

    def __init__(
        self,
        attribution_model: str = "bert-base-uncased",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        seed: int = 42
    ):
        self.attribution_model = attribution_model
        self.seed = seed
        self.pruner = PromptPruner(tokenizer_model=attribution_model)

        print("[StressTest] Initializing scorers and baselines...")
        self.scorers = {
            "random": RandomScorer(tokenizer_model=attribution_model, seed=seed),
            "embedding": EmbeddingLeaveOneOutScorer(model_name=embedding_model, tokenizer_model=attribution_model, seed=seed),
            "gradient": GradientScorer(model_name=attribution_model, seed=seed),
            "integrated_gradients": IntegratedGradientScorer(model_name=attribution_model, seed=seed)
        }
        self.baselines = {
            "llmlingua": LLMLinguaCompressor(fallback_tokenizer=attribution_model),
            "llmlingua2": LLMLingua2Compressor(fallback_tokenizer=attribution_model)
        }

    def evaluate_all(
        self,
        prompts: Optional[List[StressTestItem]] = None,
        ratios: List[float] = [0.3, 0.5, 0.7],
        output_csv: str = "results/tables/stress_test_results.csv"
    ) -> pd.DataFrame:
        if prompts is None:
            prompts = load_stress_test_prompts()

        records = []
        all_methods = list(self.scorers.keys()) + list(self.baselines.keys())

        print(f"[StressTest] Evaluating {len(prompts)} stress test prompts across {len(all_methods)} methods and {len(ratios)} ratios...")

        for item in prompts:
            prompt_obj = PromptItem(
                id=item.id,
                category=item.feature_type,
                prompt=item.prompt,
                ground_truth=None,
                metadata={"pair_id": item.pair_id, "variant": item.variant}
            )

            # Score with attribution scorers
            scored_map = {}
            for s_name, scorer in self.scorers.items():
                scored_map[s_name] = scorer.score_prompt(item.prompt, prompt_id=item.id)

            for ratio in ratios:
                for method_name in all_methods:
                    if method_name in self.scorers:
                        pr_res = self.pruner.prune(scored_map[method_name], target_removal=ratio)
                    else:
                        pr_res = self.baselines[method_name].compress(item.prompt, prompt_id=item.id, target_removal=ratio)

                    classification, rationale = classify_outcome(item, pr_res.compressed_prompt)

                    records.append({
                        "prompt_id": item.id,
                        "pair_id": item.pair_id,
                        "feature_type": item.feature_type,
                        "variant": item.variant,
                        "method": method_name,
                        "target_removal": ratio,
                        "actual_removal": pr_res.actual_removal,
                        "retention": pr_res.retention,
                        "original_prompt": item.prompt,
                        "compressed_prompt": pr_res.compressed_prompt,
                        "outcome": classification,
                        "rationale": rationale
                    })

        df = pd.DataFrame(records)
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"[StressTest] Successfully saved {len(df)} stress test evaluations to {output_csv}")
        return df
