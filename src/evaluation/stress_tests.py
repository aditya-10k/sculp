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
    
    Every POLARITY_REVERSED classification is strictly verified against the specific
    clause structure of the original prompt so that cross-clause tokens (e.g. 'not exceed'
    in a second sentence) never trigger false-positive inversions on the first sentence.
    
    Returns:
        (classification, rationale)
    """
    text_lower = compressed_text.lower().strip()
    ft = item.feature_type

    if ft == "negation_prohibition":
        has_negation = bool(re.search(r"\b(not|no|never|don't|dont|cannot|without)\b", text_lower))
        has_mention = bool(re.search(r"\b(mention|include|state|note)\b", text_lower))
        has_target = bool(re.search(r"\b(refund|refunds|compensation)\b", text_lower))

        if item.variant == "negative":
            # Original: "Draft an official apology to the customer. Do NOT mention refunds or compensation under any circumstances."
            if not has_negation and has_mention and has_target:
                return "POLARITY_REVERSED", "Negation ('NOT') dropped while 'mention' and 'refunds/compensation' retained; directive inverted to mandatory mention"
            elif has_negation and has_target:
                return "PRESERVED", "Negation operator and prohibited target ('refunds/compensation') both retained"
            elif not has_target:
                return "REMOVED", "Prohibited target ('refunds/compensation') completely removed"
            else:
                return "AMBIGUOUS", "Negation and verb dropped leaving bare noun ('refunds/compensation') without clear prohibition"
        else:
            # Original: "Draft an official apology to the customer. Explicitly mention refunds and compensation under all circumstances."
            if has_negation and (has_mention or has_target):
                return "POLARITY_REVERSED", "Negation introduced against affirmative requirement"
            elif has_mention and has_target:
                return "PRESERVED", "Affirmative 'mention' verb and target ('refunds/compensation') retained"
            elif not has_target:
                return "REMOVED", "Affirmative target ('refunds/compensation') completely removed"
            else:
                return "AMBIGUOUS", "Action verb ('mention') dropped leaving bare target nouns"

    elif ft == "numbers_bounds":
        # Both variants contain two clauses:
        # Clause 1: word count bound ("at most 50 words" vs "at least 500 words")
        # Clause 2: expenditure ceiling ("Do not exceed $10,000" vs "Do not exceed $10,000,000")
        has_exceed = bool(re.search(r"\bexceed(?:ing)?\b", text_lower))
        has_not_exceed = bool(re.search(r"\b(?:not|no|never)\s+exceed(?:ing)?\b", text_lower))
        has_unnegated_exceed = has_exceed and not has_not_exceed
        has_expenditure_or_amt = bool(re.search(r"\b(expenditure|budget|10|000)\b", text_lower))

        if item.variant == "bound_lower":
            # Original: "Summarize the project budget in at most 50 words. Do not exceed $10,000 total expenditure."
            has_50 = bool(re.search(r"\b50\b", text_lower))
            has_10k = bool(re.search(r"10\s*,?\s*000\b", text_lower))
            has_word_ceiling = bool(re.search(r"\b(at\s+most|most|maximum|max|under|below)\s+50\b", text_lower))
            has_word_floor = bool(re.search(r"\b(at\s+least|least|minimum|min|over|above)\s+50\b", text_lower))

            if has_unnegated_exceed and has_expenditure_or_amt:
                return "POLARITY_REVERSED", "Negation 'not' dropped before 'exceed', inverting 'Do not exceed $10,000 expenditure' into imperative 'exceed ... expenditure'"
            elif has_word_floor:
                return "POLARITY_REVERSED", "Word count ceiling ('at most 50') inverted to floor ('at least 50')"
            elif has_50 and has_word_ceiling and has_not_exceed and has_10k:
                return "PRESERVED", "Both upper word bound ('most 50') and negated budget ceiling ('not exceed $10,000') preserved"
            elif not has_50 and not has_10k:
                return "REMOVED", "Both numerical thresholds (50 words and $10,000) removed"
            else:
                return "AMBIGUOUS", "Numeric values partially retained but bound qualifier ('at most') or full dollar figure fragmented"
        else:
            # Original: "Summarize the project budget in at least 500 words. Do not exceed $10,000,000 total expenditure."
            has_500 = bool(re.search(r"\b500\b", text_lower))
            has_10m = bool(re.search(r"10\s*,?\s*000\s*,?\s*000\b", text_lower))
            has_word_floor = bool(re.search(r"\b(at\s+least|least|minimum|min|over|above)\s+500\b", text_lower))
            has_word_ceiling = bool(re.search(r"\b(at\s+most|most|maximum|max|under|below)\s+500\b", text_lower))

            if has_unnegated_exceed and has_expenditure_or_amt:
                return "POLARITY_REVERSED", "Negation 'not' dropped before 'exceed', inverting 'Do not exceed $10,000,000 expenditure' into imperative 'exceed ... expenditure'"
            elif has_word_ceiling:
                return "POLARITY_REVERSED", "Word count floor ('at least 500') inverted to ceiling ('at most 500')"
            elif has_500 and has_word_floor and has_not_exceed and has_10m:
                return "PRESERVED", "Both lower word bound ('least 500') and negated budget ceiling ('not exceed $10,000,000') preserved"
            elif not has_500 and not has_10m:
                return "REMOVED", "Both numerical thresholds (500 words and $10,000,000) removed"
            else:
                return "AMBIGUOUS", "Numeric values partially retained but bound qualifier ('at least') or full dollar figure fragmented"

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
        has_below = "below" in text_lower or "under" in text_lower
        has_above = "above" in text_lower or "over" in text_lower

        if item.variant == "allow_unless":
            # Original: "Authorize the wire transfer immediately, UNLESS the account balance is below $500."
            has_auth = "authorize" in text_lower
            if has_auth and not has_unless:
                return "POLARITY_REVERSED", "Exception operator ('UNLESS') dropped while 'Authorize' retained; conditional authorization inverted into unconditional authorization"
            elif has_auth and has_unless and has_below and has_500:
                return "PRESERVED", "Authorization action and full 'UNLESS ... below $500' exception preserved"
            elif not has_auth and not has_unless:
                return "REMOVED", "Core transaction action and conditional logic removed"
            else:
                return "AMBIGUOUS", "Conditional marker retained but threshold direction ('below') or amount ('$500') incomplete"
        else:
            # Original: "Reject the wire transfer immediately, UNLESS the account balance is above $500."
            has_reject = "reject" in text_lower
            if has_reject and not has_unless:
                return "POLARITY_REVERSED", "Exception operator ('UNLESS') dropped while 'Reject' retained; conditional rejection inverted into unconditional rejection"
            elif has_reject and has_unless and has_above and has_500:
                return "PRESERVED", "Rejection action and full 'UNLESS ... above $500' exception preserved"
            elif not has_reject and not has_unless:
                return "REMOVED", "Core action and exception removed"
            else:
                return "AMBIGUOUS", "Conditional marker retained but threshold direction ('above') or amount ('$500') incomplete"

    elif ft == "output_format":
        has_json = bool(re.search(r"\bjson\b", text_lower))
        has_negated_json = bool(re.search(r"\b(?:no|not|without|never)\s+(?:\w+\s+)?json\b", text_lower))
        has_affirmative_json = has_json and not has_negated_json

        has_md = bool(re.search(r"\bmarkdown\b", text_lower))
        has_negated_md = bool(re.search(r"\b(?:no|not|without|never)\s+(?:json\s+or\s+|(?:\w+\s+){0,2})?markdown\b", text_lower))
        has_affirmative_md = has_md and not has_negated_md

        has_html = bool(re.search(r"\b(html|table)\b", text_lower))
        has_raw_valid = bool(re.search(r"\b(raw|valid|rfc8259)\b", text_lower))

        if item.variant == "json_strict":
            # Original: "Output the database diagnosis strictly as raw valid RFC8259 JSON with no markdown code blocks or explanatory text."
            if not has_json and has_affirmative_md:
                return "POLARITY_REVERSED", "Required 'JSON' dropped and 'no' dropped before 'markdown', inverting format to Markdown"
            elif has_affirmative_json and has_negated_md:
                return "PRESERVED", "Affirmative JSON directive and 'no markdown' prohibition both preserved"
            elif has_affirmative_json and has_affirmative_md:
                return "AMBIGUOUS", "Negation 'no' dropped before 'markdown', leaving conflicting 'JSON markdown' terms"
            elif has_affirmative_json and not has_md:
                return "PRESERVED" if has_raw_valid else "AMBIGUOUS", "JSON format retained (markdown prohibition omitted)"
            else:
                return "REMOVED", "JSON formatting directive removed"
        else:
            # Original: "Output the database diagnosis strictly as an HTML table element with no JSON or markdown formatting."
            if not has_html and (has_affirmative_json or has_affirmative_md):
                return "POLARITY_REVERSED", "Required 'HTML table' dropped and negation 'no' dropped before 'JSON/markdown', inverting format requirement to JSON/Markdown"
            elif has_html and has_negated_json:
                return "PRESERVED", "Affirmative 'HTML table' directive and 'no JSON' prohibition both preserved"
            elif has_html and has_affirmative_json:
                return "AMBIGUOUS", "Negation 'no' dropped before 'JSON', leaving conflicting 'HTML table JSON' nouns"
            elif has_html and not has_json:
                return "PRESERVED", "Affirmative 'HTML table' format directive preserved"
            else:
                return "REMOVED", "Required 'HTML table' target removed (without affirmative inversion to JSON)"

    elif ft == "ordering_dependencies":
        has_first = "first" in text_lower
        has_second = "second" in text_lower
        has_then = bool(re.search(r"\b(then|after|finally)\b", text_lower))
        has_isolate = "isolate" in text_lower
        has_patch = "patch" in text_lower
        has_restart = "restart" in text_lower

        if item.variant == "isolate_first":
            pos_iso = text_lower.find("isolate")
            pos_patch = text_lower.find("patch")
            pos_restart = text_lower.find("restart")
            if pos_iso != -1 and pos_patch != -1 and pos_restart != -1:
                if pos_iso < pos_patch < pos_restart and (has_first or has_second or has_then):
                    return "PRESERVED", "Strict sequence isolate -> patch -> restart and ordering markers preserved"
                elif pos_restart < pos_iso or pos_patch < pos_iso:
                    return "POLARITY_REVERSED", "Action sequence reversed or permuted, violating causal safety"
                else:
                    return "AMBIGUOUS", "All three actions retained but explicit ordering qualifiers ('First', 'Second', 'Only then') dropped"
            elif not has_isolate and not has_patch and not has_restart:
                return "REMOVED", "Sequential workflow steps eliminated"
            else:
                return "AMBIGUOUS", "Partial workflow steps retained without complete sequence"
        else:
            pos_restart = text_lower.find("restart")
            pos_iso = text_lower.find("isolate")
            pos_patch = text_lower.find("patch")
            if pos_restart != -1 and pos_iso != -1 and pos_patch != -1:
                if pos_restart < pos_iso < pos_patch and (has_first or has_second or has_then):
                    return "PRESERVED", "Strict sequence restart -> isolate -> patch and ordering markers preserved"
                elif pos_iso < pos_restart:
                    return "POLARITY_REVERSED", "Action sequence inverted"
                else:
                    return "AMBIGUOUS", "All three actions retained but explicit ordering qualifiers dropped"
            elif not has_isolate and not has_patch and not has_restart:
                return "REMOVED", "Sequential workflow steps eliminated"
            else:
                return "AMBIGUOUS", "Partial workflow steps retained without complete sequence"

    elif ft == "priority_conflicts":
        has_brevity = "brevity" in text_lower
        has_precision = "precision" in text_lower
        has_prioritize = bool(re.search(r"\b(prioritize|priority|above all|favor)\b", text_lower))

        # Check which noun immediately follows 'prioritize'
        prioritize_brevity_match = bool(re.search(r"\bprioritize\s+(?:\w+\s+)?brevity\b", text_lower))
        prioritize_precision_match = bool(re.search(r"\bprioritize\s+(?:technical\s+)?precision\b", text_lower))

        if item.variant == "prioritize_brevity":
            # Original: "Summarize the technical briefing. If brevity conflicts with technical precision, prioritize brevity above all else."
            if prioritize_brevity_match:
                return "PRESERVED", "Explicit directive 'prioritize brevity' preserved"
            elif prioritize_precision_match or (has_prioritize and has_precision and not has_brevity):
                return "POLARITY_REVERSED", "Target 'brevity' dropped after 'prioritize' while 'precision' retained, inverting priority to precision"
            elif not has_brevity and not has_precision:
                return "REMOVED", "Conflict resolution directives removed"
            else:
                return "AMBIGUOUS", "Criteria mentioned without clear 'prioritize brevity' binding"
        else:
            # Original: "Summarize the technical briefing. If brevity conflicts with technical precision, prioritize technical precision above all else."
            if prioritize_precision_match:
                return "PRESERVED", "Explicit directive 'prioritize [technical] precision' preserved"
            elif prioritize_brevity_match or (has_prioritize and has_brevity and not has_precision):
                return "POLARITY_REVERSED", "Target 'precision' dropped while 'brevity' and 'prioritize' retained, inverting priority to brevity"
            elif not has_brevity and not has_precision:
                return "REMOVED", "Conflict resolution directives removed"
            else:
                return "AMBIGUOUS", "Criteria mentioned without clear 'prioritize precision' binding"

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
