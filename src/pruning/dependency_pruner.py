"""Dependency-Aware Span Pruner.

Mitigates catastrophic semantic splits (e.g. deleting negation while keeping the verb,
or deleting comparative quantifiers from numerical thresholds) by grouping tightly-coupled
syntactic/semantic dependencies into atomic spans that must be preserved or pruned together.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from src.scoring.base import PromptScoringResult, WordScoreResult
from src.tokenization.alignment import WordInfo
from src.tokenization.tokenizer import TokenizerWrapper
from src.pruning.pruner import PruningResult


@dataclass
class SemanticSpan:
    span_id: int
    span_type: str  # 'negation_clause', 'numerical_bound', 'conditional_exception', 'entity_mwe', 'atomic_word'
    words: List[WordInfo]
    text: str
    start_char: int
    end_char: int
    aggregate_score: float = 0.0

    @property
    def word_ids(self) -> Set[int]:
        return {w.word_id for w in self.words}


class DependencySpanDetector:
    """Detects syntactic and semantic dependency spans in prompt text."""

    NEGATION_WORDS = {"not", "no", "never", "dont", "don't", "cannot", "without", "prohibit", "prohibited", "neither", "nor"}
    BOUND_MODIFIERS = {"at", "most", "least", "maximum", "max", "minimum", "min", "under", "over", "below", "above", "exceed", "exceeding"}
    CONDITION_WORDS = {"unless", "except", "if", "provided", "conditional", "only"}

    @classmethod
    def detect_spans(cls, prompt_text: str, words_orig: List[WordInfo]) -> List[SemanticSpan]:
        if not words_orig:
            return []

        spans: List[SemanticSpan] = []
        assigned_word_ids: Set[int] = set()
        span_counter = 0

        # Pass 1: Multi-word regex patterns for coupled structures
        # 1. Negation + following 1-2 words (e.g. "do not mention", "never include")
        for i, w in enumerate(words_orig):
            if w.word_id in assigned_word_ids:
                continue
            w_lower = w.word.lower().strip()
            if w_lower in cls.NEGATION_WORDS:
                # Group with next 1 or 2 words if available
                span_words = [w]
                assigned_word_ids.add(w.word_id)
                for next_w in words_orig[i+1 : min(len(words_orig), i+3)]:
                    if next_w.word_id not in assigned_word_ids:
                        span_words.append(next_w)
                        assigned_word_ids.add(next_w.word_id)
                span_counter += 1
                spans.append(SemanticSpan(
                    span_id=span_counter,
                    span_type="negation_clause",
                    words=span_words,
                    text=prompt_text[span_words[0].start_char : span_words[-1].end_char],
                    start_char=span_words[0].start_char,
                    end_char=span_words[-1].end_char
                ))

        # 2. Bound quantifier + number/currency (e.g. "at most 50", "$10,000", "at least 500")
        for i, w in enumerate(words_orig):
            if w.word_id in assigned_word_ids:
                continue
            w_lower = w.word.lower().strip()
            if w_lower in cls.BOUND_MODIFIERS or re.search(r"\d", w.word) or "$" in w.word:
                # Look backwards or forwards for companion quantifier/number
                span_words = [w]
                assigned_word_ids.add(w.word_id)
                # Check next word
                if i + 1 < len(words_orig) and words_orig[i+1].word_id not in assigned_word_ids:
                    next_w = words_orig[i+1]
                    if next_w.word.lower().strip() in cls.BOUND_MODIFIERS or re.search(r"\d", next_w.word) or "$" in next_w.word:
                        span_words.append(next_w)
                        assigned_word_ids.add(next_w.word_id)
                span_counter += 1
                spans.append(SemanticSpan(
                    span_id=span_counter,
                    span_type="numerical_bound",
                    words=span_words,
                    text=prompt_text[span_words[0].start_char : span_words[-1].end_char],
                    start_char=span_words[0].start_char,
                    end_char=span_words[-1].end_char
                ))

        # 3. Conditional exceptions (e.g. "unless", "except")
        for i, w in enumerate(words_orig):
            if w.word_id in assigned_word_ids:
                continue
            w_lower = w.word.lower().strip()
            if w_lower in cls.CONDITION_WORDS:
                span_words = [w]
                assigned_word_ids.add(w.word_id)
                if i + 1 < len(words_orig) and words_orig[i+1].word_id not in assigned_word_ids:
                    span_words.append(words_orig[i+1])
                    assigned_word_ids.add(words_orig[i+1].word_id)
                span_counter += 1
                spans.append(SemanticSpan(
                    span_id=span_counter,
                    span_type="conditional_exception",
                    words=span_words,
                    text=prompt_text[span_words[0].start_char : span_words[-1].end_char],
                    start_char=span_words[0].start_char,
                    end_char=span_words[-1].end_char
                ))

        # Pass 2: Remaining unassigned words become atomic 1-word spans
        for w in words_orig:
            if w.word_id not in assigned_word_ids:
                span_counter += 1
                spans.append(SemanticSpan(
                    span_id=span_counter,
                    span_type="atomic_word",
                    words=[w],
                    text=w.word,
                    start_char=w.start_char,
                    end_char=w.end_char
                ))
                assigned_word_ids.add(w.word_id)

        # Sort spans in appearance order
        spans = sorted(spans, key=lambda s: s.start_char)
        return spans


class DependencyAwarePruner:
    """Prunes prompts by atomic semantic spans rather than individual words,
    preventing catastrophic semantic splitting and polarity reversal.
    """

    def __init__(self, tokenizer_model: str = "bert-base-uncased", aggregation: str = "max"):
        self.tokenizer_model = tokenizer_model
        self.aggregation = aggregation
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer_model)

    def prune(
        self,
        scoring_result: PromptScoringResult,
        target_removal: float
    ) -> PruningResult:
        prompt_text = scoring_result.prompt_text
        word_scores = scoring_result.word_scores
        score_by_id = {ws.word_id: ws.normalized_score for ws in word_scores}

        tokens_orig, words_orig, _ = self.tokenizer_wrapper.tokenize_and_align(prompt_text)
        n_orig_tokens = len([t for t in tokens_orig if not t.is_special])

        if target_removal <= 0.0 or not word_scores:
            return PruningResult(
                prompt_id=scoring_result.prompt_id,
                method=f"{scoring_result.method}_span",
                target_removal=0.0,
                original_tokens=n_orig_tokens,
                remaining_tokens=n_orig_tokens,
                actual_removal=0.0,
                retention=1.0,
                compressed_prompt=prompt_text,
                removed_words=[],
                kept_words=[ws.word for ws in word_scores],
                metadata={"span_pruning": True}
            )

        # 1. Detect semantic spans
        spans = DependencySpanDetector.detect_spans(prompt_text, words_orig)

        # 2. Compute aggregate score for each span
        for span in spans:
            scores = [score_by_id.get(w.word_id, 0.0) for w in span.words]
            if self.aggregation == "max":
                # Max score: if any word in span is critical (e.g. NOT), protect the whole span
                span.aggregate_score = max(scores) if scores else 0.0
            else:
                span.aggregate_score = sum(scores) / max(1, len(scores))

        # 3. Progressive pruning: sort spans by aggregate score ascending (lowest first)
        sorted_spans = sorted(spans, key=lambda s: s.aggregate_score)

        # Count total tokens and target removal
        target_tokens_to_remove = int(round(n_orig_tokens * target_removal))
        
        removed_spans: List[SemanticSpan] = []
        tokens_removed_count = 0

        for span in sorted_spans:
            # Estimate token cost of this span
            span_tokens = sum(len(w.subtoken_indices) for w in span.words)
            if tokens_removed_count + span_tokens <= target_tokens_to_remove or not removed_spans:
                removed_spans.append(span)
                tokens_removed_count += span_tokens
            if tokens_removed_count >= target_tokens_to_remove:
                break

        # Cannot remove ALL spans; keep at least highest scoring span
        if len(removed_spans) == len(spans):
            highest_span = max(spans, key=lambda s: s.aggregate_score)
            removed_spans = [s for s in removed_spans if s.span_id != highest_span.span_id]

        words_to_remove: List[WordInfo] = []
        for s in removed_spans:
            words_to_remove.extend(s.words)

        words_to_keep = [w for w in words_orig if w.word_id not in {w2.word_id for w2 in words_to_remove}]

        compressed_text = TokenizerWrapper.remove_word_spans(prompt_text, words_to_remove)
        if not compressed_text.strip():
            compressed_text = words_to_keep[0].word if words_to_keep else prompt_text

        tokens_comp, _, _ = self.tokenizer_wrapper.tokenize_and_align(compressed_text)
        n_comp_tokens = len([t for t in tokens_comp if not t.is_special])

        actual_removal = 1.0 - (n_comp_tokens / max(1, n_orig_tokens))
        retention = n_comp_tokens / max(1, n_orig_tokens)

        return PruningResult(
            prompt_id=scoring_result.prompt_id,
            method=f"{scoring_result.method}_span",
            target_removal=target_removal,
            original_tokens=n_orig_tokens,
            remaining_tokens=n_comp_tokens,
            actual_removal=max(0.0, actual_removal),
            retention=min(1.0, max(0.0, retention)),
            compressed_prompt=compressed_text,
            removed_words=[w.word for w in words_to_remove],
            kept_words=[w.word for w in words_to_keep],
            metadata={
                "spans_total": len(spans),
                "spans_removed": len(removed_spans),
                "span_types_removed": [s.span_type for s in removed_spans]
            }
        )
