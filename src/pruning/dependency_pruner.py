"""Rule-Based Critical-Span Pruner (Dependency-Aware Span Grouping).

Mitigates catastrophic semantic splits (e.g., deleting negation while keeping the governed verb,
or deleting comparative quantifiers from numerical thresholds) by grouping tightly coupled
lexical-semantic structures into atomic spans that must be preserved or pruned together.

Methodological Note:
This module implements deterministic rule-based critical-span grouping for high-risk
prompt constructs (negation clauses, numerical bounds, conditional exceptions, sequential
markers, and priority directives). It is NOT a universal syntactic constituency or
dependency parser.
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
    span_type: str  # 'negation_clause', 'numerical_bound', 'conditional_exception', 'priority_directive', 'atomic_word'
    words: List[WordInfo]
    text: str
    start_char: int
    end_char: int
    aggregate_score: float = 0.0

    @property
    def word_ids(self) -> Set[int]:
        return {w.word_id for w in self.words}


class DependencySpanDetector:
    """Detects rule-based critical lexical-semantic spans in prompt text."""

    AUX_WORDS = {"do", "does", "did", "must", "should", "can", "will", "may", "shall"}
    NEGATION_WORDS = {"not", "no", "never", "dont", "don't", "cannot", "without", "prohibit", "prohibited", "neither", "nor"}
    BOUND_MODIFIERS = {"at", "most", "least", "maximum", "max", "minimum", "min", "under", "over", "below", "above", "exceed", "exceeding", "exactly", "between"}
    CONDITION_WORDS = {"unless", "except", "provided", "only"}
    PRIORITY_WORDS = {"prioritize", "priority", "favor"}
    NUMBER_COMPONENT_RE = re.compile(r"^[\$\d,%\.]+$")

    @classmethod
    def _is_number_piece(cls, word_str: str) -> bool:
        w = word_str.strip()
        return bool(cls.NUMBER_COMPONENT_RE.match(w)) and (any(c.isdigit() for c in w) or w in {"$", ",", "%"})

    @classmethod
    def detect_spans(cls, prompt_text: str, words_orig: List[WordInfo]) -> List[SemanticSpan]:
        if not words_orig:
            return []

        spans: List[SemanticSpan] = []
        assigned_word_ids: Set[int] = set()
        span_counter = 0
        n_words = len(words_orig)

        # 1. Negation clauses: [optional aux] + negation + governed verb/object + contiguous numeric bound if present
        for i, w in enumerate(words_orig):
            if w.word_id in assigned_word_ids:
                continue
            w_lower = w.word.lower().strip()
            if w_lower in cls.NEGATION_WORDS:
                span_words: List[WordInfo] = []
                # Include preceding auxiliary verb if unassigned (e.g., "Do NOT", "must not")
                if i > 0 and words_orig[i - 1].word_id not in assigned_word_ids:
                    prev_lower = words_orig[i - 1].word.lower().strip()
                    if prev_lower in cls.AUX_WORDS:
                        span_words.append(words_orig[i - 1])
                        assigned_word_ids.add(words_orig[i - 1].word_id)

                span_words.append(w)
                assigned_word_ids.add(w.word_id)

                # Absorb following governed tokens (up to 3 words, plus any contiguous number pieces)
                j = i + 1
                steps_taken = 0
                while j < n_words and steps_taken < 3:
                    nw = words_orig[j]
                    if nw.word_id in assigned_word_ids:
                        break
                    nw_clean = nw.word.strip()
                    if nw_clean in {".", ";", ":"}:
                        break
                    span_words.append(nw)
                    assigned_word_ids.add(nw.word_id)
                    j += 1
                    steps_taken += 1
                    # If we just absorbed a bound verb like 'exceed' or a number piece, keep absorbing contiguous number pieces
                    if nw_clean.lower() in cls.BOUND_MODIFIERS or cls._is_number_piece(nw_clean):
                        while j < n_words and words_orig[j].word_id not in assigned_word_ids and cls._is_number_piece(words_orig[j].word):
                            span_words.append(words_orig[j])
                            assigned_word_ids.add(words_orig[j].word_id)
                            j += 1

                span_words = sorted(span_words, key=lambda x: x.start_char)
                span_counter += 1
                spans.append(SemanticSpan(
                    span_id=span_counter,
                    span_type="negation_clause",
                    words=span_words,
                    text=prompt_text[span_words[0].start_char : span_words[-1].end_char],
                    start_char=span_words[0].start_char,
                    end_char=span_words[-1].end_char
                ))

        # 2. Conditional exceptions (e.g. "UNLESS the account balance is below $500", "Only then restart")
        for i, w in enumerate(words_orig):
            if w.word_id in assigned_word_ids:
                continue
            w_lower = w.word.lower().strip()
            if w_lower in cls.CONDITION_WORDS:
                span_words = [w]
                assigned_word_ids.add(w.word_id)
                j = i + 1
                # Absorb up to the condition bound or governed verb (up to 6 tokens or until punctuation/number completion)
                max_lookahead = 7 if w_lower in {"unless", "except"} else 3
                steps = 0
                while j < n_words and steps < max_lookahead:
                    nw = words_orig[j]
                    if nw.word_id in assigned_word_ids:
                        break
                    if nw.word.strip() in {".", ";"}:
                        break
                    span_words.append(nw)
                    assigned_word_ids.add(nw.word_id)
                    j += 1
                    steps += 1
                    if cls._is_number_piece(nw.word):
                        while j < n_words and words_orig[j].word_id not in assigned_word_ids and cls._is_number_piece(words_orig[j].word):
                            span_words.append(words_orig[j])
                            assigned_word_ids.add(words_orig[j].word_id)
                            j += 1
                        break

                span_counter += 1
                spans.append(SemanticSpan(
                    span_id=span_counter,
                    span_type="conditional_exception",
                    words=span_words,
                    text=prompt_text[span_words[0].start_char : span_words[-1].end_char],
                    start_char=span_words[0].start_char,
                    end_char=span_words[-1].end_char
                ))

        # 3. Numerical bounds and formatted quantities (e.g. "at most 50 words", "at least 500 words", "$10,000,000")
        for i, w in enumerate(words_orig):
            if w.word_id in assigned_word_ids:
                continue
            w_lower = w.word.lower().strip()
            if w_lower in cls.BOUND_MODIFIERS or cls._is_number_piece(w.word):
                span_words = [w]
                assigned_word_ids.add(w.word_id)
                j = i + 1
                while j < n_words:
                    nw = words_orig[j]
                    if nw.word_id in assigned_word_ids:
                        break
                    nw_lower = nw.word.lower().strip()
                    if nw_lower in cls.BOUND_MODIFIERS or cls._is_number_piece(nw.word) or nw_lower in {"words", "sentences", "bullets", "points", "dollars", "usd", "percent"}:
                        span_words.append(nw)
                        assigned_word_ids.add(nw.word_id)
                        j += 1
                        if nw_lower in {"words", "sentences", "bullets", "points", "dollars", "usd", "percent"}:
                            break
                    else:
                        break

                span_counter += 1
                spans.append(SemanticSpan(
                    span_id=span_counter,
                    span_type="numerical_bound",
                    words=span_words,
                    text=prompt_text[span_words[0].start_char : span_words[-1].end_char],
                    start_char=span_words[0].start_char,
                    end_char=span_words[-1].end_char
                ))

        # 4. Priority resolution directives (e.g. "prioritize brevity", "prioritize technical precision")
        for i, w in enumerate(words_orig):
            if w.word_id in assigned_word_ids:
                continue
            w_lower = w.word.lower().strip()
            if w_lower in cls.PRIORITY_WORDS:
                span_words = [w]
                assigned_word_ids.add(w.word_id)
                for j in range(i + 1, min(n_words, i + 4)):
                    nw = words_orig[j]
                    if nw.word_id in assigned_word_ids or nw.word.strip() in {".", ",", ";"}:
                        break
                    span_words.append(nw)
                    assigned_word_ids.add(nw.word_id)
                span_counter += 1
                spans.append(SemanticSpan(
                    span_id=span_counter,
                    span_type="priority_directive",
                    words=span_words,
                    text=prompt_text[span_words[0].start_char : span_words[-1].end_char],
                    start_char=span_words[0].start_char,
                    end_char=span_words[-1].end_char
                ))

        # 5. Remaining unassigned words become atomic 1-word spans
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

        spans = sorted(spans, key=lambda s: s.start_char)
        return spans


class DependencyAwarePruner:
    """Prunes prompts by rule-based critical spans rather than individual words,
    preventing catastrophic semantic splitting and polarity reversal while matching
    target token budgets as closely as possible.
    """

    def __init__(self, tokenizer_model: str = "bert-base-uncased", aggregation: str = "max"):
        self.tokenizer_model = tokenizer_model
        self.aggregation = aggregation
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer_model)

    def prune(
        self,
        scoring_result: PromptScoringResult,
        target_removal: float,
        target_tokens_to_remove_override: Optional[int] = None
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
                metadata={
                    "span_pruning": True,
                    "method_type": "rule_based_critical_span_grouping"
                }
            )

        # 1. Detect rule-based critical spans
        spans = DependencySpanDetector.detect_spans(prompt_text, words_orig)

        # 2. Compute aggregate score for each span from the scorer's word scores
        for span in spans:
            scores = [score_by_id.get(w.word_id, 0.0) for w in span.words]
            if self.aggregation == "max":
                span.aggregate_score = max(scores) if scores else 0.0
            else:
                span.aggregate_score = sum(scores) / max(1, len(scores))

        # 3. Sort spans by (is_multiword_critical, aggregate_score) so low-importance atomic words
        # are pruned before multi-word critical spans of similar score, breaking ties by span length
        sorted_spans = sorted(
            spans,
            key=lambda s: (0 if s.span_type == "atomic_word" else 1, s.aggregate_score, len(s.words))
        )

        if target_tokens_to_remove_override is not None:
            target_tokens_to_remove = target_tokens_to_remove_override
        else:
            target_tokens_to_remove = int(round(n_orig_tokens * target_removal))

        removed_spans: List[SemanticSpan] = []
        tokens_removed_count = 0

        # Greedy selection with best-fit lookahead to match target_tokens_to_remove closely
        for span in sorted_spans:
            if len(removed_spans) >= len(spans) - 1:
                break
            span_tokens = max(1, sum(len(w.subtoken_indices) for w in span.words))
            curr_diff = abs(target_tokens_to_remove - tokens_removed_count)
            next_diff = abs(target_tokens_to_remove - (tokens_removed_count + span_tokens))
            if tokens_removed_count + span_tokens <= target_tokens_to_remove or next_diff < curr_diff or not removed_spans:
                removed_spans.append(span)
                tokens_removed_count += span_tokens
            if tokens_removed_count >= target_tokens_to_remove:
                break

        if len(removed_spans) == len(spans) and spans:
            highest_span = max(spans, key=lambda s: s.aggregate_score)
            removed_spans = [s for s in removed_spans if s.span_id != highest_span.span_id]

        words_to_remove: List[WordInfo] = []
        for s in removed_spans:
            words_to_remove.extend(s.words)

        remove_ids = {w.word_id for w in words_to_remove}
        words_to_keep = [w for w in words_orig if w.word_id not in remove_ids]

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
                "span_pruning": True,
                "method_type": "rule_based_critical_span_grouping",
                "spans_total": len(spans),
                "spans_removed": len(removed_spans),
                "span_types_removed": [s.span_type for s in removed_spans]
            }
        )
