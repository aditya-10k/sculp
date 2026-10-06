"""Progressive lexical pruning engine supporting standard and protected modes."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
from src.scoring.base import PromptScoringResult, WordScoreResult
from src.tokenization.tokenizer import TokenizerWrapper
from src.pruning.protection import ProtectionConfig


@dataclass
class PruningResult:
    prompt_id: str
    method: str
    target_removal: float
    original_tokens: int
    remaining_tokens: int
    actual_removal: float  # CR = 1 - N_comp / N_orig
    retention: float       # Retention = N_comp / N_orig
    compressed_prompt: str
    removed_words: List[str] = field(default_factory=list)
    kept_words: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "method": self.method,
            "target_removal": self.target_removal,
            "original_tokens": self.original_tokens,
            "remaining_tokens": self.remaining_tokens,
            "actual_removal": round(self.actual_removal, 4),
            "retention": round(self.retention, 4),
            "compressed_prompt": self.compressed_prompt,
            "removed_words": self.removed_words,
            "kept_words": self.kept_words,
            "metadata": self.metadata
        }


class PromptPruner:
    """
    Executes progressive lexical pruning on scored prompts,
    handling subword alignment, span removals, and token retention metrics.
    """

    def __init__(self, tokenizer_model: str = "bert-base-uncased"):
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer_model)

    def prune(
        self,
        scoring_result: PromptScoringResult,
        target_removal: float,
        protection_config: Optional[ProtectionConfig] = None
    ) -> PruningResult:
        """
        Prune prompt by target_removal ratio (e.g. 0.0, 0.1, 0.3, 0.5, 0.8).
        Removes the lowest importance words while optionally respecting protected tokens.
        """
        prompt_text = scoring_result.prompt_text
        word_scores = scoring_result.word_scores

        tokens_orig, words_orig, _ = self.tokenizer_wrapper.tokenize_and_align(prompt_text)
        # Count non-special tokens as original token count
        n_orig_tokens = len([t for t in tokens_orig if not t.is_special])

        if target_removal <= 0.0 or not word_scores:
            return PruningResult(
                prompt_id=scoring_result.prompt_id,
                method=scoring_result.method,
                target_removal=0.0,
                original_tokens=n_orig_tokens,
                remaining_tokens=n_orig_tokens,
                actual_removal=0.0,
                retention=1.0,
                compressed_prompt=prompt_text,
                removed_words=[],
                kept_words=[ws.word for ws in word_scores],
                metadata={"protected": protection_config is not None}
            )

        # Classify candidate words into protected vs eligible
        eligible_words: List[WordScoreResult] = []
        protected_words: List[WordScoreResult] = []

        for ws in word_scores:
            if protection_config and protection_config.is_protected(ws.word):
                protected_words.append(ws)
            else:
                eligible_words.append(ws)

        # Number of words to remove
        total_words = len(word_scores)
        target_k = int(round(total_words * target_removal))
        # Cannot remove more than eligible words
        k_to_remove = min(target_k, len(eligible_words))

        # Sort eligible words by normalized_score ascending (lowest importance first)
        # Secondary sort by rank descending (higher rank number = lower importance)
        sorted_eligible = sorted(eligible_words, key=lambda w: (w.normalized_score, -w.rank))

        words_to_remove = sorted_eligible[:k_to_remove]
        words_to_keep = [w for w in word_scores if w not in words_to_remove]

        # Map to WordInfo for span removal
        remove_set = {w.word_id for w in words_to_remove}
        words_info_to_remove = [w for w in words_orig if w.word_id in remove_set]

        # Clean span-based reconstruction
        compressed_text = TokenizerWrapper.remove_word_spans(prompt_text, words_info_to_remove)
        if not compressed_text.strip():
            # If everything removed, keep highest ranked word as fallback
            highest_word = sorted(word_scores, key=lambda w: w.normalized_score, reverse=True)[0]
            compressed_text = highest_word.word

        # Measure compressed tokens
        tokens_comp, _, _ = self.tokenizer_wrapper.tokenize_and_align(compressed_text)
        n_comp_tokens = len([t for t in tokens_comp if not t.is_special])

        actual_removal = 1.0 - (n_comp_tokens / max(1, n_orig_tokens))
        retention = n_comp_tokens / max(1, n_orig_tokens)

        return PruningResult(
            prompt_id=scoring_result.prompt_id,
            method=scoring_result.method,
            target_removal=target_removal,
            original_tokens=n_orig_tokens,
            remaining_tokens=n_comp_tokens,
            actual_removal=max(0.0, actual_removal),
            retention=min(1.0, max(0.0, retention)),
            compressed_prompt=compressed_text,
            removed_words=[w.word for w in words_to_remove],
            kept_words=[w.word for w in words_to_keep],
            metadata={
                "k_words_removed": k_to_remove,
                "total_words": total_words,
                "protected_count": len(protected_words)
            }
        )
