"""Deterministic Random Importance Scorer (Control Condition)."""

import numpy as np
from typing import Optional
from src.scoring.base import BaseScorer, PromptScoringResult, TokenScoreResult, WordScoreResult, min_max_normalize, compute_ranks
from src.tokenization.tokenizer import TokenizerWrapper


class RandomScorer(BaseScorer):
    """
    Random Importance Scorer.
    Serves as the experimental control baseline. Deterministic given a seed.
    """

    def __init__(self, tokenizer_model: str = "bert-base-uncased", seed: int = 42):
        super().__init__(name="random", seed=seed)
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer_model)
        self.rng = np.random.default_rng(seed)

    def score_prompt(self, prompt_text: str, prompt_id: str = "custom") -> PromptScoringResult:
        tokens, words, _ = self.tokenizer_wrapper.tokenize_and_align(prompt_text)

        # Generate deterministic pseudo-random raw scores for words
        # Re-seed deterministically with hash of prompt + base seed for per-prompt reproducibility
        prompt_seed = (self.seed + abs(hash(prompt_id))) % (2**32 - 1)
        rng = np.random.default_rng(prompt_seed)

        word_raw_scores = rng.uniform(0.01, 1.0, size=len(words)).tolist()
        word_norm_scores = min_max_normalize(word_raw_scores)
        word_ranks = compute_ranks(word_norm_scores)

        word_results = []
        for w, raw_s, norm_s, r in zip(words, word_raw_scores, word_norm_scores, word_ranks):
            word_results.append(WordScoreResult(
                word_id=w.word_id,
                word=w.word,
                start_char=w.start_char,
                end_char=w.end_char,
                subtoken_indices=w.subtoken_indices,
                raw_score=float(raw_s),
                normalized_score=float(norm_s),
                rank=r
            ))

        # Distribute word score to subtokens
        token_results = []
        token_raw_scores = []
        for t in tokens:
            if t.is_special or t.word_id is None:
                token_raw_scores.append(0.0)
            else:
                w_obj = words[t.word_id]
                # Distribute word score proportionally across its subtokens
                n_sub = max(1, len(w_obj.subtoken_indices))
                token_raw_scores.append(word_raw_scores[t.word_id] / n_sub)

        # Normalize eligible tokens (ignoring special tokens)
        eligible_indices = [i for i, t in enumerate(tokens) if not t.is_special]
        eligible_scores = [token_raw_scores[i] for i in eligible_indices]
        eligible_norm = min_max_normalize(eligible_scores)
        eligible_ranks = compute_ranks(eligible_norm)

        norm_map = dict(zip(eligible_indices, eligible_norm))
        rank_map = dict(zip(eligible_indices, eligible_ranks))

        for idx, t in enumerate(tokens):
            if t.is_special:
                token_results.append(TokenScoreResult(
                    token_index=t.token_index,
                    token_id=t.token_id,
                    token=t.token,
                    word_id=t.word_id,
                    is_special=True,
                    raw_score=0.0,
                    normalized_score=0.0,
                    rank=9999
                ))
            else:
                token_results.append(TokenScoreResult(
                    token_index=t.token_index,
                    token_id=t.token_id,
                    token=t.token,
                    word_id=t.word_id,
                    is_special=False,
                    raw_score=token_raw_scores[idx],
                    normalized_score=norm_map.get(idx, 0.0),
                    rank=rank_map.get(idx, len(eligible_indices))
                ))

        return PromptScoringResult(
            prompt_id=prompt_id,
            method="random",
            prompt_text=prompt_text,
            token_scores=token_results,
            word_scores=word_results,
            metadata={"seed": self.seed, "prompt_seed": prompt_seed}
        )
