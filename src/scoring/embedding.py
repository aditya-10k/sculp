"""Embedding Leave-One-Out (LOO) Importance Scorer using Sentence Transformers."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.scoring.base import BaseScorer, PromptScoringResult, TokenScoreResult, WordScoreResult, min_max_normalize, compute_ranks
from src.tokenization.tokenizer import TokenizerWrapper
from src.tokenization.alignment import WordInfo


class EmbeddingLeaveOneOutScorer(BaseScorer):
    """
    Computes semantic word importance by leaving out each word span from the prompt,
    computing the sentence embedding difference: I_i = 1 - cos(E_orig, E_-i).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_model: str = "bert-base-uncased",
        device: str = "auto",
        seed: int = 42
    ):
        super().__init__(name="embedding", seed=seed)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = SentenceTransformer(model_name, device=self.device)
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer_model)

    def score_prompt(self, prompt_text: str, prompt_id: str = "custom") -> PromptScoringResult:
        tokens, words, _ = self.tokenizer_wrapper.tokenize_and_align(prompt_text)

        if not words:
            return PromptScoringResult(
                prompt_id=prompt_id,
                method="embedding",
                prompt_text=prompt_text,
                token_scores=[],
                word_scores=[],
                metadata={"error": "empty_prompt"}
            )

        # 1. Encode original prompt
        e_orig = self.model.encode(prompt_text, convert_to_tensor=True, show_progress_bar=False)

        # 2. Generate leave-one-out sentences for each word span
        loo_texts: List[str] = []
        for w in words:
            loo_text = self.tokenizer_wrapper.remove_word_span(prompt_text, w)
            # If leaving out the word leaves empty text, fallback to empty string representation
            if not loo_text.strip():
                loo_text = ""
            loo_texts.append(loo_text)

        # 3. Batch encode all LOO sentences
        e_loos = self.model.encode(loo_texts, convert_to_tensor=True, show_progress_bar=False)

        # 4. Compute cosine similarity between E_orig and each E_-i
        # cos_sim returns [1, num_words] tensor
        sims = cos_sim(e_orig, e_loos)[0].detach().cpu().numpy()

        # Importance I_i = max(0.0, 1.0 - cos_sim)
        raw_word_scores = [max(0.0, float(1.0 - s)) for s in sims]
        norm_word_scores = min_max_normalize(raw_word_scores)
        word_ranks = compute_ranks(norm_word_scores)

        word_results = []
        for w, raw_s, norm_s, r in zip(words, raw_word_scores, norm_word_scores, word_ranks):
            word_results.append(WordScoreResult(
                word_id=w.word_id,
                word=w.word,
                start_char=w.start_char,
                end_char=w.end_char,
                subtoken_indices=w.subtoken_indices,
                raw_score=raw_s,
                normalized_score=norm_s,
                rank=r
            ))

        # 5. Distribute word scores to subword tokens
        token_results = []
        token_raw_scores = []
        for t in tokens:
            if t.is_special or t.word_id is None:
                token_raw_scores.append(0.0)
            else:
                w_obj = words[t.word_id]
                n_sub = max(1, len(w_obj.subtoken_indices))
                token_raw_scores.append(raw_word_scores[t.word_id] / n_sub)

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
            method="embedding",
            prompt_text=prompt_text,
            token_scores=token_results,
            word_scores=word_results,
            metadata={"model": self.model._modules['0'].auto_model.config._name_or_path if hasattr(self.model, '_modules') else "sentence-transformers"}
        )
