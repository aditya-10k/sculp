"""Integrated Gradients token attribution importance scorer using Captum."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModel
from captum.attr import LayerIntegratedGradients

from src.scoring.base import BaseScorer, PromptScoringResult, TokenScoreResult, WordScoreResult, min_max_normalize, compute_ranks
from src.tokenization.tokenizer import TokenizerWrapper


class IntegratedGradientScorer(BaseScorer):
    """
    Computes token importance via LayerIntegratedGradients on the embedding layer using Captum.
    Integrates gradients along the straight-line interpolation between baseline and input.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "auto",
        n_steps: int = 25,
        seed: int = 42
    ):
        super().__init__(name="integrated_gradients", seed=seed)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.n_steps = n_steps
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer_wrapper = TokenizerWrapper(model_name)

    def score_prompt(self, prompt_text: str, prompt_id: str = "custom") -> PromptScoringResult:
        tokens, words, encoding = self.tokenizer_wrapper.tokenize_and_align(prompt_text)

        input_ids = torch.tensor([encoding["input_ids"]], device=self.device)
        attention_mask = torch.tensor([encoding["attention_mask"]], device=self.device)

        # Step 1: Precompute target pooled representation v0
        with torch.no_grad():
            clean_out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            mask_exp = attention_mask.unsqueeze(-1).expand(clean_out.last_hidden_state.size()).float()
            sum_emb = torch.sum(clean_out.last_hidden_state * mask_exp, dim=1)
            sum_m = mask_exp.sum(dim=1).clamp(min=1e-9)
            v0_unit = F.normalize(sum_emb / sum_m, p=2, dim=-1).detach()

        # Step 2: Define forward function for Captum
        def forward_func(inputs, mask):
            out = self.model(input_ids=inputs, attention_mask=mask)
            mask_expanded = mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(out.last_hidden_state * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            v = sum_embeddings / sum_mask
            # Linear projection onto target unit direction v0_unit
            return (v * v0_unit).sum(dim=-1)

        # Target embedding layer
        embedding_layer = self.model.embeddings

        lig = LayerIntegratedGradients(forward_func, embedding_layer)

        # Run LayerIntegratedGradients
        # Baseline is PAD token or zeros
        ref_input_ids = torch.zeros_like(input_ids)  # pad tokens
        attributions = lig.attribute(
            inputs=input_ids,
            baselines=ref_input_ids,
            additional_forward_args=(attention_mask,),
            n_steps=self.n_steps
        )  # [1, seq_len, hidden_dim]

        # Calculate token importance as L2 norm of attribution vector
        token_ig_norms = torch.norm(attributions[0], p=2, dim=-1).detach().cpu().numpy()

        token_raw_scores = []
        for idx, t in enumerate(tokens):
            if t.is_special or t.word_id is None:
                token_raw_scores.append(0.0)
            else:
                token_raw_scores.append(float(token_ig_norms[idx]))

        # Aggregate subword IG scores into word scores
        word_raw_scores = [0.0] * len(words)
        for idx, t in enumerate(tokens):
            if not t.is_special and t.word_id is not None:
                word_raw_scores[t.word_id] += token_raw_scores[idx]

        norm_word_scores = min_max_normalize(word_raw_scores)
        word_ranks = compute_ranks(norm_word_scores)

        word_results = []
        for w, raw_s, norm_s, r in zip(words, word_raw_scores, norm_word_scores, word_ranks):
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

        eligible_indices = [i for i, t in enumerate(tokens) if not t.is_special]
        eligible_scores = [token_raw_scores[i] for i in eligible_indices]
        eligible_norm = min_max_normalize(eligible_scores)
        eligible_ranks = compute_ranks(eligible_norm)

        norm_map = dict(zip(eligible_indices, eligible_norm))
        rank_map = dict(zip(eligible_indices, eligible_ranks))

        token_results = []
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
            method="integrated_gradients",
            prompt_text=prompt_text,
            token_scores=token_results,
            word_scores=word_results,
            metadata={
                "model": self.model_name,
                "n_steps": self.n_steps,
                "attribution_engine": "captum.LayerIntegratedGradients"
            }
        )
