"""Gradient-based token attribution importance scorer."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModel

from src.scoring.base import BaseScorer, PromptScoringResult, TokenScoreResult, WordScoreResult, min_max_normalize, compute_ranks
from src.tokenization.tokenizer import TokenizerWrapper


class GradientScorer(BaseScorer):
    """
    Computes token importance via embedding gradients with respect to a semantic representation objective.
    Objective: F(E) = cosine_similarity(mean_pooled(E), v_target_original).
    Importance: || dF / de_i ||_2.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "auto",
        seed: int = 42
    ):
        super().__init__(name="gradient", seed=seed)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer_wrapper = TokenizerWrapper(model_name)

    def score_prompt(self, prompt_text: str, prompt_id: str = "custom") -> PromptScoringResult:
        tokens, words, encoding = self.tokenizer_wrapper.tokenize_and_align(prompt_text)

        input_ids = torch.tensor([encoding["input_ids"]], device=self.device)
        attention_mask = torch.tensor([encoding["attention_mask"]], device=self.device)

        embedding_layer = self.model.get_input_embeddings()

        # Step 1: Forward pass to obtain target semantic representation v0
        with torch.no_grad():
            clean_embeds = embedding_layer(input_ids)
            clean_out = self.model(inputs_embeds=clean_embeds, attention_mask=attention_mask)
            # Masked mean pooling over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(clean_out.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(clean_out.last_hidden_state * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            v0 = (sum_embeddings / sum_mask).detach()  # [1, hidden_dim]
            v0_unit = F.normalize(v0, p=2, dim=-1)

        # Step 2: Backward pass with requires_grad on inputs_embeds
        input_embeds = embedding_layer(input_ids).detach().clone()
        input_embeds.requires_grad_(True)

        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        v = sum_embeddings / sum_mask
        v_unit = F.normalize(v, p=2, dim=-1)

        # Objective: Semantic projection onto unperturbed target representation
        semantic_objective = (v_unit * v0_unit).sum()

        self.model.zero_grad()
        if input_embeds.grad is not None:
            input_embeds.grad.zero_()

        semantic_objective.backward()
        grads = input_embeds.grad[0]  # [seq_len, hidden_dim]

        # Calculate L2 norm of gradient per token: || dF / de_i ||_2
        token_grad_norms = torch.norm(grads, p=2, dim=-1).detach().cpu().numpy()

        # Build token raw scores
        token_raw_scores = []
        for idx, t in enumerate(tokens):
            if t.is_special or t.word_id is None:
                token_raw_scores.append(0.0)
            else:
                token_raw_scores.append(float(token_grad_norms[idx]))

        # Aggregate subword gradients into word scores: word_score = sum(subtoken_scores)
        word_raw_scores = [0.0] * len(words)
        for idx, t in enumerate(tokens):
            if not t.is_special and t.word_id is not None:
                word_raw_scores[t.word_id] += token_raw_scores[idx]

        # Normalize word scores
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

        # Normalize token scores for eligible tokens
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
            method="gradient",
            prompt_text=prompt_text,
            token_scores=token_results,
            word_scores=word_results,
            metadata={
                "model": self.model_name,
                "objective": "semantic_cosine_projection_mean_pooled"
            }
        )
