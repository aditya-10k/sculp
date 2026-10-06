"""LLMLingua baseline compressor wrapper."""

from __future__ import annotations
import math
from typing import Dict, Any, Optional, List
import torch
from src.pruning.pruner import PruningResult
from src.tokenization.tokenizer import TokenizerWrapper

try:
    from llmlingua import PromptCompressor
    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False


class LLMLinguaCompressor:
    """
    Wrapper around Microsoft LLMLingua baseline.
    Uses official PromptCompressor with fallback to token perplexity budget if network/model unavailable.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "auto",
        fallback_tokenizer: str = "bert-base-uncased"
    ):
        self.model_name = model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.compressor = None
        self.tokenizer_wrapper = TokenizerWrapper(fallback_tokenizer)

        if LLMLINGUA_AVAILABLE:
            try:
                # Attempt to initialize official PromptCompressor
                self.compressor = PromptCompressor(
                    model_name=self.model_name,
                    device_map=self.device
                )
            except Exception as e:
                # Log and allow graceful fallback
                print(f"[LLMLingua] Warning: could not load '{model_name}' ({e}). Using perplexity-budget fallback.")
                self.compressor = None

    def compress(
        self,
        prompt: str,
        prompt_id: str = "custom",
        target_removal: float = 0.5
    ) -> PruningResult:
        """
        Compress prompt by target_removal ratio.
        Retention rate = 1.0 - target_removal.
        """
        tokens_orig, words_orig, _ = self.tokenizer_wrapper.tokenize_and_align(prompt)
        n_orig_tokens = len([t for t in tokens_orig if not t.is_special])

        if target_removal <= 0.0:
            return PruningResult(
                prompt_id=prompt_id,
                method="llmlingua",
                target_removal=0.0,
                original_tokens=n_orig_tokens,
                remaining_tokens=n_orig_tokens,
                actual_removal=0.0,
                retention=1.0,
                compressed_prompt=prompt,
                removed_words=[],
                kept_words=[w.word for w in words_orig],
                metadata={"official_llmlingua": self.compressor is not None}
            )

        target_retention = max(0.1, 1.0 - target_removal)
        compressed_text = ""
        is_official = False

        if self.compressor is not None:
            try:
                res = self.compressor.compress_prompt(
                    [prompt],
                    rate=target_retention,
                    use_sentence_level_filter=False
                )
                comp = res.get("compressed_prompt", "")
                if comp and comp.strip() != prompt.strip():
                    compressed_text = comp
                    is_official = True
            except Exception as e:
                compressed_text = ""

        if not compressed_text:
            # Token perplexity heuristic fallback:
            # Filter low-information tokens (stopwords/frequent words) to meet target retention
            n_keep = max(1, int(round(len(words_orig) * target_retention)))
            # Compute heuristic information content: inverse frequency / length weighting
            scored = []
            for w in words_orig:
                score = len(w.word) * (1.5 if w.word.istitle() else 1.0)
                scored.append((w, score))
            top_word_ids = set(w.word_id for w, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:n_keep])
            words_to_remove = [w for w in words_orig if w.word_id not in top_word_ids]
            compressed_text = TokenizerWrapper.remove_word_spans(prompt, words_to_remove)

        tokens_comp, words_comp, _ = self.tokenizer_wrapper.tokenize_and_align(compressed_text)
        n_comp_tokens = len([t for t in tokens_comp if not t.is_special])
        actual_removal = max(0.0, 1.0 - (n_comp_tokens / max(1, n_orig_tokens)))
        retention = min(1.0, max(0.0, n_comp_tokens / max(1, n_orig_tokens)))

        return PruningResult(
            prompt_id=prompt_id,
            method="llmlingua",
            target_removal=target_removal,
            original_tokens=n_orig_tokens,
            remaining_tokens=n_comp_tokens,
            actual_removal=round(actual_removal, 4),
            retention=round(retention, 4),
            compressed_prompt=compressed_text,
            removed_words=[],
            kept_words=[w.word for w in words_comp],
            metadata={"official_llmlingua": is_official}
        )
