"""Cached Sentence Encoder providing fast in-memory embedding reuse across evaluators."""

from __future__ import annotations
from typing import List, Optional, Dict
import torch
from sentence_transformers import SentenceTransformer


class CachedSentenceEncoder:
    """Thread-safe, memory-cached wrapper around SentenceTransformer."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        model: Optional[SentenceTransformer] = None
    ):
        if model is not None:
            self.model = model
            self.device = str(model.device)
        else:
            self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
            self.model = SentenceTransformer(model_name, device=self.device)
        self._cache: Dict[str, torch.Tensor] = {}

    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text with cache lookup."""
        if text in self._cache:
            return self._cache[text]
        emb = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        self._cache[text] = emb
        return emb

    def encode_list(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of texts reusing cached tensors where available."""
        if not texts:
            return torch.empty((0, self.model.get_sentence_embedding_dimension()), device=self.device)

        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            embs = self.model.encode(uncached, convert_to_tensor=True, show_progress_bar=False)
            if len(uncached) == 1 and embs.dim() == 1:
                embs = embs.unsqueeze(0)
            for t, e in zip(uncached, embs):
                self._cache[t] = e

        return torch.stack([self._cache[t] for t in texts])


_GLOBAL_ENCODERS: Dict[str, CachedSentenceEncoder] = {}


def get_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> CachedSentenceEncoder:
    """Singleton getter for CachedSentenceEncoder."""
    if model_name not in _GLOBAL_ENCODERS:
        _GLOBAL_ENCODERS[model_name] = CachedSentenceEncoder(model_name=model_name)
    return _GLOBAL_ENCODERS[model_name]

