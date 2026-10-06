"""Base classes and normalization utilities for token and word importance scoring."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from src.tokenization.alignment import TokenInfo, WordInfo


@dataclass
class TokenScoreResult:
    token_index: int
    token_id: int
    token: str
    word_id: Optional[int]
    is_special: bool
    raw_score: float
    normalized_score: float
    rank: int  # 1 = most important

    def to_dict(self) -> dict:
        return {
            "token_index": self.token_index,
            "token_id": self.token_id,
            "token": self.token,
            "word_id": self.word_id,
            "is_special": self.is_special,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "rank": self.rank
        }


@dataclass
class WordScoreResult:
    word_id: int
    word: str
    start_char: int
    end_char: int
    subtoken_indices: List[int]
    raw_score: float
    normalized_score: float
    rank: int  # 1 = most important

    def to_dict(self) -> dict:
        return {
            "word_id": self.word_id,
            "word": self.word,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "subtoken_indices": self.subtoken_indices,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "rank": self.rank
        }


@dataclass
class PromptScoringResult:
    prompt_id: str
    method: str
    prompt_text: str
    token_scores: List[TokenScoreResult]
    word_scores: List[WordScoreResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "method": self.method,
            "prompt_text": self.prompt_text,
            "token_scores": [ts.to_dict() for ts in self.token_scores],
            "word_scores": [ws.to_dict() for ws in self.word_scores],
            "metadata": self.metadata
        }


def min_max_normalize(scores: List[float]) -> List[float]:
    """
    Prompt-relative min-max normalization:
    s'_i = (s_i - min) / (max - min)
    If all scores are equal or empty, returns 0.5 for each.
    """
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float64)
    min_v = np.min(arr)
    max_v = np.max(arr)
    diff = max_v - min_v
    if np.isclose(diff, 0.0) or np.isnan(diff):
        return [0.5 for _ in scores]
    norm = (arr - min_v) / diff
    return norm.tolist()


def compute_ranks(scores: List[float], descending: bool = True) -> List[int]:
    """
    Compute 1-indexed ranks. By default descending (highest score gets rank 1).
    Ties broken stably.
    """
    if not scores:
        return []
    # Order indices by score
    indexed = list(enumerate(scores))
    if descending:
        indexed.sort(key=lambda x: x[1], reverse=True)
    else:
        indexed.sort(key=lambda x: x[1])

    ranks = [0] * len(scores)
    for rank, (original_idx, _) in enumerate(indexed, start=1):
        ranks[original_idx] = rank
    return ranks


class BaseScorer(ABC):
    """Abstract base class for all importance scorers."""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed

    @abstractmethod
    def score_prompt(self, prompt_text: str, prompt_id: str = "custom") -> PromptScoringResult:
        """Score word and token importances for the given prompt."""
        pass
