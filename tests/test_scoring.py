"""Unit tests for the four importance scorers on the specified test sentences."""

import pytest
import numpy as np
from src.scoring.random import RandomScorer
from src.scoring.embedding import EmbeddingLeaveOneOutScorer
from src.scoring.gradient import GradientScorer
from src.scoring.integrated_gradients import IntegratedGradientScorer


@pytest.fixture(scope="module")
def scorers():
    return {
        "random": RandomScorer(seed=42),
        "embedding": EmbeddingLeaveOneOutScorer(seed=42),
        "gradient": GradientScorer(seed=42),
        "integrated_gradients": IntegratedGradientScorer(n_steps=10, seed=42)
    }


TEST_SENTENCES = [
    ("p_smoke_1", "That cat chased the mouse across the garden."),
    ("p_smoke_2", "The cat chased the mouse across the garden.")
]


@pytest.mark.parametrize("prompt_id, text", TEST_SENTENCES)
def test_all_scorers_basic_properties(scorers, prompt_id, text):
    for method_name, scorer in scorers.items():
        res = scorer.score_prompt(text, prompt_id=prompt_id)

        assert res.prompt_id == prompt_id
        assert res.method == method_name
        assert len(res.word_scores) > 0
        assert len(res.token_scores) > 0

        # Check words
        words = [ws.word for ws in res.word_scores]
        assert "cat" in words
        assert "mouse" in words
        assert "garden" in words
        assert "." in words

        # 1. Number of scored eligible tokens == number of eligible tokens
        eligible_tokens = [ts for ts in res.token_scores if not ts.is_special]
        assert len(eligible_tokens) > 0

        # Special tokens [CLS], [SEP] must not have positive rank or invalid state
        special_tokens = [ts for ts in res.token_scores if ts.is_special]
        assert len(special_tokens) >= 2
        for st in special_tokens:
            assert st.raw_score == 0.0
            assert st.normalized_score == 0.0

        # 2. All scores are finite and >= 0
        for ts in eligible_tokens:
            assert np.isfinite(ts.raw_score)
            assert np.isfinite(ts.normalized_score)
            assert ts.raw_score >= 0.0
            assert 0.0 <= ts.normalized_score <= 1.0
            assert 1 <= ts.rank <= len(eligible_tokens)

        for ws in res.word_scores:
            assert np.isfinite(ws.raw_score)
            assert np.isfinite(ws.normalized_score)
            assert ws.raw_score >= 0.0
            assert 0.0 <= ws.normalized_score <= 1.0
            assert 1 <= ws.rank <= len(res.word_scores)

        # 3. Check min-max normalization behavior
        norm_scores = [ws.normalized_score for ws in res.word_scores]
        assert max(norm_scores) == pytest.approx(1.0, abs=1e-5)
        assert min(norm_scores) == pytest.approx(0.0, abs=1e-5)
