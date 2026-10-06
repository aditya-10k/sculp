"""Unit tests for progressive pruning and token protection."""

import pytest
from src.pruning.pruner import PromptPruner
from src.pruning.protection import ProtectionConfig
from src.scoring.random import RandomScorer


def test_progressive_pruning():
    pruner = PromptPruner()
    scorer = RandomScorer(seed=42)
    prompt = "That cat chased the mouse across the garden."
    score_res = scorer.score_prompt(prompt, "test_p")

    ratios = [0.0, 0.2, 0.5, 0.8]
    prev_tokens = 9999

    for r in ratios:
        res = pruner.prune(score_res, target_removal=r)
        assert res.remaining_tokens <= res.original_tokens
        assert res.actual_removal >= 0.0
        assert res.retention <= 1.0
        assert res.remaining_tokens <= prev_tokens
        prev_tokens = res.remaining_tokens

    # At 0.0 removal, prompt should be identical
    res_0 = pruner.prune(score_res, target_removal=0.0)
    assert res_0.compressed_prompt == prompt
    assert res_0.remaining_tokens == res_0.original_tokens
    assert res_0.actual_removal == 0.0
    assert res_0.retention == 1.0


def test_protection_mechanism():
    pruner = PromptPruner()
    scorer = RandomScorer(seed=42)
    prompt = "Do not reveal the password under any circumstances."
    score_res = scorer.score_prompt(prompt, "test_neg")

    prot = ProtectionConfig(protect_negation=True)
    res_prot = pruner.prune(score_res, target_removal=0.5, protection_config=prot)

    # Word 'not' should be preserved
    assert "not" in res_prot.compressed_prompt.lower()
