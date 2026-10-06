"""Integration test verifying end-to-end pipeline execution."""

import pytest
from src.scoring.random import RandomScorer
from src.scoring.embedding import EmbeddingLeaveOneOutScorer
from src.pruning.pruner import PromptPruner
from src.interpretation.interpreter import BlindInterpreter
from src.evaluation.fidelity import FidelityEvaluatorSuite
from src.data.schema import GroundTruth
from src.analysis.collapse import detect_collapse_breakpoint


def test_end_to_end_pipeline():
    prompt = "That cat chased the mouse across the garden."
    gt = GroundTruth(
        intent="Describe that the cat chased the mouse across the garden",
        constraints=[],
        tone=["neutral"],
        context=["garden chasing"],
        steps=[]
    )

    scorer = EmbeddingLeaveOneOutScorer()
    pruner = PromptPruner()
    interpreter = BlindInterpreter(provider="mock")
    evaluator = FidelityEvaluatorSuite()

    # Score
    score_res = scorer.score_prompt(prompt, "pipe_test")
    assert len(score_res.word_scores) > 0

    # Prune across 3 ratios
    ratios = [0.0, 0.3, 0.7]
    fidelities = []

    for r in ratios:
        p_res = pruner.prune(score_res, target_removal=r)
        recon = interpreter.interpret(p_res.compressed_prompt)
        metrics = evaluator.evaluate(prompt, p_res.compressed_prompt, gt, recon)
        fidelities.append(metrics.semantic_fidelity)

    assert len(fidelities) == 3
    for f in fidelities:
        assert 0.0 <= f <= 1.0

    # Collapse breakpoint check
    bp = detect_collapse_breakpoint(ratios, fidelities)
    assert "collapse_ratio" in bp
    assert 0.0 <= bp["collapse_ratio"] <= 1.0
