"""Unit tests for semantic fidelity evaluation suite."""

import pytest
from src.evaluation.intent import IntentEvaluator
from src.evaluation.constraints import ConstraintEvaluator
from src.evaluation.tone import ToneEvaluator
from src.evaluation.dependencies import DependencyEvaluator
from src.data.schema import GroundTruth
from src.interpretation.prompts import ReconstructedSemantics
from src.evaluation.fidelity import FidelityEvaluatorSuite


def test_intent_evaluator():
    evaluator = IntentEvaluator()
    sim_high = evaluator.evaluate(
        "Summarize the quarterly earnings report",
        "Summarize quarterly financial results"
    )
    sim_low = evaluator.evaluate(
        "Summarize the quarterly earnings report",
        "Bake a chocolate birthday cake"
    )
    assert 0.0 <= sim_high <= 1.0
    assert 0.0 <= sim_low <= 1.0
    assert sim_high > sim_low


def test_constraint_evaluator():
    evaluator = ConstraintEvaluator()
    gt = ["under 100 words", "do not use technical terminology"]
    pred_exact = ["under 100 words", "do not use technical terminology"]
    res_exact = evaluator.evaluate(gt, pred_exact)
    assert res_exact["recall"] == pytest.approx(1.0, abs=1e-3)
    assert res_exact["f1"] == pytest.approx(1.0, abs=1e-3)

    pred_empty = []
    res_empty = evaluator.evaluate(gt, pred_empty)
    assert res_empty["recall"] == 0.0
    assert res_empty["f1"] == 0.0


def test_dependency_evaluator():
    evaluator = DependencyEvaluator()
    gt_steps = [
        "First identify the problem",
        "Second explain cause",
        "Third offer solution",
        "Finally pick best option"
    ]
    # Exact in order
    res_ordered = evaluator.evaluate(gt_steps, gt_steps)
    assert res_ordered["step_recall"] == 1.0
    assert res_ordered["ordering_preservation"] == 1.0
    assert res_ordered["dependency_preservation"] == 1.0

    # Reversed order
    reversed_steps = list(reversed(gt_steps))
    res_rev = evaluator.evaluate(gt_steps, reversed_steps)
    assert res_rev["step_recall"] == 1.0
    assert res_rev["ordering_preservation"] < 0.5


def test_fidelity_suite():
    suite = FidelityEvaluatorSuite()
    gt = GroundTruth(
        intent="Summarize article in 5 sentences",
        constraints=["exactly five sentences"],
        tone=["neutral"],
        context=["news article"],
        steps=[]
    )
    recon = ReconstructedSemantics(
        task="Summarize article in five sentences",
        constraints=["five sentences"],
        tone=["neutral"],
        context=["news article"],
        required_actions=["summarize"],
        ordering_dependencies=[]
    )
    res = suite.evaluate(
        original_prompt="Summarize article in 5 sentences. Exactly five sentences.",
        compressed_prompt="Summarize article in 5 sentences. five sentences.",
        ground_truth=gt,
        reconstructed=recon
    )
    assert 0.0 <= res.semantic_fidelity <= 1.0
    assert 0.0 <= res.embedding_similarity <= 1.0
    assert res.semantic_fidelity > 0.6
