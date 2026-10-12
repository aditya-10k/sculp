"""Unit and integration tests for enhanced research components:
- Protocol A/B schemas & prompts
- Deterministic downstream validators
- Stress test outcome classifier
- Dependency-aware span detector & pruner
- Holm-Bonferroni & cluster bootstrap statistical routines
"""

import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.interpretation.prompts import (
    BlindInterpretabilityResult,
    ReferenceGuidedPreservationResult,
    format_protocol_a_prompt,
    format_protocol_b_prompt
)
from src.evaluation.stress_tests import (
    StressTestItem,
    classify_outcome
)
from src.evaluation.downstream import (
    validate_wifi_json,
    validate_bullets_and_length,
    validate_negative_constraints,
    validate_entity_extraction,
    validate_legal_translation,
    validate_multi_step_order
)
from src.pruning.dependency_pruner import (
    DependencySpanDetector,
    DependencyAwarePruner
)
from src.scoring.base import PromptScoringResult, WordScoreResult
from src.tokenization.alignment import WordInfo
from src.analysis.statistics import (
    holm_bonferroni_correction,
    prompt_cluster_bootstrap_ci,
    paired_cohens_d
)


def test_protocol_a_schema():
    # Valid instance
    res = BlindInterpretabilityResult(
        standalone_interpretability=0.85,
        clarity=0.90,
        actionability=0.80,
        inferred_task="apologize to client",
        detected_ambiguities=[],
        reconstructed_components={"task": "apology", "tone": ["formal"]}
    )
    assert res.clarity == 0.90
    assert res.standalone_interpretability == 0.85

    # Out of bounds check
    with pytest.raises(ValidationError):
        BlindInterpretabilityResult(
            standalone_interpretability=1.5,  # > 1.0
            clarity=0.5,
            actionability=0.5
        )


def test_protocol_b_schema():
    res = ReferenceGuidedPreservationResult(
        intent_preservation=0.9,
        constraint_preservation=0.8,
        context_preservation=0.7,
        tone_preservation=0.7,
        dependency_preservation=0.6,
        composite_preservation=0.74,
        preserved_requirements=["apology tone", "client context"],
        lost_requirements=[],
        ambiguities=[]
    )
    assert res.composite_preservation == 0.74
    assert len(res.preserved_requirements) == 2


def test_protocol_prompt_formatters():
    prompt_a = format_protocol_a_prompt("That cat chased mouse garden")
    assert "That cat chased mouse garden" in prompt_a
    assert "standalone_interpretability" in prompt_a

    prompt_b = format_protocol_b_prompt(
        compressed_prompt="cat chased mouse garden",
        reference_intent="Cat chased mouse across the garden",
        reference_constraints=["neutral tone"],
        reference_entities=["cat", "mouse", "garden"]
    )
    assert "cat chased mouse garden" in prompt_b
    assert "Structured Reference Criteria" in prompt_b


def test_stress_test_classification_negation():
    item_neg = StressTestItem(
        id="stress_01a",
        pair_id="pair_01",
        feature_type="negation_prohibition",
        variant="negative",
        prompt="Do NOT mention refunds or compensation.",
        critical_requirement="strictly prohibit refunds",
        critical_tokens=["NOT", "refunds"]
    )
    # Case 1: Polarity reversed (negation dropped)
    outcome_rev, _ = classify_outcome(item_neg, "mention refunds or compensation")
    assert outcome_rev == "POLARITY_REVERSED"

    # Case 2: Preserved
    outcome_pres, _ = classify_outcome(item_neg, "not mention refunds")
    assert outcome_pres == "PRESERVED"

    # Case 3: Removed
    outcome_rem, _ = classify_outcome(item_neg, "apologize politely")
    assert outcome_rem == "REMOVED"


def test_stress_test_classification_bounds():
    item_bounds = StressTestItem(
        id="stress_02a",
        pair_id="pair_02",
        feature_type="numbers_bounds",
        variant="bound_lower",
        prompt="at most 50 words. Do not exceed $10,000.",
        critical_requirement="at most 50 words, <= 10000",
        critical_tokens=["50", "10,000"]
    )
    # Preserved
    out_pres, _ = classify_outcome(item_bounds, "at most 50 words not exceed $10,000")
    assert out_pres == "PRESERVED"

    # Polarity reversed (direction inverted)
    out_rev, _ = classify_outcome(item_bounds, "at least 50 words exceed $10,000")
    assert out_rev == "POLARITY_REVERSED"


def test_downstream_validators():
    # Wi-Fi JSON
    score_wifi_good, _ = validate_wifi_json('{"steps": ["reboot router", "check dns"]}')
    assert score_wifi_good == 1.0

    score_wifi_md, _ = validate_wifi_json('```json\n{"steps": ["reboot router"]}\n```')
    assert score_wifi_md == 0.5  # markdown penalty

    score_wifi_bad, _ = validate_wifi_json("Here are your router steps: reboot router.")
    assert score_wifi_bad == 0.2  # text only

    # Negative constraints (forbidden words: comfortable, fast, lightweight)
    score_clean, _ = validate_negative_constraints("These running shoes provide responsive cushion and enduring grip.")
    assert score_clean == 1.0

    score_violation, _ = validate_negative_constraints("These fast running shoes are comfortable.")
    assert score_violation < 1.0

    # Bullets & word length (French Revolution, exactly 3 bullets, 15-20 words each)
    good_bullets = (
        "- The French Revolution started in 1789 due to economic hardship, social inequality, and severe famine across France.\n"
        "- The monarchy was overthrown leading directly to the establishment of the first French Republic under republican leaders.\n"
        "- The Reign of Terror emerged under Robespierre before the rise of Napoleon Bonaparte stabilized the nation completely."
    )
    score_b, meta_b = validate_bullets_and_length(good_bullets)
    assert score_b == 1.0
    assert meta_b["n_bullets"] == 3


def test_dependency_span_detector():
    words = [
        WordInfo(word_id=0, word="Do", start_char=0, end_char=2, subtoken_indices=[1]),
        WordInfo(word_id=1, word="NOT", start_char=3, end_char=6, subtoken_indices=[2]),
        WordInfo(word_id=2, word="mention", start_char=7, end_char=14, subtoken_indices=[3]),
        WordInfo(word_id=3, word="refunds", start_char=15, end_char=22, subtoken_indices=[4])
    ]
    prompt = "Do NOT mention refunds"
    spans = DependencySpanDetector.detect_spans(prompt, words)
    assert len(spans) > 0
    # NOT + mention should be grouped in a negation_clause span
    neg_spans = [s for s in spans if s.span_type == "negation_clause"]
    assert len(neg_spans) == 1
    assert "NOT" in [w.word for w in neg_spans[0].words]


def test_statistics_holm_and_bootstrap():
    # Test Holm-Bonferroni correction
    p_vals = [0.001, 0.04, 0.03, 0.20]
    adj_p = holm_bonferroni_correction(p_vals)
    assert len(adj_p) == 4
    assert adj_p[0] <= adj_p[1] <= adj_p[3]
    assert all(0.0 <= p <= 1.0 for p in adj_p)

    # Test prompt cluster bootstrap CI
    prompt_diffs = {
        "p1": 0.82,
        "p2": 0.72,
        "p3": 0.91,
        "p4": 0.63,
        "p5": 0.86
    }
    mean_val, ci_low, ci_high = prompt_cluster_bootstrap_ci(prompt_diffs, n_boot=100)
    assert 0.6 <= ci_low <= ci_high <= 0.95

    # Test Cohen's d (paired)
    d = paired_cohens_d(np.array([0.9, 0.8, 0.85]), np.array([0.5, 0.45, 0.35]))
    assert d > 1.0  # Large positive effect size


def test_legal_and_multistep_validators():
    # Legal translation
    good_legal = "La partie contractante s'engage à indemniser et garantir contre tout préjudice selon la loi."
    score_l, meta_l = validate_legal_translation(good_legal)
    assert score_l == 1.0
    assert meta_l["has_french"] is True and meta_l["has_legal"] is True

    # Multi-step ordering
    good_steps = "Follow this sequence:\n1. Revoke active tokens\n2. Rotate keys\n3. Audit access logs"
    score_s, meta_s = validate_multi_step_order(good_steps)
    assert score_s == 1.0
    assert meta_s["has_steps"] is True and meta_s["ordered"] is True


def test_attribution_faithfulness_structure():
    from src.analysis.attribution_faithfulness import AttributionFaithfulnessEvaluator
    evaluator = AttributionFaithfulnessEvaluator()
    res = evaluator.measure_prompt_empirical_damage("Summarize the quarterly earnings report.")
    assert len(res) > 0
    assert "damage_self" in res[0]
    assert "damage_gt" in res[0]

