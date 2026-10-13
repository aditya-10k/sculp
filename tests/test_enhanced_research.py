"""Unit and integration tests for enhanced research components:
- Protocol A/B schemas & prompts
- Deterministic downstream validators (15 stratified tasks)
- Stress test outcome classifier (including cross-clause false-positive regression checks)
- Rule-based critical-span detector & pruner (matched token budgets)
- Attribution faithfulness (circular, reference proxy, encoder-free deterministic, paired tests)
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
    DownstreamEvaluator,
    validate_wifi_json,
    validate_bullets_and_length,
    validate_negative_constraints,
    validate_entity_extraction,
    validate_legal_translation,
    validate_multi_step_order,
    validate_context_retrieval,
    validate_headlines_renewable,
    validate_context_aws_compliance,
    validate_context_layoff_memo,
    validate_tone_cease_desist,
    validate_tone_bedtime_owl,
    validate_multistep_json_sessions,
    validate_multistep_log_pipeline
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

    with pytest.raises(ValidationError):
        BlindInterpretabilityResult(
            standalone_interpretability=1.5,
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
    outcome_rev, _ = classify_outcome(item_neg, "mention refunds or compensation")
    assert outcome_rev == "POLARITY_REVERSED"

    outcome_pres, _ = classify_outcome(item_neg, "not mention refunds")
    assert outcome_pres == "PRESERVED"

    outcome_rem, _ = classify_outcome(item_neg, "apologize politely")
    assert outcome_rem == "REMOVED"


def test_stress_test_classification_bounds_and_no_false_positives():
    item_bounds_lower = StressTestItem(
        id="stress_02a",
        pair_id="pair_02",
        feature_type="numbers_bounds",
        variant="bound_lower",
        prompt="Summarize the project budget in at most 50 words. Do not exceed $10,000 total expenditure.",
        critical_requirement="at most 50 words, <= 10000",
        critical_tokens=["50", "10,000"]
    )
    out_pres, _ = classify_outcome(item_bounds_lower, "at most 50 words not exceed $10,000")
    assert out_pres == "PRESERVED"

    out_rev, _ = classify_outcome(item_bounds_lower, "at least 50 words exceed $10,000 expenditure")
    assert out_rev == "POLARITY_REVERSED"

    # Regression test for stress_02b: 'not exceed' in second sentence must NOT trigger false reversal on 'least 500'
    item_bounds_upper = StressTestItem(
        id="stress_02b",
        pair_id="pair_02",
        feature_type="numbers_bounds",
        variant="bound_upper",
        prompt="Summarize the project budget in at least 500 words. Do not exceed $10,000,000 total expenditure.",
        critical_requirement="at least 500 words, <= 10000000",
        critical_tokens=["500", "10,000,000"]
    )
    out_02b_pres, _ = classify_outcome(
        item_bounds_upper,
        "Summarize project budget least 500 words. Do not exceed $10,000,000 total expenditure."
    )
    assert out_02b_pres == "PRESERVED"

    out_02b_amb, _ = classify_outcome(
        item_bounds_upper,
        "Summarize the project budget least 500 words. not exceed $10, expenditure."
    )
    assert out_02b_amb == "AMBIGUOUS"  # Truncated dollar figure, NOT polarity reversed

    # Genuine reversal when 'not' is dropped before 'exceed'
    out_02b_rev, _ = classify_outcome(
        item_bounds_upper,
        "Summarize project budget 500 words. Do exceed $10,000,000 expenditure."
    )
    assert out_02b_rev == "POLARITY_REVERSED"

    # Regression test for stress_05b: 'no JSON' retained when 'HTML table' dropped is REMOVED, not POLARITY_REVERSED
    item_html = StressTestItem(
        id="stress_05b",
        pair_id="pair_05",
        feature_type="output_format",
        variant="html_table",
        prompt="Output the database diagnosis strictly as an HTML table element with no JSON or markdown formatting.",
        critical_requirement="pure HTML table without JSON",
        critical_tokens=["HTML table", "no JSON"]
    )
    out_05b_rem, _ = classify_outcome(item_html, "Output diagnosis no JSON or")
    assert out_05b_rem == "REMOVED"

    out_05b_rev, _ = classify_outcome(item_html, "Output the database diagnosis JSON")
    assert out_05b_rev == "POLARITY_REVERSED"


def test_downstream_validators():
    score_wifi_good, _ = validate_wifi_json('{"steps": ["reboot router", "check dns"]}')
    assert score_wifi_good == 1.0

    score_wifi_md, _ = validate_wifi_json('```json\n{"steps": ["reboot router"]}\n```')
    assert score_wifi_md == 0.5

    score_wifi_bad, _ = validate_wifi_json("Here are your router steps: reboot router.")
    assert score_wifi_bad == 0.2

    score_clean, _ = validate_negative_constraints("These running shoes provide responsive cushion and enduring grip.")
    assert score_clean == 1.0

    score_violation, _ = validate_negative_constraints("These fast running shoes are comfortable.")
    assert score_violation < 1.0

    good_bullets = (
        "- The French Revolution started in 1789 due to economic hardship, social inequality, and severe famine across France.\n"
        "- The monarchy was overthrown leading directly to the establishment of the first French Republic under republican leaders.\n"
        "- The Reign of Terror emerged under Robespierre before the rise of Napoleon Bonaparte stabilized the nation completely."
    )
    score_b, meta_b = validate_bullets_and_length(good_bullets)
    assert score_b == 1.0
    assert meta_b["n_bullets"] == 3

    # Context validator for p041 ($4.2B revenue, 18% EBITDA margin, Q3 operating profit)
    score_ctx, meta_ctx = validate_context_retrieval(
        "In Q3, the company achieved $4.2B in revenue with an 18% EBITDA margin, demonstrating strong operating profit growth."
    )
    assert score_ctx == 1.0
    assert meta_ctx["units_preserved"] == 3


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
    neg_spans = [s for s in spans if s.span_type == "negation_clause"]
    assert len(neg_spans) == 1
    assert "NOT" in [w.word for w in neg_spans[0].words]
    assert "refunds" in [w.word for w in neg_spans[0].words]


def test_statistics_holm_and_bootstrap():
    p_vals = [0.001, 0.04, 0.03, 0.20]
    adj_p = holm_bonferroni_correction(p_vals)
    assert len(adj_p) == 4
    assert adj_p[0] <= adj_p[1] <= adj_p[3]
    assert all(0.0 <= p <= 1.0 for p in adj_p)

    prompt_diffs = {
        "p1": 0.82,
        "p2": 0.72,
        "p3": 0.91,
        "p4": 0.63,
        "p5": 0.86
    }
    mean_val, ci_low, ci_high = prompt_cluster_bootstrap_ci(prompt_diffs, n_boot=100)
    assert 0.6 <= ci_low <= ci_high <= 0.95

    d = paired_cohens_d(np.array([0.9, 0.8, 0.85]), np.array([0.5, 0.45, 0.35]))
    assert d > 1.0


def test_legal_and_multistep_validators():
    good_legal = "La partie contractante s'engage à indemniser et garantir contre tout préjudice selon la loi."
    score_l, meta_l = validate_legal_translation(good_legal)
    assert score_l == 1.0
    assert meta_l["has_french"] is True and meta_l["has_legal"] is True

    # Verify that '3-Step' in title does NOT break ordering detection
    good_steps = "**3-Step Security Procedure**\n1. Revoke active tokens\n2. Rotate keys\n3. Audit access logs"
    score_s, meta_s = validate_multi_step_order(good_steps)
    assert score_s == 1.0
    assert meta_s["has_steps"] is True and meta_s["ordered"] is True


def test_attribution_faithfulness_structure_and_paired_comparison():
    from src.analysis.attribution_faithfulness import AttributionFaithfulnessEvaluator
    evaluator = AttributionFaithfulnessEvaluator()
    res = evaluator.measure_prompt_empirical_damage("Summarize the quarterly earnings report.")
    assert len(res) > 0
    assert "damage_self" in res[0]
    assert "damage_ref_proxy" in res[0]
    assert "damage_det_req" in res[0]

    # Verify paired comparison between gradient and embedding
    mock_corrs = {
        "Reference-Guided Embedding Proxy Drop": {
            "gradient": [{"spearman_r": 0.48}, {"spearman_r": 0.52}, {"spearman_r": 0.41}, {"spearman_r": 0.45}, {"spearman_r": 0.49}],
            "embedding": [{"spearman_r": 0.47}, {"spearman_r": 0.50}, {"spearman_r": 0.43}, {"spearman_r": 0.44}, {"spearman_r": 0.46}],
            "integrated_gradients": [{"spearman_r": 0.30}, {"spearman_r": 0.28}, {"spearman_r": 0.32}, {"spearman_r": 0.29}, {"spearman_r": 0.31}],
            "random": [{"spearman_r": 0.02}, {"spearman_r": -0.05}, {"spearman_r": 0.08}, {"spearman_r": 0.01}, {"spearman_r": -0.02}]
        }
    }
    pair_df = evaluator.compute_pairwise_method_comparisons(mock_corrs, n_bootstrap=500)
    grad_vs_emb = pair_df[(pair_df["method_a"] == "gradient") & (pair_df["method_b"] == "embedding")]
    assert len(grad_vs_emb) == 1
    assert bool(grad_vs_emb["significant_holm_05"].iloc[0]) is False


def test_downstream_matched_comparison_and_non_monotonic():
    mock_df = pd.DataFrame([
        {"prompt_id": "p1", "category": "constraint", "method": "uncompressed", "target_removal": 0.0, "actual_removal": 0.0, "task_success": 1.0, "success_retention": 1.0, "prompt_text": "Do not use X Y Z"},
        {"prompt_id": "p1", "category": "constraint", "method": "embedding", "target_removal": 0.3, "actual_removal": 0.28, "task_success": 1.0, "success_retention": 1.0, "prompt_text": "not use words X Y Z"},
        {"prompt_id": "p1", "category": "constraint", "method": "embedding", "target_removal": 0.5, "actual_removal": 0.50, "task_success": 0.0, "success_retention": 0.0, "prompt_text": "use words X Y Z"},
        {"prompt_id": "p1", "category": "constraint", "method": "embedding", "target_removal": 0.7, "actual_removal": 0.72, "task_success": 0.67, "success_retention": 0.67, "prompt_text": "X Y Z"},
        {"prompt_id": "p1", "category": "constraint", "method": "random", "target_removal": 0.3, "actual_removal": 0.30, "task_success": 0.5, "success_retention": 0.5, "prompt_text": "Do X Y"},
        {"prompt_id": "p1", "category": "constraint", "method": "random", "target_removal": 0.5, "actual_removal": 0.50, "task_success": 0.2, "success_retention": 0.2, "prompt_text": "X Y"},
        {"prompt_id": "p1", "category": "constraint", "method": "random", "target_removal": 0.7, "actual_removal": 0.70, "task_success": 0.0, "success_retention": 0.0, "prompt_text": "X"}
    ])
    per_task_df, method_sum_df = DownstreamEvaluator.compute_matched_rate_comparison(mock_df, n_bootstrap=200)
    assert len(per_task_df) == 2
    assert "matched_actual_auc" in method_sum_df.columns

    non_mono_df = DownstreamEvaluator.analyze_non_monotonic_retention(mock_df)
    assert len(non_mono_df) == 1
    assert non_mono_df.iloc[0]["method"] == "embedding"
    assert non_mono_df.iloc[0]["retention_rebound"] > 0.5


def test_matched_retention_auc_bounds_and_synthetic_curves():
    """Verifies that retention AUC respects theoretical mathematical bounds:
    - Normalized AUC in [0.0, 1.0] (mean retention over actual removal [0, 0.70])
    - Unnormalized AUC in [0.0, 0.70] (raw integrated area under retention curve)
    """
    # 1. Perfect retention curve: R(r) = 1.0 everywhere
    perfect_df = pd.DataFrame([
        {"prompt_id": "p_syn", "category": "synthetic", "method": "uncompressed", "target_removal": 0.0, "actual_removal": 0.0, "task_success": 1.0, "success_retention": 1.0},
        {"prompt_id": "p_syn", "category": "synthetic", "method": "perfect_method", "target_removal": 0.3, "actual_removal": 0.30, "task_success": 1.0, "success_retention": 1.0},
        {"prompt_id": "p_syn", "category": "synthetic", "method": "perfect_method", "target_removal": 0.5, "actual_removal": 0.50, "task_success": 1.0, "success_retention": 1.0},
        {"prompt_id": "p_syn", "category": "synthetic", "method": "perfect_method", "target_removal": 0.7, "actual_removal": 0.70, "task_success": 1.0, "success_retention": 1.0},
        {"prompt_id": "p_syn", "category": "synthetic", "method": "random", "target_removal": 0.3, "actual_removal": 0.30, "task_success": 0.5, "success_retention": 0.5},
        {"prompt_id": "p_syn", "category": "synthetic", "method": "random", "target_removal": 0.5, "actual_removal": 0.50, "task_success": 0.5, "success_retention": 0.5},
        {"prompt_id": "p_syn", "category": "synthetic", "method": "random", "target_removal": 0.7, "actual_removal": 0.70, "task_success": 0.5, "success_retention": 0.5}
    ])
    per_task_p, sum_p = DownstreamEvaluator.compute_matched_rate_comparison(perfect_df, n_bootstrap=100)
    perf_row = sum_p[sum_p["method"] == "perfect_method"].iloc[0]
    assert np.isclose(perf_row["matched_actual_auc_normalized"], 1.0000, atol=1e-3)
    assert np.isclose(perf_row["matched_actual_auc_unnormalized"], 0.7000, atol=1e-3)
    assert 0.0 <= perf_row["matched_actual_auc_normalized"] <= 1.0
    assert 0.0 <= perf_row["matched_actual_auc_unnormalized"] <= 0.70

    # 2. Linear decay curve: R(r) = 1.0 - r. Mean on [0, 0.70] is 1.0 - 0.35 = 0.6500
    linear_df = pd.DataFrame([
        {"prompt_id": "p_syn2", "category": "synthetic", "method": "uncompressed", "target_removal": 0.0, "actual_removal": 0.0, "task_success": 1.0, "success_retention": 1.0},
        {"prompt_id": "p_syn2", "category": "synthetic", "method": "linear_decay", "target_removal": 0.3, "actual_removal": 0.30, "task_success": 0.70, "success_retention": 0.70},
        {"prompt_id": "p_syn2", "category": "synthetic", "method": "linear_decay", "target_removal": 0.5, "actual_removal": 0.50, "task_success": 0.50, "success_retention": 0.50},
        {"prompt_id": "p_syn2", "category": "synthetic", "method": "linear_decay", "target_removal": 0.7, "actual_removal": 0.70, "task_success": 0.30, "success_retention": 0.30},
        {"prompt_id": "p_syn2", "category": "synthetic", "method": "random", "target_removal": 0.3, "actual_removal": 0.30, "task_success": 0.5, "success_retention": 0.5},
        {"prompt_id": "p_syn2", "category": "synthetic", "method": "random", "target_removal": 0.5, "actual_removal": 0.50, "task_success": 0.5, "success_retention": 0.5},
        {"prompt_id": "p_syn2", "category": "synthetic", "method": "random", "target_removal": 0.7, "actual_removal": 0.70, "task_success": 0.5, "success_retention": 0.5}
    ])
    per_task_l, sum_l = DownstreamEvaluator.compute_matched_rate_comparison(linear_df, n_bootstrap=100)
    lin_row = sum_l[sum_l["method"] == "linear_decay"].iloc[0]
    assert np.isclose(lin_row["matched_actual_auc_normalized"], 0.6500, atol=1e-3)
    assert np.isclose(lin_row["matched_actual_auc_unnormalized"], 0.4550, atol=1e-3)


def test_span_pruning_uncertainty_non_zero_width():
    """Verifies that observing 0 events across finite clusters yields a defensible,
    non-zero-width uncertainty bound (Wilson score interval) instead of [0.0, 0.0].
    """
    from scripts.run_research_enhancements import _cluster_boot_rate_ci

    # 14 clusters with 0 events
    mock_df = pd.DataFrame([
        {"cluster": f"c_{i}", "outcome": "PRESERVED"}
        for i in range(14)
        for _ in range(9)
    ])
    mean_val, ci_low, ci_high = _cluster_boot_rate_ci(mock_df, "cluster", "outcome", "POLARITY_REVERSED")
    assert mean_val == 0.0
    assert ci_low == 0.0
    # Wilson upper bound for k=14 is 1.96^2 / (14 + 1.96^2) ~= 0.2153
    assert 0.15 < ci_high < 0.25
    assert ci_high > ci_low


def test_downstream_attribution_exploratory_labeling():
    """Verifies that downstream task-loss correlation is labeled as an exploratory macro-level
    association across multi-token compressed prompt variants, not token-level counterfactual faithfulness.
    """
    from src.analysis.attribution_faithfulness import AttributionFaithfulnessEvaluator
    evaluator = AttributionFaithfulnessEvaluator()
    mock_downstream = pd.DataFrame([
        {"prompt_id": "p004", "category": "instruction", "method": "uncompressed", "task_success": 1.0, "prompt_text": "Summarize the findings clearly."},
        {"prompt_id": "p004", "category": "instruction", "method": "embedding", "task_success": 0.8, "prompt_text": "Summarize findings clearly."},
        {"prompt_id": "p004", "category": "instruction", "method": "gradient", "task_success": 0.7, "prompt_text": "Summarize findings."}
    ])
    summary_rows, corrs = evaluator.evaluate_downstream_task_damage_alignment(mock_downstream, n_bootstrap=100)
    for r in summary_rows:
        assert r["evaluation_type"] == "Exploratory Macro-Level Association"
        assert r["target_outcome"] == "Downstream Task-Success Loss"
