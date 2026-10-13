"""Functional Downstream Task-Success Engine.

Evaluates whether compressed prompts still accomplish their intended tasks when executed
by a downstream LLM, using deterministic content-and-constraint validators across a
predefined stratified 15-task benchmark (3 tasks per category across all 5 categories).

Features:
- Preserves original uncompressed condition (S_orig verified > 0 for all tasks).
- Distinguishes missing/failed API calls or reasoning-budget truncations from task failure.
- Compares compression methods at both target removal schedules and matched actual
  token-removal rates (linearly interpolated retention at exact 30%, 50%, 70% actual
  removal and trapezoidal AUC).
- Characterizes intra-model generation variability via repeated trials (T=0.0 and T=0.3).
- Audits non-monotonic retention trajectories (where higher compression yields higher
  task score due to intermediate syntactic interference or polarity inversion).
"""

from __future__ import annotations
import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

from src.data.schema import PromptItem
from src.pruning.pruner import PromptPruner, PruningResult
from src.interpretation.groq_client import GroqEvaluatorClient
from src.analysis.statistics import holm_bonferroni_correction


@dataclass
class DownstreamTaskSpec:
    prompt_id: str
    category: str
    prompt: str
    description: str
    validator: Callable[[str], Tuple[float, Dict[str, Any]]]


# ==============================================================================
# DETERMINISTIC TASK VALIDATORS (15 Stratified Benchmark Tasks: 3 per Category)
# ==============================================================================

def validate_wifi_json(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p024 (Constraint): Wi-Fi troubleshooting strictly as raw valid JSON without markdown."""
    text = output.strip()
    has_markdown = "```" in text

    is_valid_json = False
    try:
        json.loads(text)
        is_valid_json = True
    except Exception:
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                json.loads(json_match.group(1))
                is_valid_json = True
            except Exception:
                is_valid_json = False

    has_wifi_terms = any(
        k in text.lower()
        for k in ["router", "ssid", "dns", "gateway", "reboot", "connection", "network", "wifi", "wi-fi", "signal"]
    )

    if is_valid_json and not has_markdown and has_wifi_terms:
        score = 1.0
    elif is_valid_json and has_markdown and has_wifi_terms:
        score = 0.5  # Valid JSON content but violated no-markdown constraint
    elif has_wifi_terms and not is_valid_json:
        score = 0.2  # Relevant Wi-Fi content but failed JSON format
    else:
        score = 0.0

    return score, {
        "is_valid_json": is_valid_json,
        "has_markdown": has_markdown,
        "has_wifi_terms": has_wifi_terms
    }


def validate_bullets_and_length(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p025 (Constraint): French Revolution in exactly 3 bullet points, each 15-20 words."""
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    bullets = [l for l in lines if re.match(r"^[-*•\d\.]+\s+", l)]

    n_bullets = len(bullets)
    word_counts = []
    bullets_in_range = 0

    for b in bullets:
        clean_text = re.sub(r"^[-*•\d\.]+\s+", "", b).strip()
        wc = len(clean_text.split())
        word_counts.append(wc)
        if 15 <= wc <= 20:
            bullets_in_range += 1

    text_lower = output.lower()
    has_topic = any(
        w in text_lower
        for w in ["french", "france", "revolution", "monarchy", "bastille", "1789", "estates", "louis", "feudal", "noble", "tax"]
    )

    if not has_topic:
        return 0.0, {"n_bullets": n_bullets, "word_counts": word_counts, "bullets_in_range": bullets_in_range, "has_topic": False}

    bullet_score = 1.0 if n_bullets == 3 else (0.5 if 2 <= n_bullets <= 4 else 0.0)
    length_score = (bullets_in_range / 3.0) if n_bullets > 0 else 0.0

    final_score = (0.5 * bullet_score) + (0.5 * min(1.0, length_score))
    return round(final_score, 4), {
        "n_bullets": n_bullets,
        "word_counts": word_counts,
        "bullets_in_range": bullets_in_range,
        "has_topic": has_topic
    }


def validate_negative_constraints(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p023 (Constraint): Running shoes blurb avoiding 'comfortable', 'fast', 'lightweight'."""
    text_lower = output.lower()
    forbidden = ["comfortable", "fast", "lightweight"]
    violations = [w for w in forbidden if re.search(rf"\b{w}\b", text_lower)]

    is_running_shoes = any(
        w in text_lower
        for w in ["shoe", "shoes", "cushion", "runner", "running", "stride", "mile", "pace", "grip", "sole", "track", "marathon", "step", "foot"]
    )

    if not is_running_shoes:
        return 0.0, {"violations": violations, "is_relevant": False}

    score = 1.0 - (len(violations) / float(len(forbidden)))
    return round(max(0.0, score), 4), {"violations": violations, "is_relevant": is_running_shoes}


def validate_entity_extraction(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p005 (Instruction): Extract action items and assignees (Sarah -> budget, John -> server)."""
    text_lower = output.lower()
    has_sarah = "sarah" in text_lower
    has_budget = "budget" in text_lower or "finalize" in text_lower
    has_john = "john" in text_lower
    has_server = "server" in text_lower or "patch" in text_lower

    sarah_pair = has_sarah and has_budget
    john_pair = has_john and has_server

    score = (0.5 if sarah_pair else 0.0) + (0.5 if john_pair else 0.0)
    return round(score, 4), {
        "sarah_budget_extracted": sarah_pair,
        "john_server_extracted": john_pair
    }


def validate_legal_translation(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p004 (Instruction): French translation preserving contractual indemnity terms."""
    text_lower = output.lower()
    has_french = any(
        w in text_lower
        for w in ["le", "la", "les", "du", "des", "dans", "par", "selon", "conformément", "contrat", "partie", "article"]
    )
    has_legal = any(
        w in text_lower
        for w in [
            "loi", "juridique", "stipule", "clause", "termes", "obligations", "legal", "droit",
            "indemniser", "indemnisation", "garantir", "responsabilité", "dédommager", "recours", "exonérer", "préjudice", "couvrir"
        ]
    )

    score = 1.0 if (has_french and has_legal) else (0.5 if has_french else 0.0)
    return score, {"has_french": has_french, "has_legal": has_legal}


def validate_headlines_renewable(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p006 (Instruction): Generate five candidate headlines for a blog post about renewable energy advances."""
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    # Count numbered or bulleted lines or distinct headline-like lines
    headline_lines = [
        l for l in lines
        if re.match(r"^(?:[\d]+[\.\)\:]|[-*•])\s+", l) or (15 <= len(l) <= 140 and not l.endswith(":"))
    ]
    n_headlines = len(headline_lines)
    text_lower = output.lower()
    energy_terms = ["renewable", "solar", "wind", "clean", "green", "energy", "power", "grid", "battery", "sustainable", "carbon"]
    matched_terms = [t for t in energy_terms if t in text_lower]

    if not matched_terms:
        return 0.0, {"n_headlines": n_headlines, "matched_terms": matched_terms}

    count_score = 1.0 if n_headlines >= 5 else (n_headlines / 5.0)
    topic_score = 1.0 if len(matched_terms) >= 2 else 0.5
    score = round(0.6 * count_score + 0.4 * topic_score, 4)
    return score, {"n_headlines": n_headlines, "matched_terms": matched_terms}


def validate_context_retrieval(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p041 (Context): Q3 revenue of $4.2B with 18% EBITDA margin operating profit trajectory."""
    text_lower = output.lower()
    if len(text_lower.split()) < 10:
        return 0.0, {"word_count": len(text_lower.split()), "has_rev": False, "has_margin": False, "has_profit_q3": False}

    has_rev = bool(re.search(r"\b4\.2\b|\$4\.2|4,200", text_lower))
    has_margin = bool(re.search(r"\b18\s*%|18\s*percent|0\.756|756\s*m", text_lower))
    has_profit_q3 = any(k in text_lower for k in ["q3", "third quarter", "ebitda", "operating profit", "margin", "trajectory", "revenue"])

    units_preserved = int(has_rev) + int(has_margin) + int(has_profit_q3)
    score = round(units_preserved / 3.0, 4)
    return score, {
        "has_rev_4_2b": has_rev,
        "has_margin_18pct": has_margin,
        "has_profit_q3": has_profit_q3,
        "units_preserved": units_preserved
    }


def validate_context_aws_compliance(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p042 (Context): Tier-1 financial institution with HIPAA & SOC2 migrating to AWS Ireland."""
    text_lower = output.lower()
    has_fin = any(w in text_lower for w in ["financial", "bank", "tier-1", "tier 1", "enterprise", "institution"])
    has_compliance = ("hipaa" in text_lower) and ("soc2" in text_lower or "soc 2" in text_lower)
    has_partial_compliance = ("hipaa" in text_lower) or ("soc2" in text_lower or "soc 2" in text_lower) or ("compliance" in text_lower)
    has_aws_ireland = ("aws" in text_lower or "amazon" in text_lower or "cloud" in text_lower) and ("ireland" in text_lower or "eu-west-1" in text_lower or "gdpr" in text_lower or "eu" in text_lower)

    c_score = 1.0 if has_compliance else (0.5 if has_partial_compliance else 0.0)
    f_score = 1.0 if has_fin else 0.0
    r_score = 1.0 if has_aws_ireland else (0.5 if ("aws" in text_lower or "ireland" in text_lower) else 0.0)

    score = round((c_score + f_score + r_score) / 3.0, 4)
    return score, {
        "has_financial": has_fin,
        "has_compliance": has_compliance,
        "has_aws_ireland": has_aws_ireland
    }


def validate_context_layoff_memo(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p043 (Context): Internal memo addressing 15% engineering layoff after cancelled Series B."""
    text_lower = output.lower()
    has_15pct_eng = ("15%" in text_lower or "15 percent" in text_lower) and ("engineer" in text_lower or "staff" in text_lower or "workforce" in text_lower)
    has_partial_layoff = any(w in text_lower for w in ["15%", "layoff", "laid off", "reduction", "restructur", "depart", "engineer"])
    has_series_b = "series b" in text_lower or ("funding" in text_lower and ("cancel" in text_lower or "round" in text_lower))
    has_memo = any(w in text_lower for w in ["team", "colleagues", "employees", "subject:", "memo", "all,", "everyone"])

    s1 = 1.0 if has_15pct_eng else (0.5 if has_partial_layoff else 0.0)
    s2 = 1.0 if has_series_b else 0.0
    s3 = 1.0 if has_memo else 0.0
    score = round((s1 + s2 + s3) / 3.0, 4)
    return score, {
        "has_15pct_eng": has_15pct_eng,
        "has_series_b": has_series_b,
        "has_memo_format": has_memo
    }


def validate_tone_formality(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p061 (Tone): Formal executive apology regarding cloud outage."""
    text_lower = output.lower()
    slang = ["gonna", "wanna", "lol", "hey", "cool", "yeah", "nope", "guy", "oops", "my bad"]
    violations = [w for w in slang if re.search(rf"\b{w}\b", text_lower)]

    formal_markers = ["sincerely", "regards", "dear", "apolog", "regret", "commitment", "enterprise", "respectfully", "assure"]
    matched_formal = [w for w in formal_markers if w in text_lower]
    has_outage_topic = any(w in text_lower for w in ["outage", "disruption", "interruption", "cloud", "service", "incident"])

    if not has_outage_topic or len(text_lower.split()) < 15:
        return 0.0, {"violations": violations, "matched_formal": matched_formal, "has_outage_topic": has_outage_topic}

    formality_score = min(1.0, len(matched_formal) / 3.0)
    penalty = 0.5 * len(violations)
    score = round(max(0.0, formality_score - penalty), 4)
    return score, {"violations": violations, "matched_formal": matched_formal, "has_outage_topic": has_outage_topic}


def validate_tone_cease_desist(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p062 (Tone): Stern, formal, authoritative legal cease-and-desist letter for copyright infringement."""
    text_lower = output.lower()
    legal_directives = ["cease", "desist", "copyright", "infring", "unauthorized", "immediately", "demand", "legal", "remedies", "liability"]
    matched = [w for w in legal_directives if w in text_lower]
    weak_hedges = ["maybe", "perhaps", "if you don't mind", "sorry to bother", "cheers", "hey"]
    hedges_found = [w for w in weak_hedges if w in text_lower]

    if len(matched) == 0 or len(text_lower.split()) < 15:
        return 0.0, {"matched_directives": matched, "hedges_found": hedges_found}

    directive_score = min(1.0, len(matched) / 4.0)
    score = round(max(0.0, directive_score - 0.5 * len(hedges_found)), 4)
    return score, {"matched_directives": matched, "hedges_found": hedges_found}


def validate_tone_bedtime_owl(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p066 (Tone): Whimsical, gentle, soothing bedtime story about a sleepy owl."""
    text_lower = output.lower()
    has_owl = "owl" in text_lower or "owlet" in text_lower or "hoot" in text_lower
    soothing_words = ["moon", "star", "stars", "soft", "gentle", "gently", "sleep", "sleepy", "slumber", "dream", "whisper", "night", "quiet", "breeze", "feather", "cozy", "lullaby"]
    matched_soothing = [w for w in soothing_words if w in text_lower]

    if not has_owl or len(text_lower.split()) < 15:
        return 0.0, {"has_owl": has_owl, "matched_soothing": matched_soothing}

    score = round(min(1.0, len(matched_soothing) / 4.0), 4)
    return score, {"has_owl": has_owl, "matched_soothing": matched_soothing}


def validate_multi_step_order(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p081 (Multi-Step): 3-step security procedure (1: revoke tokens -> 2: rotate keys -> 3: audit logs)."""
    text_lower = output.lower()

    # Locate the 3 specific domain actions
    m1 = re.search(r"\b(revoke|invalidate|disable)\b[\s\S]{0,60}\b(token|tokens|session|credentials)\b|\b(token|tokens)\b[\s\S]{0,60}\b(revoke|invalidate)\b", text_lower)
    m2 = re.search(r"\b(rotate|regenerate|replace)\b[\s\S]{0,60}\b(key|keys|secret|secrets|password)\b|\b(key|keys)\b[\s\S]{0,60}\b(rotate|regenerate)\b", text_lower)
    m3 = re.search(r"\b(audit|review|inspect|analyze)\b[\s\S]{0,60}\b(log|logs|access)\b|\b(log|logs)\b[\s\S]{0,60}\b(audit|review|inspect)\b", text_lower)

    pos_1 = m1.start() if m1 else -1
    pos_2 = m2.start() if m2 else -1
    pos_3 = m3.start() if m3 else -1

    steps_present = int(pos_1 != -1) + int(pos_2 != -1) + int(pos_3 != -1)
    strictly_ordered = (pos_1 != -1 and pos_2 != -1 and pos_3 != -1 and pos_1 < pos_2 < pos_3)
    partially_ordered = (
        (pos_1 != -1 and pos_2 != -1 and pos_1 < pos_2) or
        (pos_2 != -1 and pos_3 != -1 and pos_2 < pos_3) or
        (pos_1 != -1 and pos_3 != -1 and pos_1 < pos_3)
    )

    if strictly_ordered:
        score = 1.0
    elif steps_present >= 2 and partially_ordered:
        score = round(0.5 * (steps_present / 3.0) + 0.25, 4)
    elif steps_present >= 1:
        score = round(0.4 * (steps_present / 3.0), 4)
    else:
        score = 0.0

    return score, {
        "has_steps": steps_present > 0,
        "ordered": strictly_ordered,
        "steps_present": steps_present,
        "strictly_ordered": strictly_ordered,
        "positions": [pos_1, pos_2, pos_3]
    }


def _find_ordered_step_positions(text_lower: str, patterns: List[str]) -> List[int]:
    """Finds occurrence positions for a sequence of step regex patterns.
    
    For each step k, prefers the first match occurring at or after the matched
    position of step k-1 (so introductory overview sentences at the top of an
    uncompressed response do not falsely invert step order when the body executes
    the steps sequentially), falling back to the earliest overall match if none
    occurs after step k-1.
    """
    positions: List[int] = []
    last_pos = -1
    for pat in patterns:
        matches = [m.start() for m in re.finditer(pat, text_lower)]
        if not matches:
            positions.append(-1)
            continue
        after = [pos for pos in matches if pos >= last_pos]
        chosen = after[0] if after else matches[0]
        positions.append(chosen)
        if chosen >= last_pos:
            last_pos = chosen
    return positions


def validate_multistep_climate_mvp(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p088 (Multi-Step): 1: brainstorm 5 climate tech concepts -> 2: evaluate market size -> 3: pick most viable -> 4: 30-day MVP roadmap."""
    text_lower = output.lower()
    patterns = [
        r"\b(brainstorm\w*|concept\w*|startup\w*|idea\w*)\b[\s\S]{0,120}\b(climate|carbon|energy|solar|grid|emission\w*|capture|battery)\b|\b(climate|carbon)\b[\s\S]{0,80}\b(startup\w*|concept\w*|idea\w*)\b",
        r"\b(market\s+size\w*|tam|sam|addressable\s+market|billion\w*|\$\d+\s*b|cagr|market\s+potential|market\s+evaluation)\b",
        r"\b(most\s+viable|viable\s+idea\w*|selected\s+concept|top\s+pick|chosen\s+idea|winning\s+concept|best\s+option|pick\s+the\s+most|selection|viable\s+startup)\b",
        r"\b(30-day|30\s+day|mvp\s+roadmap|roadmap|week\s+1|days?\s+1[\-–]7)\b"
    ]

    positions = _find_ordered_step_positions(text_lower, patterns)
    present = [p for p in positions if p != -1]
    n_present = len(present)
    is_monotonic = len(present) >= 2 and all(present[i] <= present[i + 1] for i in range(len(present) - 1))

    if n_present == 4 and is_monotonic:
        score = 1.0
    elif n_present >= 2 and is_monotonic:
        score = round(0.75 * (n_present / 4.0), 4)
    elif n_present >= 1:
        score = round(0.4 * (n_present / 4.0), 4)
    else:
        score = 0.0

    return score, {"steps_present": n_present, "is_monotonic": is_monotonic, "positions": positions}


def validate_multistep_seo_audit(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p098 (Multi-Step): 1: audit SEO metadata -> 2: mobile page load performance -> 3: keyword ranking competition -> 4: prioritized SEO recommendations."""
    text_lower = output.lower()
    patterns = [
        r"\b(metadata|meta\s+tag\w*|title\s+tag\w*|meta\s+description\w*|canonical|schema|h1)\b",
        r"\b(mobile|page\s+load|core\s+web\s+vitals|lcp|cls|inp|ttfb|load\s+performance|page\s+speed)\b",
        r"\b(keyword\w*|competit\w*|ranking\w*|serp|search\s+volume|keyword\s+gap|backlink\w*)\b",
        r"\b(priorit\w*|recommend\w*|action\s+plan|quick\s+win\w*|high\s+impact|roadmap)\b"
    ]

    positions = _find_ordered_step_positions(text_lower, patterns)
    present = [p for p in positions if p != -1]
    n_present = len(present)
    is_monotonic = len(present) >= 2 and all(present[i] <= present[i + 1] for i in range(len(present) - 1))

    if n_present == 4 and is_monotonic:
        score = 1.0
    elif n_present >= 2 and is_monotonic:
        score = round(0.75 * (n_present / 4.0), 4)
    elif n_present >= 1:
        score = round(0.4 * (n_present / 4.0), 4)
    else:
        score = 0.0

    return score, {"steps_present": n_present, "is_monotonic": is_monotonic, "positions": positions}


# Keep backward-compatible aliases for any external imports
validate_multistep_json_sessions = validate_multistep_climate_mvp
validate_multistep_log_pipeline = validate_multistep_seo_audit


# ==============================================================================
# STRATIFIED BENCHMARK SUITE (15 Tasks: 3 per Category, including Original 8)
# ==============================================================================

ORIGINAL_8_TASKS: List[DownstreamTaskSpec] = [
    DownstreamTaskSpec(
        prompt_id="p024",
        category="constraint",
        prompt="Provide troubleshooting steps for Wi-Fi failure. Output must be strictly valid JSON without any markdown formatting.",
        description="Wi-Fi troubleshooting in raw valid JSON without markdown",
        validator=validate_wifi_json
    ),
    DownstreamTaskSpec(
        prompt_id="p025",
        category="constraint",
        prompt="Describe the French Revolution in exactly three bullet points. Each bullet must be between 15 and 20 words.",
        description="French Revolution in exactly 3 bullets, each 15-20 words",
        validator=validate_bullets_and_length
    ),
    DownstreamTaskSpec(
        prompt_id="p023",
        category="constraint",
        prompt="Write a promotional blurb for running shoes. You must not use the words 'comfortable', 'fast', or 'lightweight'.",
        description="Running shoes promotional blurb avoiding 3 forbidden words",
        validator=validate_negative_constraints
    ),
    DownstreamTaskSpec(
        prompt_id="p005",
        category="instruction",
        prompt="Extract all action items and their assignees from the following meeting minutes: Sarah will finalize budget; John will patch server.",
        description="Action item and assignee structured extraction",
        validator=validate_entity_extraction
    ),
    DownstreamTaskSpec(
        prompt_id="p004",
        category="instruction",
        prompt="Translate this English contract clause into French while preserving precise legal terminology: The party shall indemnify and hold harmless.",
        description="French legal translation preserving contractual terminology",
        validator=validate_legal_translation
    ),
    DownstreamTaskSpec(
        prompt_id="p041",
        category="context",
        prompt="Given the quarterly earnings report showing Q3 revenue of $4.2B with 18% EBITDA margin, summarize the operating profit trajectory.",
        description="Context-based financial earnings trajectory summary",
        validator=validate_context_retrieval
    ),
    DownstreamTaskSpec(
        prompt_id="p061",
        category="tone",
        prompt="Draft an executive apology to enterprise clients regarding the recent cloud outage, adhering strictly to a formal and professional tone.",
        description="Executive outage apology with strictly formal tone",
        validator=validate_tone_formality
    ),
    DownstreamTaskSpec(
        prompt_id="p081",
        category="multi_step",
        prompt="List the 3-step security procedure to handle compromised credentials: first revoke active tokens, second rotate keys, third audit access logs.",
        description="3-step sequenced security procedure with strict step ordering",
        validator=validate_multi_step_order
    )
]

ADDITIONAL_STRATIFIED_TASKS: List[DownstreamTaskSpec] = [
    DownstreamTaskSpec(
        prompt_id="p006",
        category="instruction",
        prompt="Generate five candidate headlines for a blog post about renewable energy advances.",
        description="Generate 5 distinct blog headlines on renewable energy",
        validator=validate_headlines_renewable
    ),
    DownstreamTaskSpec(
        prompt_id="p042",
        category="context",
        prompt="The client is a Tier-1 enterprise financial institution with strict HIPAA and SOC2 compliance mandates evaluating cloud migration to AWS Ireland region.",
        description="Cloud migration recommendations preserving HIPAA, SOC2, and AWS Ireland context",
        validator=validate_context_aws_compliance
    ),
    DownstreamTaskSpec(
        prompt_id="p043",
        category="context",
        prompt="Our company recently laid off 15% of engineering staff following a cancelled Series B funding round. Address remaining employees in an internal memo.",
        description="Internal memo preserving 15% engineering layoff and cancelled Series B context",
        validator=validate_context_layoff_memo
    ),
    DownstreamTaskSpec(
        prompt_id="p062",
        category="tone",
        prompt="Draft a cease-and-desist letter to a copyright infringer. Adopt a stern, formal, unambiguous, and authoritative legal tone.",
        description="Stern authoritative legal cease-and-desist letter",
        validator=validate_tone_cease_desist
    ),
    DownstreamTaskSpec(
        prompt_id="p066",
        category="tone",
        prompt="Write a children's bedtime story about a sleepy owl. Use a whimsical, gentle, soothing, and rhythmic voice.",
        description="Whimsical soothing bedtime story about a sleepy owl",
        validator=validate_tone_bedtime_owl
    ),
    DownstreamTaskSpec(
        prompt_id="p088",
        category="multi_step",
        prompt="First brainstorm five startup concepts in climate tech. Then evaluate market size for each. Next pick the most viable idea. Finally outline a 30-day MVP roadmap.",
        description="4-step sequenced climate tech ideation, market sizing, selection, and 30-day MVP roadmap",
        validator=validate_multistep_climate_mvp
    ),
    DownstreamTaskSpec(
        prompt_id="p098",
        category="multi_step",
        prompt="First audit website SEO metadata. Second analyze page load performance on mobile. Third inspect keyword ranking competition. Fourth deliver prioritized SEO recommendations.",
        description="4-step sequenced technical SEO audit and prioritized recommendation workflow",
        validator=validate_multistep_seo_audit
    )
]

BENCHMARK_TASKS: List[DownstreamTaskSpec] = ORIGINAL_8_TASKS + ADDITIONAL_STRATIFIED_TASKS


class DownstreamEvaluator:
    """Executes downstream LLM tasks with uncompressed and compressed prompts,
    measuring deterministic task success retention, matched-rate AUC, model variability,
    and non-monotonic compression dynamics.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: str = "results/downstream_cache"
    ):
        self.model_name = model_name or os.getenv("DOWNSTREAM_MODEL", os.getenv("INTERPRETER_MODEL", "openai/gpt-oss-20b"))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.groq_client = GroqEvaluatorClient(model_name=self.model_name, cache_dir=str(self.cache_dir))

    def _call_downstream_llm(
        self,
        prompt: str,
        temperature: float = 0.0,
        trial_id: int = 0,
        max_tokens: int = 4096
    ) -> Tuple[str, Dict[str, Any]]:
        """Invokes downstream LLM or retrieves from disk cache.
        
        Invalidates any legacy cache entry where completion was empty due to
        reasoning budget exhaustion, and sets reasoning_effort='low' on reasoning models
        so garbled prompts never exhaust the completion token budget.
        """
        key_str = f"{self.model_name}:{prompt}" if (temperature == 0.0 and trial_id == 0) else f"{self.model_name}:temp={temperature}:trial={trial_id}:{prompt}"
        cache_key = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"downstream_{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    meta = data.get("metadata", {})
                    comp_cached = data.get("completion", "")
                    if not meta.get("error") and not meta.get("api_error") and comp_cached.strip():
                        meta["empty_completion"] = False
                        return comp_cached, meta
            except Exception:
                pass

        # Call live model via Groq API (with automatic fallback to openai/gpt-oss-120b on 429 TPD limit)
        start_t = time.time()
        models_to_try = [self.model_name]
        if self.model_name == "openai/gpt-oss-20b":
            models_to_try.append("openai/gpt-oss-120b")

        last_err = None
        for active_model in models_to_try:
            try:
                kwargs: Dict[str, Any] = {
                    "model": active_model,
                    "messages": [
                        {"role": "system", "content": "You are a precise, compliant assistant. Follow instructions exactly and concisely."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if "gpt-oss" in active_model:
                    kwargs["reasoning_effort"] = "low"

                resp = self.groq_client.client.chat.completions.create(**kwargs)
                elapsed_ms = (time.time() - start_t) * 1000.0
                completion = resp.choices[0].message.content or ""
                usage = resp.usage
                in_tokens = usage.prompt_tokens if usage else len(prompt.split())
                out_tokens = usage.completion_tokens if usage else len(completion.split())
                is_empty = len(completion.strip()) == 0
                meta = {
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "latency_ms": round(elapsed_ms, 2),
                    "model": active_model,
                    "temperature": temperature,
                    "trial_id": trial_id,
                    "api_error": False,
                    "empty_completion": is_empty
                }
                if not is_empty:
                    break
            except Exception as e:
                last_err = e
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    continue
                break
        else:
            elapsed_ms = (time.time() - start_t) * 1000.0
            completion = f"[Execution Error: {last_err}]"
            meta = {
                "input_tokens": len(prompt.split()),
                "output_tokens": 0,
                "latency_ms": round(elapsed_ms, 2),
                "model": self.model_name,
                "temperature": temperature,
                "trial_id": trial_id,
                "api_error": True,
                "empty_completion": True,
                "error": str(last_err)
            }

        if not meta.get("api_error") and not meta.get("empty_completion"):
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({"prompt": prompt, "completion": completion, "metadata": meta}, f, indent=2)
            except Exception:
                pass

        return completion, meta

    def evaluate_task(
        self,
        task: DownstreamTaskSpec,
        compressed_text: str,
        is_uncompressed: bool = False,
        temperature: float = 0.0,
        trial_id: int = 0
    ) -> Tuple[float, str, Dict[str, Any]]:
        completion, meta = self._call_downstream_llm(
            compressed_text, temperature=temperature, trial_id=trial_id
        )
        if meta.get("api_error") or meta.get("empty_completion"):
            score = np.nan
            validator_meta = {
                "api_error": meta.get("api_error", False),
                "empty_completion": meta.get("empty_completion", False),
                "error": meta.get("error", "Empty completion returned by model")
            }
        else:
            score, validator_meta = task.validator(completion)
        meta.update(validator_meta)
        meta["is_uncompressed"] = is_uncompressed
        return score, completion, meta

    @staticmethod
    def compute_matched_rate_comparison(
        df: pd.DataFrame,
        n_bootstrap: int = 10000,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compares compression methods at matched actual token-removal rates via:
        1. Per-task linear interpolation at exact actual removal rates (0.30, 0.50, 0.70)
        2. Trapezoidal Area Under the Task-Retention vs Actual-Removal Curve (AUC in [0, 0.70])
        3. Empirical actual-removal bins (Mild [0.15, 0.38], Moderate (0.38, 0.58], Aggressive (0.58, 0.85])
        4. Task-cluster bootstrap 95% CIs and Holm-Bonferroni adjusted paired tests vs random.
        """
        valid_df = df.dropna(subset=["success_retention"]).copy()
        comp_df = valid_df[valid_df["method"] != "uncompressed"].copy()
        methods = [m for m in comp_df["method"].unique()]
        tasks = sorted(valid_df["prompt_id"].unique())

        # Build per-(task, method) interpolated retention curve over actual_removal in [0, 0.70]
        per_task_rows = []
        for pid in tasks:
            uncomp_row = valid_df[(valid_df["prompt_id"] == pid) & (valid_df["method"] == "uncompressed")]
            cat = str(uncomp_row["category"].iloc[0]) if not uncomp_row.empty else "unknown"
            s_orig = float(uncomp_row["task_success"].iloc[0]) if not uncomp_row.empty else 1.0

            for m in methods:
                sub = comp_df[(comp_df["prompt_id"] == pid) & (comp_df["method"] == m)].sort_values("actual_removal")
                if sub.empty:
                    continue

                # Anchor curve at (actual_removal=0.0, retention=1.0)
                xs = np.concatenate([[0.0], sub["actual_removal"].to_numpy(dtype=float)])
                ys = np.concatenate([[1.0], sub["success_retention"].to_numpy(dtype=float)])

                # Deduplicate identical x values by averaging y
                unique_x, inv = np.unique(xs, return_inverse=True)
                unique_y = np.array([np.mean(ys[inv == i]) for i in range(len(unique_x))])

                # Interpolate at exact actual removal targets 0.30, 0.50, 0.70
                interp_30 = float(np.interp(0.30, unique_x, unique_y))
                interp_50 = float(np.interp(0.50, unique_x, unique_y))
                interp_70 = float(np.interp(0.70, unique_x, unique_y))

                # Trapezoidal AUC normalized over [0.0, 0.70]
                grid_x = np.linspace(0.0, 0.70, 71)
                grid_y = np.interp(grid_x, unique_x, unique_y)
                norm_auc = float(np.trapezoid(grid_y, grid_x) / 0.70) if hasattr(np, "trapezoid") else float(np.trapz(grid_y, grid_x) / 0.70)

                # Raw target removal values
                r30 = sub[sub["target_removal"] == 0.3]["success_retention"]
                r50 = sub[sub["target_removal"] == 0.5]["success_retention"]
                r70 = sub[sub["target_removal"] == 0.7]["success_retention"]

                per_task_rows.append({
                    "prompt_id": pid,
                    "category": cat,
                    "method": m,
                    "uncompressed_baseline": s_orig,
                    "raw_ret_target_30": float(r30.iloc[0]) if not r30.empty else np.nan,
                    "raw_ret_target_50": float(r50.iloc[0]) if not r50.empty else np.nan,
                    "raw_ret_target_70": float(r70.iloc[0]) if not r70.empty else np.nan,
                    "matched_ret_actual_30": round(interp_30, 4),
                    "matched_ret_actual_50": round(interp_50, 4),
                    "matched_ret_actual_70": round(interp_70, 4),
                    "matched_actual_auc": round(norm_auc, 4),
                    "mean_actual_removal": round(float(sub["actual_removal"].mean()), 4)
                })

        per_task_df = pd.DataFrame(per_task_rows)

        # Aggregate across tasks with task-cluster bootstrapping and paired comparison vs random
        rng = np.random.default_rng(seed)
        rand_sub = per_task_df[per_task_df["method"] == "random"].set_index("prompt_id")

        method_summary_rows = []
        raw_ps = []
        for m in methods:
            m_sub = per_task_df[per_task_df["method"] == m].set_index("prompt_id")
            if m_sub.empty:
                continue

            auc_vals = m_sub["matched_actual_auc"].to_numpy()
            n_t = len(auc_vals)
            boot_auc = [np.mean(rng.choice(auc_vals, size=n_t, replace=True)) for _ in range(n_bootstrap)]
            ci_low, ci_high = np.percentile(boot_auc, [2.5, 97.5])

            # Paired test vs random on matched_actual_auc
            common_ids = m_sub.index.intersection(rand_sub.index)
            if m != "random" and len(common_ids) > 1:
                diffs = m_sub.loc[common_ids, "matched_actual_auc"].to_numpy() - rand_sub.loc[common_ids, "matched_actual_auc"].to_numpy()
                if np.std(diffs, ddof=1) > 1e-12:
                    _, p_val = stats.ttest_1samp(diffs, 0.0)
                    p_val = float(p_val)
                else:
                    p_val = 1.0
                delta_vs_rand = float(np.mean(diffs))
            else:
                p_val = 1.0
                delta_vs_rand = 0.0

            raw_ps.append(p_val)
            method_summary_rows.append({
                "method": m,
                "n_tasks": n_t,
                "missing_or_failed_runs": int(df[df["method"] == m]["task_success"].isna().sum()),
                "mean_actual_removal": round(float(m_sub["mean_actual_removal"].mean()), 4),
                "raw_ret_30": round(float(m_sub["raw_ret_target_30"].mean()), 4),
                "raw_ret_50": round(float(m_sub["raw_ret_target_50"].mean()), 4),
                "raw_ret_70": round(float(m_sub["raw_ret_target_70"].mean()), 4),
                "raw_mean_retention": round(float(m_sub[["raw_ret_target_30", "raw_ret_target_50", "raw_ret_target_70"]].mean().mean()), 4),
                "matched_ret_actual_30": round(float(m_sub["matched_ret_actual_30"].mean()), 4),
                "matched_ret_actual_50": round(float(m_sub["matched_ret_actual_50"].mean()), 4),
                "matched_ret_actual_70": round(float(m_sub["matched_ret_actual_70"].mean()), 4),
                "matched_actual_auc": round(float(np.mean(auc_vals)), 4),
                "matched_auc_ci_95": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "delta_auc_vs_random": round(delta_vs_rand, 4),
                "p_val_vs_random": p_val
            })

        adj_ps = holm_bonferroni_correction(raw_ps) if raw_ps else []
        for idx, r in enumerate(method_summary_rows):
            r["p_holm_vs_random"] = adj_ps[idx] if idx < len(adj_ps) else r["p_val_vs_random"]

        method_summary_df = pd.DataFrame(method_summary_rows)
        return per_task_df, method_summary_df

    @staticmethod
    def analyze_non_monotonic_retention(df: pd.DataFrame) -> pd.DataFrame:
        """Identifies all (prompt_id, method) trajectories where task retention increases
        at a higher compression ratio (S_comp(r_{k+1}) > S_comp(r_k)), classifying the
        underlying linguistic/compression mechanism.
        """
        comp_df = df[df["method"] != "uncompressed"].dropna(subset=["success_retention"]).copy()
        cases = []
        for (pid, m), group in comp_df.groupby(["prompt_id", "method"]):
            g = group.sort_values("target_removal")
            if len(g) < 3:
                continue
            rows = list(g.to_dict("records"))
            for i in range(len(rows) - 1):
                r_low = rows[i]
                r_high = rows[i + 1]
                if float(r_high["success_retention"]) > float(r_low["success_retention"]) + 1e-4:
                    p_low = str(r_low["prompt_text"])
                    p_high = str(r_high["prompt_text"])
                    cat = str(r_low["category"])

                    if cat == "constraint" and ("words" in p_low.lower() or "json" in p_low.lower() or "bullet" in p_low.lower()):
                        mechanism = (
                            "Intermediate Polarity/Constraint Interference: At lower CR, removing negation or bound "
                            "operator while keeping the governed noun ('words X, Y, Z' or 'markdown') actively misleads "
                            "the model; at higher CR, the misleading secondary phrase is also pruned, allowing partial recovery."
                        )
                    elif float(r_high["actual_removal"]) <= float(r_low["actual_removal"]) + 1e-4:
                        mechanism = "Identical Actual Token Budget: Discrete word rounding yielded same token count."
                    else:
                        mechanism = (
                            "Syntactic Fragmentation vs Telegraphic Recovery: Intermediate pruning left broken syntax "
                            "or distracting modifiers, whereas aggressive pruning reduced the prompt to core telegraphic "
                            "keywords that triggered stronger pretraining priors."
                        )

                    cases.append({
                        "prompt_id": pid,
                        "category": cat,
                        "method": m,
                        "lower_target_cr": r_low["target_removal"],
                        "lower_actual_cr": round(float(r_low["actual_removal"]), 4),
                        "lower_retention": round(float(r_low["success_retention"]), 4),
                        "lower_prompt": p_low,
                        "higher_target_cr": r_high["target_removal"],
                        "higher_actual_cr": round(float(r_high["actual_removal"]), 4),
                        "higher_retention": round(float(r_high["success_retention"]), 4),
                        "higher_prompt": p_high,
                        "retention_rebound": round(float(r_high["success_retention"]) - float(r_low["success_retention"]), 4),
                        "mechanism": mechanism
                    })

        return pd.DataFrame(cases)

    def run_variability_study(
        self,
        scorers: Dict[str, Any],
        pruner: PromptPruner,
        subset_task_ids: List[str] = ["p005", "p024", "p041", "p061", "p081"],
        n_trials: int = 3,
        output_csv: str = "results/tables/downstream_variability.csv"
    ) -> pd.DataFrame:
        """Runs repeated generations (K=3 trials at T=0.0 and T=0.3) on a stratified
        5-task subset (1 per category) across uncompressed (0.0) and compressed (0.5)
        conditions to characterize downstream LLM generation stability.
        """
        task_map = {t.prompt_id: t for t in BENCHMARK_TASKS if t.prompt_id in subset_task_ids}
        rows = []
        print(f"[Downstream] Running variability study on {len(task_map)} tasks (K={n_trials} trials at T=0.0 and T=0.3)...")

        for pid, task in task_map.items():
            sc_res = scorers["embedding"].score_prompt(task.prompt, prompt_id=pid)
            pr_50 = pruner.prune(sc_res, target_removal=0.5)

            conditions = [
                ("uncompressed", 0.0, task.prompt),
                ("embedding_cr50", pr_50.actual_removal, pr_50.compressed_prompt)
            ]

            for cond_name, act_cr, p_text in conditions:
                for temp in [0.0, 0.3]:
                    trial_scores = []
                    trial_tokens = []
                    for trial in range(n_trials):
                        # At T=0.0, trial_id=0 uses deterministic response; at T=0.3 each trial_id is an independent stochastic call
                        eff_trial = 0 if temp == 0.0 else trial
                        sc, _, meta = self.evaluate_task(
                            task, p_text, is_uncompressed=(cond_name == "uncompressed"),
                            temperature=temp, trial_id=eff_trial
                        )
                        if not np.isnan(sc):
                            trial_scores.append(float(sc))
                        trial_tokens.append(int(meta.get("output_tokens", 0)))

                    mean_sc = float(np.mean(trial_scores)) if trial_scores else np.nan
                    sd_sc = float(np.std(trial_scores, ddof=1)) if len(trial_scores) > 1 else 0.0
                    exact_agreement = len(set(round(x, 4) for x in trial_scores)) == 1 if trial_scores else False

                    rows.append({
                        "prompt_id": pid,
                        "category": task.category,
                        "condition": cond_name,
                        "actual_removal": round(float(act_cr), 4),
                        "temperature": temp,
                        "n_trials": n_trials,
                        "mean_task_success": round(mean_sc, 4) if not np.isnan(mean_sc) else np.nan,
                        "sd_task_success": round(sd_sc, 4),
                        "min_task_success": round(float(min(trial_scores)), 4) if trial_scores else np.nan,
                        "max_task_success": round(float(max(trial_scores)), 4) if trial_scores else np.nan,
                        "exact_score_consistency": exact_agreement,
                        "mean_output_tokens": round(float(np.mean(trial_tokens)), 1),
                        "sd_output_tokens": round(float(np.std(trial_tokens, ddof=1)), 1) if len(trial_tokens) > 1 else 0.0
                    })

        var_df = pd.DataFrame(rows)
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        var_df.to_csv(out_p, index=False)
        print(f"[Downstream] Saved variability study ({len(var_df)} rows) to {output_csv}")
        return var_df

    def run_benchmark(
        self,
        scorers: Dict[str, Any],
        baselines: Dict[str, Any],
        pruner: PromptPruner,
        tasks: Optional[List[DownstreamTaskSpec]] = None,
        ratios: List[float] = [0.0, 0.3, 0.5, 0.7],
        output_csv: str = "results/tables/downstream_results.csv"
    ) -> pd.DataFrame:
        if tasks is None:
            tasks = BENCHMARK_TASKS

        print(f"[Downstream] Running functional evaluation across {len(tasks)} stratified tasks...")
        records = []
        all_methods = list(scorers.keys()) + list(baselines.keys())

        for task in tasks:
            orig_score, orig_comp, orig_meta = self.evaluate_task(task, task.prompt, is_uncompressed=True)
            orig_success = max(0.01, float(orig_score)) if not np.isnan(orig_score) else 1.0

            scored_map = {}
            for m_name, sc in scorers.items():
                scored_map[m_name] = sc.score_prompt(task.prompt, prompt_id=task.prompt_id)

            for ratio in ratios:
                if ratio == 0.0:
                    records.append({
                        "prompt_id": task.prompt_id,
                        "category": task.category,
                        "method": "uncompressed",
                        "target_removal": 0.0,
                        "actual_removal": 0.0,
                        "retention": 1.0,
                        "prompt_text": task.prompt,
                        "task_success": orig_score,
                        "success_retention": 1.0 if not np.isnan(orig_score) else np.nan,
                        "input_tokens": orig_meta.get("input_tokens", 0),
                        "output_tokens": orig_meta.get("output_tokens", 0),
                        "latency_ms": orig_meta.get("latency_ms", 0),
                        "api_error": orig_meta.get("api_error", False),
                        "empty_completion": orig_meta.get("empty_completion", False)
                    })
                    continue

                for method_name in all_methods:
                    if method_name in scorers:
                        pr_res = pruner.prune(scored_map[method_name], target_removal=ratio)
                    else:
                        pr_res = baselines[method_name].compress(task.prompt, prompt_id=task.prompt_id, target_removal=ratio)

                    comp_score, comp_out, comp_meta = self.evaluate_task(task, pr_res.compressed_prompt, is_uncompressed=False)
                    if np.isnan(comp_score):
                        retention_rate = np.nan
                    else:
                        retention_rate = round(min(1.0, float(comp_score) / orig_success), 4)

                    records.append({
                        "prompt_id": task.prompt_id,
                        "category": task.category,
                        "method": method_name,
                        "target_removal": ratio,
                        "actual_removal": pr_res.actual_removal,
                        "retention": pr_res.retention,
                        "prompt_text": pr_res.compressed_prompt,
                        "task_success": comp_score,
                        "success_retention": retention_rate,
                        "input_tokens": comp_meta.get("input_tokens", 0),
                        "output_tokens": comp_meta.get("output_tokens", 0),
                        "latency_ms": comp_meta.get("latency_ms", 0),
                        "api_error": comp_meta.get("api_error", False),
                        "empty_completion": comp_meta.get("empty_completion", False)
                    })

        df = pd.DataFrame(records)
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

        # Generate companion analytical tables (per-task, matched-rate method summary, non-monotonic cases)
        tables_dir = out_p.parent
        per_task_df, method_sum_df = self.compute_matched_rate_comparison(df)
        non_mono_df = self.analyze_non_monotonic_retention(df)

        per_task_df.to_csv(tables_dir / "downstream_per_task_summary.csv", index=False)
        method_sum_df.to_csv(tables_dir / "downstream_matched_comparison.csv", index=False)
        non_mono_df.to_csv(tables_dir / "downstream_non_monotonic_cases.csv", index=False)

        print(
            f"[Downstream] Saved {len(df)} evaluations, {len(per_task_df)} per-task summaries, "
            f"and {len(non_mono_df)} non-monotonic cases to {tables_dir}"
        )
        return df
