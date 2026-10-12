"""Prompt definitions, rubrics, and JSON schemas for LLM Interpretation and Evaluation.
Defines Protocol A (Blind Interpretability) and Protocol B (Reference-Guided Preservation).
"""

from __future__ import annotations
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ==============================================================================
# Legacy ReconstructedSemantics (Maintained for backward compatibility)
# ==============================================================================

INTERPRETER_SYSTEM_PROMPT = """You are evaluating whether a compressed prompt preserves its intended meaning. You have NOT seen the original prompt. Read the supplied prompt and reconstruct:
1. Intended task
2. Role
3. Constraints
4. Tone
5. Context
6. Required actions
7. Ordering/dependencies

Return JSON only. Do not assume information that is not present. If an aspect is not present or cannot be determined from the compressed prompt, leave its list empty or string empty."""

INTERPRETER_USER_TEMPLATE = """Compressed Prompt:
\"\"\"{compressed_prompt}\"\"\"

Reconstruct the semantic components into the following JSON format:
{{
  "task": "...",
  "role": "...",
  "constraints": ["..."],
  "tone": ["..."],
  "context": ["..."],
  "required_actions": ["..."],
  "ordering_dependencies": ["..."]
}}
Return raw valid JSON only."""


class ReconstructedSemantics(BaseModel):
    task: str = Field(default="", description="Reconstructed primary task/intent")
    role: str = Field(default="", description="Reconstructed role/persona")
    constraints: List[str] = Field(default_factory=list, description="Reconstructed constraints")
    tone: List[str] = Field(default_factory=list, description="Reconstructed tone attributes")
    context: List[str] = Field(default_factory=list, description="Reconstructed background context elements")
    required_actions: List[str] = Field(default_factory=list, description="Extracted actionable verbs/commands")
    ordering_dependencies: List[str] = Field(default_factory=list, description="Extracted ordered steps or dependencies")
    raw_response: Optional[str] = Field(default="", description="Original LLM response text")


# ==============================================================================
# Protocol A: Blind Standalone Interpretability
# Evaluator receives ONLY the compressed prompt text. Zero access to original prompt or ground truth.
# ==============================================================================

PROTOCOL_A_RUBRIC_VERSION = "1.0.0"

PROTOCOL_A_SYSTEM_PROMPT = """You are an expert evaluator assessing the STANDALONE INTERPRETABILITY of a prompt.
You have NOT seen any original version of this prompt.
Your task is to judge whether this text is understandable, actionable, coherent, and sufficiently specified for a human or AI agent to execute.
Do NOT guess or hallucinate details that are not present.

Evaluate according to this rubric (all scores 0.0 to 1.0):
1. "standalone_interpretability": Overall rating of how interpretable and complete this prompt is on its own.
2. "clarity": Linguistic fluency, grammatical coherence, and lack of syntactic confusion.
3. "actionability": Whether a reader knows exactly what to do without critical missing parameters.
4. "inferred_task": What primary task or goal is stated in the prompt.
5. "detected_ambiguities": Specific words, omissions, or instructions that are confusing or underspecified.
6. "reconstructed_components": Extracted components present in the prompt (task, role, constraints, tone, context, steps).

You MUST respond strictly with a valid JSON object matching the requested schema. No surrounding markdown formatting."""

PROTOCOL_A_USER_TEMPLATE = """Evaluate the standalone interpretability of this prompt:

\"\"\"{compressed_prompt}\"\"\"

Return a valid JSON object matching:
{{
  "standalone_interpretability": 0.0,
  "clarity": 0.0,
  "actionability": 0.0,
  "inferred_task": "...",
  "detected_ambiguities": ["..."],
  "reconstructed_components": {{
    "task": "...",
    "role": "...",
    "constraints": ["..."],
    "tone": ["..."],
    "context": ["..."],
    "steps": ["..."]
  }}
}}"""


class BlindInterpretabilityResult(BaseModel):
    """Protocol A result: Evaluates standalone interpretability without ground truth."""
    standalone_interpretability: float = Field(..., ge=0.0, le=1.0, description="Overall standalone interpretability score (0-1)")
    clarity: float = Field(..., ge=0.0, le=1.0, description="Linguistic clarity and coherence (0-1)")
    actionability: float = Field(..., ge=0.0, le=1.0, description="Whether the prompt is actionable (0-1)")
    inferred_task: str = Field(default="", description="Task inferred solely from compressed text")
    detected_ambiguities: List[str] = Field(default_factory=list, description="Ambiguities or confusing omissions")
    reconstructed_components: Dict[str, Any] = Field(default_factory=dict, description="Extracted semantic components")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution and provenance metadata")


# ==============================================================================
# Protocol B: Reference-Guided Semantic Preservation
# Evaluator receives structured reference criteria (intent, constraints, context, tone, steps)
# and compares against the compressed prompt.
# ==============================================================================

PROTOCOL_B_RUBRIC_VERSION = "1.0.0"

PROTOCOL_B_SYSTEM_PROMPT = """You are an expert evaluator assessing SEMANTIC PRESERVATION in a compressed prompt against structured reference requirements.
You are provided with:
1. Structured Reference Criteria (the required intent, constraints, context, tone, and steps).
2. The Compressed Prompt text.

Your job is to determine which reference criteria remain satisfied, which are ambiguous, and which are lost or contradicted.
Use this scoring scale (0.0 to 1.0):
- 1.0: Requirement is fully and clearly preserved.
- 0.5: Requirement is partially preserved, vague, or ambiguous.
- 0.0: Requirement is missing, broken, or contradicted.

Assess the following components (each 0.0 to 1.0):
- "intent_preservation": Is the core intended action/goal preserved?
- "constraint_preservation": Are negative/positive constraints and formatting rules preserved? (If no constraints exist in reference, assign 1.0 unless false constraints were introduced).
- "context_preservation": Are key background entities/facts preserved? (If none in reference, assign 1.0).
- "tone_preservation": Is the required style, tone, or demeanor preserved? (If none in reference, assign 1.0).
- "dependency_preservation": Are procedural step orderings and dependencies preserved? (If none in reference, assign 1.0).
- "composite_preservation": Unweighted arithmetic mean of the five component scores.
- "preserved_requirements": List of specific reference requirements verified present.
- "lost_requirements": List of reference requirements omitted or contradicted.
- "ambiguities": List of partially preserved or vague items.
- "evidence_from_compressed": Short quoted phrases from the compressed prompt proving preserved items.
- "reasons_for_lost": Explanation of why lost items failed.

You MUST respond strictly with a valid JSON object matching the requested schema. No surrounding markdown formatting."""

PROTOCOL_B_USER_TEMPLATE = """Structured Reference Criteria:
- Intent: {intent}
- Constraints: {constraints}
- Context: {context}
- Tone: {tone}
- Steps / Dependencies: {steps}

Compressed Prompt:
\"\"\"{compressed_prompt}\"\"\"

Return a valid JSON object matching:
{{
  "intent_preservation": 0.0,
  "constraint_preservation": 0.0,
  "context_preservation": 0.0,
  "tone_preservation": 0.0,
  "dependency_preservation": 0.0,
  "composite_preservation": 0.0,
  "preserved_requirements": ["..."],
  "lost_requirements": ["..."],
  "ambiguities": ["..."],
  "evidence_from_compressed": ["..."],
  "reasons_for_lost": ["..."]
}}"""


class ReferenceGuidedPreservationResult(BaseModel):
    """Protocol B result: Evaluates semantic preservation against structured reference criteria."""
    intent_preservation: float = Field(..., ge=0.0, le=1.0, description="Score for intent preservation (0-1)")
    constraint_preservation: float = Field(..., ge=0.0, le=1.0, description="Score for constraint preservation (0-1)")
    context_preservation: float = Field(..., ge=0.0, le=1.0, description="Score for context preservation (0-1)")
    tone_preservation: float = Field(..., ge=0.0, le=1.0, description="Score for tone preservation (0-1)")
    dependency_preservation: float = Field(..., ge=0.0, le=1.0, description="Score for dependency preservation (0-1)")
    composite_preservation: float = Field(..., ge=0.0, le=1.0, description="Composite preservation score (0-1)")
    preserved_requirements: List[str] = Field(default_factory=list, description="Requirements confirmed preserved")
    lost_requirements: List[str] = Field(default_factory=list, description="Requirements lost or contradicted")
    ambiguities: List[str] = Field(default_factory=list, description="Requirements rendered ambiguous")
    evidence_from_compressed: List[str] = Field(default_factory=list, description="Phrases from text supporting preservation")
    reasons_for_lost: List[str] = Field(default_factory=list, description="Explanations for failed requirements")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution and provenance metadata")


def format_protocol_a_prompt(compressed_prompt: str) -> str:
    """Format prompt for Protocol A (Blind Standalone Interpretability)."""
    return PROTOCOL_A_USER_TEMPLATE.format(compressed_prompt=compressed_prompt.strip())


def format_protocol_b_prompt(
    compressed_prompt: str,
    reference_intent: str,
    reference_constraints: Optional[List[str]] = None,
    reference_context: Optional[List[str]] = None,
    reference_tone: Optional[List[str]] = None,
    reference_steps: Optional[List[str]] = None,
    reference_entities: Optional[List[str]] = None
) -> str:
    """Format prompt for Protocol B (Reference-Guided Semantic Preservation)."""
    return PROTOCOL_B_USER_TEMPLATE.format(
        compressed_prompt=compressed_prompt.strip(),
        intent=reference_intent or "None specified",
        constraints=json.dumps(reference_constraints or []),
        context=json.dumps((reference_context or []) + (reference_entities or [])),
        tone=json.dumps(reference_tone or []),
        steps=json.dumps(reference_steps or [])
    )

