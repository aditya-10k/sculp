"""Prompt definitions and JSON schema for the Blind LLM Interpreter."""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

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
