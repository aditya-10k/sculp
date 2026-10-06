"""Data schema definitions for the Semantic Collapse evaluation framework."""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class GroundTruth(BaseModel):
    """Ground truth semantic components of a prompt."""
    intent: str = Field(..., description="Core intended task / action of the prompt")
    role: Optional[str] = Field("", description="Assigned role or persona (e.g., 'technical support specialist')")
    constraints: List[str] = Field(default_factory=list, description="Negative or positive constraints (length, formatting, forbidden words, etc.)")
    tone: List[str] = Field(default_factory=list, description="Tone attributes (empathetic, concise, professional, etc.)")
    context: List[str] = Field(default_factory=list, description="Key background factual context units")
    steps: List[str] = Field(default_factory=list, description="Ordered procedural steps or dependencies")


class PromptItem(BaseModel):
    """Single prompt item in the dataset."""
    id: str = Field(..., description="Unique prompt identifier (e.g. 'p001')")
    category: str = Field(..., description="One of: 'instruction', 'constraint', 'context', 'tone', 'multi_step'")
    prompt: str = Field(..., description="The full prompt text")
    ground_truth: GroundTruth = Field(..., description="Annotated semantic ground truth")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional extra metadata")


class PromptDataset(BaseModel):
    """Collection of prompts."""
    version: str = "1.0.0"
    description: str = "Semantic Collapse benchmark prompts"
    prompts: List[PromptItem] = Field(default_factory=list)
