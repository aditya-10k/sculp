"""Experiment configuration loader and dataclass definitions."""

from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ExperimentConfig:
    num_prompts: int = 100
    prompts_path: str = "data/processed/prompts.json"
    compression_ratios: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    scorers: List[str] = field(default_factory=lambda: ["random", "embedding", "gradient", "integrated_gradients"])
    baselines: List[str] = field(default_factory=lambda: ["llmlingua", "llmlingua2"])
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    attribution_model: str = "bert-base-uncased"
    interpreter_provider: str = "mock"
    interpreter_model: str = "gemini-1.5-flash"
    protect_negation: bool = False
    bootstrap_samples: int = 10000
    seed: int = 42
    results_dir: str = "results"

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "configs/default.yaml") -> ExperimentConfig:
        p = Path(yaml_path)
        if not p.exists():
            return cls()

        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        return cls(
            num_prompts=cfg.get("dataset", {}).get("num_prompts", 100),
            prompts_path=cfg.get("dataset", {}).get("processed_path", "data/processed/prompts.json"),
            compression_ratios=cfg.get("compression_ratios", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            scorers=cfg.get("scorers", ["random", "embedding", "gradient", "integrated_gradients"]),
            baselines=cfg.get("baselines", ["llmlingua", "llmlingua2"]),
            embedding_model=cfg.get("embedding_model", {}).get("name", "sentence-transformers/all-MiniLM-L6-v2"),
            attribution_model=cfg.get("attribution_model", {}).get("name", "bert-base-uncased"),
            interpreter_provider=cfg.get("interpreter", {}).get("provider", "mock"),
            interpreter_model=cfg.get("interpreter", {}).get("model", "gemini-1.5-flash"),
            protect_negation=cfg.get("pruning", {}).get("protect_negations", False),
            bootstrap_samples=cfg.get("analysis", {}).get("bootstrap_samples", 10000),
            seed=cfg.get("seed", 42)
        )
