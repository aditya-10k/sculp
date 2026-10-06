"""Dataset management: loading, saving, and querying prompts."""

import json
from pathlib import Path
from typing import List, Optional, Dict
from src.data.schema import PromptItem, PromptDataset
from src.data.validation import validate_dataset


class DatasetManager:
    """Manages loading and querying benchmark prompts."""

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = Path(filepath) if filepath else None
        self.dataset: Optional[PromptDataset] = None
        if self.filepath and self.filepath.exists():
            self.load()

    def load(self, filepath: Optional[str] = None) -> PromptDataset:
        path = Path(filepath) if filepath else self.filepath
        if not path or not path.exists():
            raise FileNotFoundError(f"Dataset file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.dataset = PromptDataset(**data)
        self.filepath = path
        return self.dataset

    def save(self, filepath: Optional[str] = None) -> None:
        path = Path(filepath) if filepath else self.filepath
        if not path:
            raise ValueError("No filepath specified to save dataset")
        if not self.dataset:
            raise ValueError("No dataset loaded to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.dataset.model_dump(), f, indent=2, ensure_ascii=False)

    def get_by_id(self, prompt_id: str) -> Optional[PromptItem]:
        if not self.dataset:
            return None
        for item in self.dataset.prompts:
            if item.id == prompt_id:
                return item
        return None

    def get_by_category(self, category: str) -> List[PromptItem]:
        if not self.dataset:
            return []
        return [p for p in self.dataset.prompts if p.category == category]

    def all_prompts(self) -> List[PromptItem]:
        return self.dataset.prompts if self.dataset else []

    def validate(self, expected_total: int = 100) -> Dict:
        if not self.dataset:
            return {"valid": False, "errors": ["No dataset loaded"]}
        return validate_dataset(self.dataset, expected_total=expected_total)
