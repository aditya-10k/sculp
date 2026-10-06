"""Validation utilities for prompt datasets."""

from typing import List, Dict
from src.data.schema import PromptItem, PromptDataset

EXPECTED_CATEGORIES = {"instruction", "constraint", "context", "tone", "multi_step"}


def validate_prompt_item(item: PromptItem) -> List[str]:
    """Validate an individual PromptItem. Returns list of errors (empty if valid)."""
    errors = []
    if not item.id or not isinstance(item.id, str):
        errors.append("Invalid or missing 'id'")
    if item.category not in EXPECTED_CATEGORIES:
        errors.append(f"Invalid category '{item.category}'. Must be one of {sorted(EXPECTED_CATEGORIES)}")
    if not item.prompt or not item.prompt.strip():
        errors.append("Empty prompt text")
    if not item.ground_truth.intent or not item.ground_truth.intent.strip():
        errors.append("Empty ground_truth.intent")

    # Category-specific structural checks
    if item.category == "constraint" and len(item.ground_truth.constraints) == 0:
        errors.append("Constraint-heavy prompt must have at least one ground-truth constraint")
    if item.category == "context" and len(item.ground_truth.context) == 0:
        errors.append("Context-heavy prompt must have at least one ground-truth context element")
    if item.category == "tone" and len(item.ground_truth.tone) == 0:
        errors.append("Tone-focused prompt must have at least one ground-truth tone element")
    if item.category == "multi_step" and len(item.ground_truth.steps) < 2:
        errors.append("Multi-step prompt must have at least two ordered ground-truth steps")

    return errors


def validate_dataset(dataset: PromptDataset, expected_total: int = 100) -> Dict[str, Any]:
    """Validate full dataset integrity and category balance."""
    errors = []
    seen_ids = set()
    category_counts: Dict[str, int] = {cat: 0 for cat in EXPECTED_CATEGORIES}

    for i, item in enumerate(dataset.prompts):
        if item.id in seen_ids:
            errors.append(f"Duplicate id '{item.id}' at index {i}")
        seen_ids.add(item.id)

        item_errors = validate_prompt_item(item)
        for err in item_errors:
            errors.append(f"Prompt {item.id} (index {i}): {err}")

        if item.category in category_counts:
            category_counts[item.category] += 1

    total = len(dataset.prompts)
    if total != expected_total:
        errors.append(f"Expected {expected_total} prompts, got {total}")

    expected_per_cat = expected_total // len(EXPECTED_CATEGORIES)
    for cat, count in category_counts.items():
        if count != expected_per_cat:
            errors.append(f"Category '{cat}' count mismatch: expected {expected_per_cat}, found {count}")

    return {
        "valid": len(errors) == 0,
        "total": total,
        "category_counts": category_counts,
        "errors": errors
    }
