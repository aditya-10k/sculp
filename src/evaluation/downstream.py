"""Functional Downstream Task-Success Engine.

Evaluates whether compressed prompts still accomplish their intended tasks when executed
by a downstream LLM, using deterministic validators (JSON validity, regex constraints,
forbidden word absence, bullet count and length bounds, entity matching, and sequential ordering).

Calculates Task Success Retention relative to uncompressed baseline:
    Retention = S_comp / S_orig
"""

from __future__ import annotations
import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import pandas as pd

from src.data.schema import PromptItem
from src.pruning.pruner import PromptPruner, PruningResult
from src.interpretation.groq_client import GroqEvaluatorClient


@dataclass
class DownstreamTaskSpec:
    prompt_id: str
    category: str
    prompt: str
    description: str
    validator: Callable[[str], Tuple[float, Dict[str, Any]]]


def validate_wifi_json(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p024: Wi-Fi troubleshooting strictly as raw valid JSON without markdown."""
    text = output.strip()
    has_markdown = "```" in text
    
    # Try parsing pure text
    is_valid_json = False
    parsed = None
    try:
        parsed = json.loads(text)
        is_valid_json = True
    except Exception:
        # Check if wrapped in markdown
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                is_valid_json = True
            except Exception:
                is_valid_json = False

    # Wi-Fi relevance check
    has_wifi_terms = any(k in text.lower() for k in ["router", "ssid", "dns", "gateway", "reboot", "connection", "network"])
    
    score = 0.0
    if is_valid_json and not has_markdown and has_wifi_terms:
        score = 1.0
    elif is_valid_json and has_markdown and has_wifi_terms:
        score = 0.5  # Valid JSON but violated no-markdown constraint
    elif has_wifi_terms and not is_valid_json:
        score = 0.2  # Useful troubleshooting but failed JSON format entirely
    else:
        score = 0.0

    return score, {
        "is_valid_json": is_valid_json,
        "has_markdown": has_markdown,
        "has_wifi_terms": has_wifi_terms
    }


def validate_bullets_and_length(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p025: Exactly 3 bullet points, each between 15 and 20 words."""
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    bullets = [l for l in lines if re.match(r"^[-*•\d\.]+\s+", l)]
    
    n_bullets = len(bullets)
    word_counts = []
    bullets_in_range = 0
    
    for b in bullets:
        # Strip bullet marker
        clean_text = re.sub(r"^[-*•\d\.]+\s+", "", b).strip()
        wc = len(clean_text.split())
        word_counts.append(wc)
        if 15 <= wc <= 20:
            bullets_in_range += 1

    # Scoring
    bullet_score = 1.0 if n_bullets == 3 else (0.5 if 2 <= n_bullets <= 4 else 0.0)
    length_score = (bullets_in_range / 3.0) if n_bullets > 0 else 0.0
    
    final_score = (0.5 * bullet_score) + (0.5 * length_score)
    return round(final_score, 4), {
        "n_bullets": n_bullets,
        "word_counts": word_counts,
        "bullets_in_range": bullets_in_range
    }


def validate_negative_constraints(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p023: Ad copy without 'comfortable', 'fast', 'lightweight'."""
    text_lower = output.lower()
    forbidden = ["comfortable", "fast", "lightweight"]
    violations = [w for w in forbidden if re.search(rf"\b{w}\b", text_lower)]
    
    is_running_shoes = any(w in text_lower for w in ["shoe", "shoes", "cushion", "runner", "stride", "mile", "pace", "grip"])
    
    if not is_running_shoes:
        return 0.0, {"violations": violations, "is_relevant": False}
        
    score = 1.0 - (len(violations) / float(len(forbidden)))
    return max(0.0, score), {"violations": violations, "is_relevant": is_running_shoes}


def validate_entity_extraction(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p005: Extract action items and assignees."""
    text_lower = output.lower()
    has_actions = any(w in text_lower for w in ["action item", "task", "todo", "action", "assigned to", "assignee"])
    has_entities = bool(re.search(r"\b(sarah|john|alex|lisa|david|michael|engineer|lead|manager)\b", text_lower)) or ":" in output
    
    score = 1.0 if (has_actions and has_entities) else (0.5 if (has_actions or has_entities) else 0.0)
    return score, {"has_actions": has_actions, "has_entities": has_entities}


def validate_legal_translation(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p004: French translation preserving legal terms."""
    text_lower = output.lower()
    # Check for basic French function words and legal terms
    has_french = any(w in text_lower for w in ["le", "la", "les", "du", "des", "dans", "par", "selon", "conformément", "contrat", "partie", "article"])
    has_legal = any(w in text_lower for w in [
        "loi", "juridique", "stipule", "clause", "termes", "obligations", "legal", "droit",
        "indemniser", "garantir", "responsabilité", "dédommager", "recours", "exonérer", "indemnisation"
    ])
    
    score = 1.0 if (has_french and has_legal) else (0.5 if has_french else 0.0)
    return score, {"has_french": has_french, "has_legal": has_legal}


def validate_multi_step_order(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p081/p082: Sequenced procedural instructions."""
    text_lower = output.lower()
    has_steps = bool(re.search(r"(?:\bstep\s*[1-3]|\b(?:first|second|third|then)\b|^\s*[1-3][\.\)]|\b[1-3][\.\)]\s+)", text_lower, re.MULTILINE))
    
    # Check sequential numbering 1 -> 2 -> 3 or sequence words
    pos_1 = text_lower.find("1") if "1" in text_lower else text_lower.find("first")
    pos_2 = text_lower.find("2") if "2" in text_lower else text_lower.find("second")
    pos_3 = text_lower.find("3") if "3" in text_lower else text_lower.find("third")
    ordered = (pos_1 != -1 and pos_2 != -1 and pos_1 < pos_2)
    if pos_3 != -1 and ordered:
        ordered = (pos_2 < pos_3)
        
    score = 1.0 if (has_steps and ordered) else (0.5 if has_steps else 0.0)
    return score, {"has_steps": has_steps, "ordered": ordered}


def validate_tone_formality(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p061: Formal executive tone."""
    text_lower = output.lower()
    slang = ["gonna", "wanna", "lol", "hey", "cool", "yeah", "nope", "guy"]
    violations = [w for w in slang if re.search(rf"\b{w}\b", text_lower)]
    
    formal_markers = ["sincerely", "regards", "dear", "furthermore", "consequently", "pleased to", "hereby", "executive"]
    has_formal = any(w in text_lower for w in formal_markers) or len(violations) == 0
    
    score = 1.0 if (len(violations) == 0 and has_formal) else (0.5 if len(violations) == 0 else 0.0)
    return score, {"violations": violations, "has_formal": has_formal}


def validate_context_retrieval(output: str) -> Tuple[float, Dict[str, Any]]:
    """Validator for p041: Contextual answer synthesis."""
    text_lower = output.lower()
    has_substance = len(text_lower.split()) >= 10
    score = 1.0 if has_substance else 0.2
    return score, {"word_count": len(text_lower.split())}


# Stratified Benchmark Suite for Downstream Evaluation
BENCHMARK_TASKS: List[DownstreamTaskSpec] = [
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


class DownstreamEvaluator:
    """Executes downstream LLM tasks with uncompressed and compressed prompts,
    measuring deterministic task success retention and performance metrics.
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

    def _call_downstream_llm(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Invokes downstream LLM or retrieves from disk cache."""
        cache_key = hashlib.sha256(f"{self.model_name}:{prompt}".encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"downstream_{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    meta = data.get("metadata", {})
                    if not meta.get("error") and not meta.get("api_error"):
                        return data["completion"], meta
            except Exception:
                pass

        # Call live model via Groq API
        start_t = time.time()
        try:
            resp = self.groq_client.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise, compliant assistant. Follow instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=600
            )
            elapsed_ms = (time.time() - start_t) * 1000.0
            completion = resp.choices[0].message.content or ""
            usage = resp.usage
            in_tokens = usage.prompt_tokens if usage else len(prompt.split())
            out_tokens = usage.completion_tokens if usage else len(completion.split())
            meta = {
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "latency_ms": round(elapsed_ms, 2),
                "model": self.model_name,
                "api_error": False
            }
        except Exception as e:
            # Fallback for offline / quota exhaustion
            elapsed_ms = (time.time() - start_t) * 1000.0
            completion = f"[Execution Error: {e}]"
            meta = {
                "input_tokens": len(prompt.split()),
                "output_tokens": 0,
                "latency_ms": round(elapsed_ms, 2),
                "model": self.model_name,
                "api_error": True,
                "error": str(e)
            }

        # Cache response
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
        is_uncompressed: bool = False
    ) -> Tuple[float, str, Dict[str, Any]]:
        completion, meta = self._call_downstream_llm(compressed_text)
        if meta.get("api_error"):
            score = 0.0
            validator_meta = {"api_error": True, "error": meta.get("error")}
        else:
            score, validator_meta = task.validator(completion)
        meta.update(validator_meta)
        meta["is_uncompressed"] = is_uncompressed
        return score, completion, meta

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

        print(f"[Downstream] Running functional evaluation across {len(tasks)} tasks...")
        records = []
        all_methods = list(scorers.keys()) + list(baselines.keys())

        for task in tasks:
            # 1. Run uncompressed baseline
            orig_score, orig_comp, orig_meta = self.evaluate_task(task, task.prompt, is_uncompressed=True)
            orig_success = max(0.01, orig_score)  # avoid division by zero
            
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
                        "success_retention": 1.0,
                        "input_tokens": orig_meta.get("input_tokens", 0),
                        "output_tokens": orig_meta.get("output_tokens", 0),
                        "latency_ms": orig_meta.get("latency_ms", 0)
                    })
                    continue

                for method_name in all_methods:
                    if method_name in scorers:
                        pr_res = pruner.prune(scored_map[method_name], target_removal=ratio)
                    else:
                        pr_res = baselines[method_name].compress(task.prompt, prompt_id=task.prompt_id, target_removal=ratio)

                    comp_score, comp_out, comp_meta = self.evaluate_task(task, pr_res.compressed_prompt, is_uncompressed=False)
                    retention_rate = comp_score / orig_success

                    records.append({
                        "prompt_id": task.prompt_id,
                        "category": task.category,
                        "method": method_name,
                        "target_removal": ratio,
                        "actual_removal": pr_res.actual_removal,
                        "retention": pr_res.retention,
                        "prompt_text": pr_res.compressed_prompt,
                        "task_success": comp_score,
                        "success_retention": round(min(1.0, retention_rate), 4),
                        "input_tokens": comp_meta.get("input_tokens", 0),
                        "output_tokens": comp_meta.get("output_tokens", 0),
                        "latency_ms": comp_meta.get("latency_ms", 0)
                    })

        df = pd.DataFrame(records)
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"[Downstream] Saved {len(df)} downstream task evaluations to {output_csv}")
        return df
