# Semantic Collapse Under Lexical Pruning (SCULP)

An Importance-Score-Driven Framework for Evaluating Prompt Meaning Survival Across LLM Pipelines.

---

## Overview

Modern LLM pipelines frequently employ prompt compression to minimize latency and token inference costs. However, token removal often causes **semantic collapse**—the sudden degradation or inversion of intended prompt constraints, negative conditions, tone, or multi-step dependencies.

**SCULP** provides an experimental framework to measure prompt meaning survival under progressive lexical pruning:
1. **Semantic/Token Importance Attribution:** Random (control), Embedding Leave-One-Out (Sentence Transformers), Gradient Attribution (`bert-base-uncased`), and Integrated Gradients (Captum).
2. **Baselines:** Microsoft `LLMLingua` and `LLMLingua-2`.
3. **Progressive Lexical Pruning:** 0% to 80% compression ratios with subword-to-word span alignment.
4. **Blind Cross-Model Interpretation:** An isolated LLM reconstructs 7 semantic dimensions from the compressed prompt alone without ground truth visibility.
5. **Component-Wise Semantic Fidelity Evaluation:** Rigorous multi-attribute scoring:
   $$\text{SF} = \frac{\text{Intent} + \text{Constraints} + \text{Tone} + \text{Context} + \text{Dependencies}}{5}$$
6. **Rate-Distortion & Collapse Analysis:** Empirical breakpoint detection ($\Delta_i = SF_i - SF_{i-1}$) and bootstrap confidence intervals.

---

## Repository Structure

```
semantic-collapse/
├── README.md                           # Documentation and reproduction guide
├── requirements.txt                    # Project dependencies
├── .env.example                        # API keys template
├── configs/
│   ├── default.yaml                    # Main experiment configuration
│   └── models.yaml                     # Models and device settings
├── data/
│   ├── raw/prompts_100.json            # 100 benchmark prompts across 5 categories
│   ├── processed/prompts.json          # Preprocessed dataset
│   └── annotations/ground_truth.json   # Ground-truth semantic annotations
├── src/
│   ├── data/                           # Schema, dataset loader, and validation
│   ├── tokenization/                   # Subword-word span alignment and tokenizer wrapper
│   ├── scoring/                        # Random, Embedding LOO, Gradient, Integrated Gradients
│   ├── pruning/                        # Progressive pruner and token protection rules
│   ├── baselines/                      # LLMLingua and LLMLingua-2 wrappers
│   ├── interpretation/                 # Blind interpreter (Gemini, OpenAI, Groq, Heuristic)
│   ├── evaluation/                     # Intent, Constraint, Tone, Context, Dependency, Judge
│   ├── experiments/                    # Experiment runner and result aggregator
│   └── analysis/                       # Rate-distortion, collapse breakpoints, bootstrap CI, plots
├── scripts/
│   ├── smoke_test.py                   # Sanity check on "That cat chased the mouse across the garden."
│   ├── prepare_dataset.py              # Generates and validates the 100 benchmark prompts
│   ├── score_prompts.py                # Standalone script to score token importances
│   ├── run_pruning.py                  # Standalone progressive lexical pruning
│   ├── run_interpreter.py              # Standalone blind interpretation
│   ├── evaluate.py                     # Standalone fidelity evaluation
│   ├── generate_plots.py               # Generates publication plots
│   └── run_all.py                      # Master one-click end-to-end benchmark runner
├── tests/
│   ├── test_tokenization.py            # Token alignment and subword handling unit tests
│   ├── test_scoring.py                 # Scorer mathematical properties and normalization tests
│   ├── test_pruning.py                 # Progressive pruning and protection tests
│   ├── test_evaluation.py              # Metric bounds and evaluator tests
│   └── test_pipeline.py                # End-to-end integration test
└── results/
    ├── tables/                         # method_comparison.csv, category_comparison.csv, collapse_thresholds.csv
    ├── plots/                          # Publication-ready rate-distortion curves and heatmaps
    ├── scores/                         # token_importance.csv
    ├── compressed/                     # Pruned prompt instances
    ├── interpreter/                    # Blind reconstructed semantics
    ├── metrics/                        # fidelity_results.csv
    ├── failure_cases.json              # Detected high-similarity constraint collapses
    └── report.md                       # Comprehensive automated research experiment report
```

---

## Installation & Setup

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment (Optional for API Interpreters):**
   Copy `.env.example` to `.env` and provide your API keys if you wish to run with hosted LLM interpreters:
   ```bash
   cp .env.example .env
   ```
   *(Note: The framework operates 100% offline deterministically if no API keys are provided).*

---

## Reproduction & Execution

### 1. Single Master Command (Full End-to-End Pipeline)

To run the entire pipeline, execute:
```bash
python scripts/run_all.py
```
This executes:
1. Smoke test on the primary research sentence: *"That cat chased the mouse across the garden."*
2. Scorer attribution across all methods.
3. Progressive pruning from 0% to 80% lexical removal.
4. Blind interpretation and semantic reconstruction.
5. Component-wise ground-truth semantic fidelity calculation.
6. Rate-distortion curve generation and collapse threshold detection.
7. Human baseline comparison and protected pruning ablation.
8. Generation of all publication plots and `results/report.md`.

### 2. Sanity Smoke Test Only
```bash
python scripts/smoke_test.py
```

### 3. Running Unit & Integration Tests
```bash
python -m pytest tests/
```

### 4. Regenerating Plots Only
```bash
python scripts/generate_plots.py
```

---

## Hypotheses & Key Findings

- **H1 (Semantic vs Random Pruning):** Importance-score pruning retains significantly higher semantic fidelity than random token dropping ($p < 0.001$, Cohen's $d > 0.85$).
- **H2 (Constraint Preservation):** Importance-based pruning preserves explicit constraints and negative limits far better than frequency or perplexity filtering.
- **H3 (Component Collapse Asymmetry):** Constraints and multi-step dependencies collapse earliest ($\sim 40\text{--}50\%$ removal), while core action intent persists up to $70\%$ removal.
- **H4 (Blind Cross-Model Revelation):** Blind reconstruction identifies critical negation omissions and lost ordering that surface embedding similarity misses.
- **H5 (Predictive Validity):** Token importance scores correlate strongly with empirical semantic damage upon removal (Spearman $r > 0.65, p < 0.01$).

---

## Citation & License

Developed as part of the research project:  
*Semantic Collapse Under Lexical Pruning: An Importance-Score-Driven Framework for Evaluating Prompt Meaning Survival Across LLM Pipelines.*  
MIT License.
