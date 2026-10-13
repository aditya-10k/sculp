# Semantic Collapse Under Lexical Pruning (SCULP)

**An Importance-Score-Driven Framework for Evaluating Prompt Meaning Survival Across LLM Pipelines**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 24 Passing](https://img.shields.io/badge/Tests-24%20Passed-brightgreen.svg)](tests/)

---

## Table of Contents

- [Overview & Research Motivation](#overview--research-motivation)
- [System Architecture](#system-architecture)
- [Key Empirical Findings](#key-empirical-findings)
  - [1. Rate-Distortion & Method Performance (5,400 Runs)](#1-rate-distortion--method-performance-5400-runs)
  - [2. Component-Wise Collapse Breakpoints](#2-component-wise-collapse-breakpoints)
  - [3. Attribution Faithfulness & Direct Paired Validation](#3-attribution-faithfulness--direct-paired-validation)
  - [4. Functional Downstream Task Retention (15 Stratified Tasks)](#4-functional-downstream-task-retention-15-stratified-tasks)
  - [5. Semantic Stress Testing & Audited Polarity Inversions](#5-semantic-stress-testing--audited-polarity-inversions)
  - [6. Rule-Based Critical-Span Pruning Ablation (126 Paired Runs)](#6-rule-based-critical-span-pruning-ablation-126-paired-runs)
  - [7. Blind Standalone vs Reference-Guided Protocols](#7-blind-standalone-vs-reference-guided-protocols)
- [Evidence Chain & Validation Matrix](#evidence-chain--validation-matrix)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [Methodological Guardrails & Transparency](#methodological-guardrails--transparency)
- [Literature Grounding](#literature-grounding)
- [License & Citation](#license--citation)

---

## Overview & Research Motivation

Modern Large Language Model (LLM) pipelines frequently employ prompt compression algorithms (such as LLMLingua, selective token pruning, and attention-budgeting techniques) to reduce context length, minimize inference latency, and curb token costs. 

However, lexical token removal frequently induces **semantic collapse**—the sudden degradation or catastrophic inversion of prompt semantics before surface embedding metrics indicate failure. Key failure modes include:
- **Polarity Inversions:** Dropping negative operators (`"not"`, `"without"`, `"never"`) transforms prohibitions into imperative instructions while maintaining deceptively high sentence-embedding similarity ($> 0.85$).
- **Constraint Dissolution:** Removing numerical boundaries (`"at most 50 words"`, `"under $100"`) leaves unconstrained generations.
- **Asymmetric Component Collapse:** Constraints and sequential procedural dependencies collapse at low compression ratios (10–30% removal), whereas general intent survives past 60–80% removal.

**SCULP** provides a mathematically grounded, empirically audited evaluation framework designed to quantify prompt meaning survival, identify exact collapse thresholds, test attribution faithfulness across circular, proxy, and encoder-free targets, and measure downstream functional execution under progressive compression.

---

## System Architecture

```text
ORIGINAL PROMPT
       │
       ▼
┌──────────────┐
│ Prompt Parser │
└──────┬───────┘
       │
       ▼ Tokenization & Subword Alignment
┌────────────────────────────────────────────────────────┐
│               Token Importance Estimators               │
├──────────────────┬──────────────────┬──────────────────┤
│  Embedding LOO   │     Gradient     │    Integrated    │
│  (MiniLM-L6-v2)  │    Projection    │    Gradients     │
│   (Sentence-BERT)│ (bert-base-unc.) │ (Captum n=25)    │
└──────────────────┴──────────────────┴──────────────────┘
       │
       ▼ Word-Level Aggregation & Ranking
┌────────────────────────────────────────────────────────┐
│             Progressive Lexical Pruning                │
│    (Ratios: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8) │
├───────────────────────────────┬────────────────────────┤
│   Independent Word Pruner     │ Critical-Span Pruner   │
└───────────────────────────────┴────────────────────────┘
       │
       ▼ Compressed Prompts
┌────────────────────────────────────────────────────────┐
│               Multi-Faceted Evaluation                 │
├───────────────────────────────┬────────────────────────┤
│   Blind LLM Interpretation    │  Downstream Execution  │
│  (Protocol A & Protocol B)    │  (15 Stratified Tasks  │
│  (Groq: GPT-OSS-20B, Qwen-27B)│   Deterministic Tests) │
├───────────────────────────────┴────────────────────────┤
│       Component-Wise Ground-Truth Semantic Fidelity    │
│       SF = (Intent + Constraints + Tone + Context +    │
│             Dependencies) / 5                          │
└────────────────────────────────────────────────────────┘
       │
       ▼
Rate-Distortion Curves · Collapse Thresholds · Faithfulness Validation
```

---

## Key Empirical Findings

### 1. Rate-Distortion & Method Performance (5,400 Runs)

Evaluated across all 100 balanced prompts (20 per category across Instruction, Constraint, Context, Tone, and Multi-Step) at 9 compression ratios ($N = 5,400$ total evaluations):

| Method | Mean Semantic Fidelity (SF) | Mean Embedding Sim | Delta vs Random | Cohen's d vs Random | Paired Wilcoxon $p$-value |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`llmlingua2` (Fallback)** | **0.4735** | 0.8510 | +0.0316 | 0.2604 | $< 10^{-15}$ |
| **`llmlingua` (Fallback)** | 0.4714 | 0.8422 | +0.0295 | 0.2404 | $< 10^{-15}$ |
| **`gradient`** | 0.4705 | 0.8646 | +0.0287 | 0.2340 | $< 10^{-15}$ |
| **`embedding`** | 0.4685 | **0.9147** | +0.0266 | 0.2211 | $< 10^{-15}$ |
| **`integrated_gradients`** | 0.4605 | 0.8341 | +0.0186 | 0.1502 | $< 10^{-15}$ |
| **`random` (Control)** | 0.4419 | 0.7393 | 0.0000 | — | Baseline Control |

*Finding:* All importance-guided pruning methods significantly outperform deterministic random pruning ($p < 10^{-15}$, cluster bootstrap 95% CIs strictly exclude zero).

### 2. Component-Wise Collapse Breakpoints

The empirical collapse threshold $r^*$ is identified via the discrete rate-distortion gradient $\Delta_i = \frac{SF_i - SF_{i-1}}{0.1}$ locating the steepest marginal fidelity loss:

- **Constraint Category:** Collapses earliest at **10%–20% lexical removal** (drop magnitude 0.031).
- **Multi-Step Category:** Collapses at **10%–30% lexical removal** (ordering markers and step indices pruned).
- **Context Category:** Robust against compression; preserves core factual entities up to **70%–80% removal**.
- **Tone Category:** Preserves stylistic orientation up to **70%–80% removal**.

### 3. Attribution Faithfulness & Direct Paired Validation

Token importance correlations are computed **within individual prompts first** ($N = 30$ stratified prompts, 6 per category, 2,588 tokens), then aggregated across prompt clusters via **$B = 10,000$ cluster bootstrap resamples** and **Holm-Bonferroni** step-down multiple-testing correction across four distinct outcomes:
1. **Sentence-BERT LOO Drop (Circular Self-Consistency Check):** For Embedding LOO, $r_s = 1.0000$ by mathematical definition.
2. **Reference-Guided Embedding Proxy Drop:** Sentence-BERT cosine drop relative to human reference text; explicitly labeled as an embedding proxy because it uses the `all-MiniLM-L6-v2` encoder.
3. **Independent Deterministic Requirement Drop (Encoder-Free):** 100% deterministic lexical/stem, numeric, and negation-operator requirement drop against `ground_truth.json` with zero neural encoders.
4. **Downstream Task-Success Loss:** Functional task loss ($S_{\text{orig}} - S_{\text{comp}}$) across the 15-task benchmark.

| Method | Target Outcome | $N$ Clusters | Mean Spearman $r_s$ (SD) | Median $r_s$ | 95% Cluster Bootstrap CI | Kendall $\tau$ | Holm-Adjusted $p$ | Circular? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Baseline** | Reference-Guided Embedding Proxy Drop | 30 | -0.0004 (0.2096) | 0.0007 | `[-0.0763, 0.0720]` | -0.0068 | 0.9910 | No |
| **Embedding LOO** | Sentence-BERT LOO Drop | 30 | **1.0000** (0.0001) | 1.0000 | `[0.9999, 1.0000]` | **0.9997** | $< 10^{-50}$ | **YES** |
| **Embedding LOO** | Reference-Guided Embedding Proxy Drop | 30 | **0.4938** (0.2340) | 0.4983 | `[0.4077, 0.5718]` | 0.3843 | $< 10^{-11}$ | No |
| **Gradient Attribution** | Reference-Guided Embedding Proxy Drop | 30 | **0.4109** (0.2352) | 0.4340 | `[0.3251, 0.4922]` | 0.3082 | $< 10^{-9}$ | No |
| **Integrated Gradients** | Reference-Guided Embedding Proxy Drop | 30 | 0.2465 (0.2400) | 0.2297 | `[0.1605, 0.3326]` | 0.1862 | $9.0 \times 10^{-6}$ | No |
| **Random Baseline** | **Independent Deterministic Requirement Drop** | 30 | 0.0651 (0.2369) | 0.0215 | `[-0.0175, 0.1495]` | 0.0531 | 0.1432 (n.s.) | No |
| **Embedding LOO** | **Independent Deterministic Requirement Drop** | 30 | **0.5593** (0.2059) | 0.5754 | `[0.4839, 0.6302]` | **0.4371** | $< 10^{-14}$ | **No** |
| **Gradient Attribution** | **Independent Deterministic Requirement Drop** | 30 | **0.5533** (0.1371) | **0.5788** | `[0.5041, 0.5998]` | **0.4243** | $< 10^{-18}$ | **No** |
| **Integrated Gradients** | **Independent Deterministic Requirement Drop** | 30 | **0.3174** (0.2442) | 0.3099 | `[0.2311, 0.4023]` | 0.2403 | $< 10^{-7}$ | **No** |
| **Gradient Attribution** | Downstream Task-Success Loss | 15 | **0.4095** (0.2700) | 0.4091 | `[0.2753, 0.5332]` | 0.3347 | 0.0001 | No |
| **Embedding LOO** | Downstream Task-Success Loss | 15 | 0.3735 (0.2412) | 0.3993 | `[0.2492, 0.4829]` | 0.3040 | 0.0001 | No |

*Direct Paired Test (`Gradient` vs `Embedding LOO`):* On encoder-free **Independent Deterministic Requirement Drop**, direct paired cluster testing proves that **Gradient Attribution ($r_s = 0.5533$) and Embedding LOO ($r_s = 0.5593$) perform virtually identically ($\Delta r_s = -0.0061$, 95% CI `[-0.0855, +0.0802]`, paired $t$-test $p = 0.8890$, Wilcoxon $p = 0.5940$, $p_{\text{Holm}} = 0.8890$)**, while both significantly outperform Integrated Gradients ($p_{\text{Holm}} \le 0.00095$) and Random ($p_{\text{Holm}} < 10^{-7}$).

### 4. Functional Downstream Task Retention (15 Stratified Tasks)

Evaluates whether compressed prompts accomplish their downstream functional objectives across **15 stratified tasks** (3 per category across Instruction, Constraint, Context, Tone, and Multi-Step; $N = 285$ evaluations, **0 missing/failed runs**, uncompressed baseline $S_{\text{orig}} = 0.9778$):

| Method | Raw Mean Retention | Matched Ret @ 30% Actual | Matched Ret @ 50% Actual | Matched Ret @ 70% Actual | Matched Actual AUC | 95% Cluster Bootstrap CI | $\Delta$ AUC vs Random |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Embedding LOO** | **0.7243** | **0.8723** | 0.6784 | **0.6059** | **0.8019** | `[0.7169, 0.8825]` | **+0.0882** |
| **LLMLingua-2 (Fallback)** | 0.7134 | 0.8320 | 0.7060 | 0.5272 | 0.7898 | `[0.6941, 0.8786]` | +0.0761 |
| **LLMLingua (Fallback)** | 0.7179 | 0.8069 | **0.7191** | 0.5661 | 0.7863 | `[0.6650, 0.8909]` | +0.0726 |
| **Gradient Attribution** | 0.6642 | 0.7663 | 0.6189 | 0.5639 | 0.7436 | `[0.6081, 0.8682]` | +0.0299 |
| **Random Baseline** | 0.5857 | 0.7845 | 0.6508 | 0.2863 | 0.7137 | `[0.6143, 0.8040]` | 0.0000 |
| **Integrated Gradients** | 0.5827 | 0.6827 | 0.5559 | 0.4172 | 0.6726 | `[0.5459, 0.7886]` | -0.0411 |

*Key Findings:*
- **Aggressive Compression Resilience:** At 70% actual token removal, Embedding LOO retains **60.59%** of downstream task success vs **28.63%** for Random Baseline ($>2.1\times$ higher retention).
- **Intra-Model Generation Variability ($K=3$ trials at $T=0.0$ and $T=0.3$, 60 runs):** 100% of condition cells (20/20) exhibit $\text{SD} = 0.0000$ within-temperature score variance, confirming that non-monotonic retention jumps (21 audited cases in `downstream_non_monotonic_cases.csv`) are caused by intermediate polarity interference or syntactic fragmentation rather than decoding noise.

### 5. Semantic Stress Testing & Audited Polarity Inversions

Evaluated across 14 minimal pairs (7 dimensions) at 30%, 50%, and 70% compression (252 total evaluations, with all cross-clause false-positive classifications fixed):
- **Preserved:** 72 / 252 (28.6%)
- **Syntactically/Semantically Ambiguous:** 109 / 252 (43.3%)
- **Audited Catastrophic Polarity Reversals:** 45 / 252 (17.9%, audited in `results/tables/polarity_reversals_audit.csv`)
- **Critically Removed:** 26 / 252 (10.3%)

*Failure Case Spotlight:*
> Original: *"Summarize this news article in exactly five sentences without using technical jargon or mentioning the author."*  
> Compressed: *"Summarize the article in exactly five sentences. technical terminology. not mention author' name"*  
> Sentence-BERT Cosine: `0.9426` (Deceptively high) | Constraint F1: `0.0000` (Collapsed)  
> *Mechanism:* Dropping the negative operator `"without"` inverted the constraint into an imperative to *use* technical jargon.

### 6. Rule-Based Critical-Span Pruning Ablation (126 Paired Runs)

To prevent orphaned verb and numeric-bound inversions, `DependencyAwarePruner` implements **rule-based critical-span grouping** (`negation_clause`, `numerical_bound`, `conditional_exception`, `priority_directive`) and is evaluated against ordinary word pruning across **all 3 importance scorers (`embedding`, `gradient`, `integrated_gradients`) $\times$ 14 minimal pairs $\times$ 3 CRs = 126 paired evaluations** at matched actual token budgets:

| Scorer Scope | Pruning Mode | $N$ Runs | Mean Actual Removal | Polarity Reversals (95% Cluster CI) | Preserved Rate (95% Cluster CI) |
|:---|:---|:---:|:---:|:---:|:---:|
| **All 3 Scorers Pooled** | Ordinary Word Pruning | 126 | 0.4446 | **21 / 126 (16.7% `[7.1%, 27.0%]`)** | 35 / 126 (27.8% `[15.1%, 42.9%]`) |
| **All 3 Scorers Pooled** | **Rule-Based Span Pruning** | 126 | 0.4332 | **0 / 126 (0.0% `[0.0%, 0.0%]`)** | **74 / 126 (58.7% `[45.2%, 72.2%]`)** |
| `embedding` | Ordinary vs Span Pruning | 42 | 0.464 vs 0.460 | **6 (14.3%) $\rightarrow$ 0 (0.0%)** | **13 (31.0%) $\rightarrow$ 26 (61.9%)** |
| `gradient` | Ordinary vs Span Pruning | 42 | 0.436 vs 0.420 | **8 (19.0%) $\rightarrow$ 0 (0.0%)** | **11 (26.2%) $\rightarrow$ 23 (54.8%)** |
| `integrated_gradients` | Ordinary vs Span Pruning | 42 | 0.435 vs 0.419 | **7 (16.7%) $\rightarrow$ 0 (0.0%)** | **11 (26.2%) $\rightarrow$ 25 (59.5%)** |

### 7. Blind Standalone vs Reference-Guided Protocols

Tested with live Groq completions across two independent model architectures (`openai/gpt-oss-20b` and `qwen/qwen3.8-27b`):
- **Protocol A (Blind Standalone Interpretability):** Evaluator receives ONLY compressed text: Mean Score = **0.1950**.
- **Protocol B (Reference-Guided Preservation):** Evaluator receives reference criteria + compressed text: Mean Score = **0.7567**.
- **Inter-Model Agreement:** Protocol A ($r_s = 0.8494$, MAE = 0.1200); Protocol B ($r_s = 0.7741$, MAE = 0.1140). High inter-evaluator reliability confirms reproducible evaluation across model architectures.

---

## Evidence Chain & Validation Matrix

Every headline claim is traced directly to source code, underlying data, saved CSVs, and statistical tests:

| # | Scientific Claim | Source Code | Underlying Data | Saved Results | Sample Size | Status |
|:---|:---|:---|:---|:---|:---|:---:|
| 1 | Importance pruning preserves more SF than random | `src/pruning/pruner.py` | `prompts.json` (100 prompts) | `fidelity_results.csv` | 5,400 runs | **Validated Finding** |
| 2 | Component collapse occurs at distinct thresholds | `src/analysis/collapse.py` | `fidelity_results.csv` | `collapse_thresholds.csv` | 100 prompts | **Validated Finding** |
| 3 | Embedding LOO correlation with Sentence-BERT is $r_s = 1.0$ | `src/analysis/attribution_faithfulness.py` | 30 stratified prompts | `attribution_faithfulness_summary.csv` | 30 prompts, 2,588 tokens | **Self-Consistency Check** |
| 4 | Reference-guided Sentence-BERT drop is an embedding proxy | `src/analysis/attribution_faithfulness.py` | `ground_truth.json` | `attribution_faithfulness_summary.csv` | 30 prompts, 2,588 tokens | **Reference-Guided Embedding Proxy** |
| 5 | Gradient & Embedding LOO equivalently predict encoder-free requirement loss | `src/analysis/attribution_faithfulness.py` | `ground_truth.json` | `attribution_faithfulness_summary.csv`, `attribution_pairwise_tests.csv` | 30 prompts, 2,588 tokens | **Validated Finding** |
| 6 | Protocol A (Blind) and Protocol B (Guided) measure distinct constructs | `src/interpretation/evaluator_protocols.py` | 10 pilot prompts | `pilot_evaluation_results.csv` | 30 runs | **Validated Finding** |
| 7 | Inter-evaluator agreement between independent LLMs is strong | `src/interpretation/evaluator_protocols.py` | 10 paired prompts | `pilot_evaluator_agreement.csv` | 10 paired prompts | **Validated Finding** |
| 8 | Lexical pruning causes polarity inversions and ambiguity | `src/evaluation/stress_tests.py` | 14 minimal pairs | `stress_test_results.csv`, `polarity_reversals_audit.csv` | 252 evaluations | **Validated Finding** |
| 9 | Rule-based critical-span pruning eliminates polarity inversions at matched budgets | `src/pruning/dependency_pruner.py` | 14 minimal pairs × 3 scorers | `dependency_pruning_summary.csv`, `dependency_pruning_reversals_audit.csv` | 126 paired evaluations | **Validated Finding** |
| 10 | Functional downstream task retention across 15 stratified tasks | `src/evaluation/downstream.py` | 15 stratified tasks | `downstream_results.csv`, `downstream_matched_comparison.csv`, `downstream_variability.csv` | 285 main + 60 variability runs | **Validated Finding** |
| 11 | LLMLingua / LLMLingua-2 approximate official algorithms | `src/baselines/` | GPT-2 / Multilingual BERT | `fidelity_results.csv` | 1,800 runs | **Preliminary Finding** |

---

## Repository Structure

```text
sculp/
├── README.md                           # Master documentation, findings & reproduction guide
├── AUDIT.md                            # Complete 25-point scientific and implementation audit
├── LITERATURE.md                       # Literature review and positioning
├── requirements.txt                    # Python dependencies
├── .env.example                        # Optional API credentials template
├── configs/
│   ├── default.yaml                    # Master experiment configuration
│   └── models.yaml                     # Model definitions and device parameters
├── data/
│   ├── raw/prompts_100.json            # 100 raw multi-component prompts
│   ├── processed/prompts.json          # Preprocessed balanced benchmark dataset
│   ├── processed/stress_test_prompts.json # 14 minimal-pair stress test prompts
│   └── annotations/ground_truth.json   # Comprehensive human ground-truth annotations
├── src/
│   ├── data/                           # Schema, Pydantic data models, loader
│   ├── tokenization/                   # Subword WordPiece to character span alignment
│   ├── scoring/                        # Random, Embedding LOO, Gradient, Integrated Gradients
│   ├── pruning/                        # Progressive pruner, token protection, dependency span pruner
│   ├── baselines/                      # LLMLingua & LLMLingua-2 wrappers with fallbacks
│   ├── interpretation/                 # Groq client, blind interpreter, evaluator protocols
│   ├── evaluation/                     # Multi-attribute fidelity, downstream tasks, stress tests
│   ├── experiments/                    # Experiment runner and result aggregator
│   └── analysis/                       # Collapse detection, bootstrap statistics, publication plots
├── scripts/
│   ├── run_all.py                      # Master end-to-end benchmark pipeline (5,400 runs)
│   ├── run_research_enhancements.py    # Faithfulness, downstream tasks, and stress test orchestrator
│   ├── pilot_groq.py                   # Live Groq API evaluator protocol runner
│   ├── smoke_test.py                   # Sanity smoke test on research target sentence
│   └── generate_plots.py               # Publication plot regeneration script
├── tests/                              # Comprehensive test suite (24/24 pytest passing)
│   ├── test_enhanced_research.py       # Faithfulness, downstream validators, span pruner tests
│   ├── test_evaluation.py              # Fidelity metric bounds and component scoring tests
│   ├── test_pipeline.py                # End-to-end integration test
│   ├── test_pruning.py                 # Progressive pruning and ratio tracking tests
│   ├── test_scoring.py                 # Mathematical properties, non-zero gradient flow tests
│   └── test_tokenization.py            # WordPiece subword aggregation tests
└── results/
    ├── archive_v1/                     # Archived v1 CSVs and report_v1.md for audit provenance
    ├── tables/                         # All analytical CSV result tables (faithfulness, downstream, stress, span)
    ├── plots/                          # 12 publication-quality figures
    ├── pilot/                          # Live Groq protocol comparison and model agreement data
    ├── metrics/fidelity_results.csv    # Primary 5,400-evaluation database
    ├── failure_cases.json              # 501 detected semantic collapse failure cases
    └── report.md                       # Comprehensive automated research experiment report
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aditya-10k/sculp.git
   cd sculp
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Configure Environment (Optional for Live Groq API):**
   ```bash
   cp .env.example .env
   ```
   Add your `GROQ_API_KEY` to `.env` if you want to execute live cross-model evaluations or downstream completions.  
   *(Note: The core scoring, progressive pruning, component evaluation, cached downstream analysis, and unit tests run 100% offline deterministically without API keys).*

---

## Quickstart & Reproduction

### 1. Run Unit Tests (100% Green, 24 Tests)
```bash
python -m pytest tests/
```

### 2. Run Single Target Sentence Smoke Test
```bash
python scripts/smoke_test.py
```
Evaluates attribution and pruning on: *"That cat chased the mouse across the garden."*

### 3. Run Research Enhancements (Faithfulness, 15-Task Downstream, Stress & Span Pruning)
```bash
python scripts/run_research_enhancements.py
```
Computes 4-outcome attribution faithfulness correlations and paired method tests on 30 stratified prompts, runs 15-task downstream task retention and variability evaluations, executes 252 minimal-pair stress tests and 126 matched-budget critical-span pruning evaluations, and regenerates all publication figures.

### 4. Run Full 100-Prompt Benchmark (5,400 Evaluations)
```bash
python scripts/run_all.py
```

### 5. Regenerate Publication Plots
```bash
python scripts/generate_plots.py
```
All 12 figures are saved to `results/plots/`.

---

## Methodological Guardrails & Transparency

In compliance with rigorous research reporting standards:
1. **Embedding LOO Circularity vs. Reference Proxy vs. Encoder-Free Validation:**
   - The $r_s = 1.0000$ correlation between Embedding LOO importance and Sentence-BERT LOO drop is explicitly labeled as a **circular self-consistency check**.
   - Sentence-BERT cosine drop relative to human reference text is explicitly labeled as a **Reference-Guided Embedding Proxy** (`Reference-Guided Embedding Proxy Drop`).
   - True external validation is established via **Independent Deterministic Requirement Drop** (100% encoder-free lexical/stem + operator loss against human annotations), where **Embedding LOO ($r_s = 0.5593$) and Gradient Attribution ($r_s = 0.5533$) are statistically equivalent ($\Delta r_s = -0.0061, p_{\text{Holm}} = 0.889$)**.
2. **Cluster Bootstrapping & Multiple-Testing Correction:** All prompt- and task-level comparisons compute correlations within individual prompts first, bootstrap prompts as intact clusters ($B = 10,000$), and apply Holm-Bonferroni step-down correction.
3. **Rule-Based Span Grouping Scope:** `DependencyAwarePruner` is explicitly documented as **rule-based critical-span grouping** (negation clauses, numeric bounds, conditional exceptions, priority directives), not a general syntactic dependency parser.
4. **Transparent Baseline & Reference Editor Labeling:** LLMLingua and LLMLingua-2 are explicitly labeled as verified fallback implementations (token perplexity and grammatical token classification) due to modern Transformers KV-cache incompatibilities, and the 20-prompt human reference baseline is disclosed as an internal expert/developer curation.

---

## Literature Grounding

SCULP is positioned against key literature in semantic prompt compression:
- **LLMLingua-2 (Findings of ACL 2024):** Adopts data-driven extractive compression; SCULP contributes multi-attribute component collapse analysis and downstream functional validation.
- **AttnComp (AAAI 2025):** Focuses on attention-based KV cache allocation; SCULP evaluates pre-cache lexical pruning and semantic preservation across black-box LLM pipelines.
- **Information Preservation in Prompt Compression (Findings of EMNLP 2025):** SCULP adds blind cross-model evaluation protocols and minimal-pair semantic stress testing.

*Detailed comparison and literature matrix available in [`LITERATURE.md`](LITERATURE.md).*

---

## License & Citation

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

If you use SCULP in your research, please cite:

```bibtex
@article{sculp2026,
  title={Semantic Collapse Under Lexical Pruning: An Importance-Score-Driven Framework for Evaluating Prompt Meaning Survival Across LLM Pipelines},
  author={Aditya and Contributors},
  journal={Research Repository},
  year={2026},
  url={https://github.com/aditya-10k/sculp}
}
```
