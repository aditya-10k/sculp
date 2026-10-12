# SCULP Literature Synthesis & Research Positioning

**Project:** Semantic Collapse Under Lexical Pruning (SCULP)  
**Auditor / Researcher:** Google DeepMind Agentic Pair-Programmer & Human Co-Investigator  
**Document Date:** 2026-10-12  

---

## 1. Literature Grounding & Comparative Matrix

The prompt compression literature has evolved rapidly from early heuristic stopword removal to perplexity filtering, learned classification, and KV-cache attention steering. Below is an objective synthesis of related work, identifying established techniques, points of overlap, and the specific gap SCULP addresses.

| Literature Reference | Core Innovation | Evaluation Methodology | Overlap with SCULP | Remaining Research Gap Addressed by SCULP |
|:---|:---|:---|:---|:---|
| **LLMLingua-2**<br>*(Pan et al., Findings of ACL 2024)* | Distills task-agnostic token compression from GPT-4 into a multilingual BERT token classification model. | Perplexity budgets, QA accuracy on MeetingBank, LongBench, ZeroScrolls. | Uses transformer token classification as an external baseline; prunes lexical units to target budgets. | Treats token compression as generic binary classification ($P(\text{keep})$). Does not explain *which* semantic dimensions (constraints, tone, dependencies) collapse first, nor does it evaluate whether feature attributions align with actual token deletion damage. |
| **AttnComp**<br>*(AAAI 2025)* | Steers KV-cache token retention by aggregating multi-head attention weights during LLM generation. | Long-context memory retention, throughput speedup, token perplexity. | Exploits internal transformer representations to evaluate token necessity. | Focuses on generation-time KV-cache eviction in long documents. Does not address prompt-level semantic preservation, structural constraint survival, or blind downstream interpretability. |
| **Understanding & Improving Info Preservation**<br>*(Wang et al., Findings of EMNLP 2025)* | Identifies that perplexity/entropy filters disproportionately discard critical low-frequency modifiers, negations, and formatting instructions. | Semantic error classification, downstream constraint satisfaction on structured prompts. | Directly shares the observation that high surface/embedding similarity ($> 0.80$) often masks severe constraint collapse ($< 0.50$). | Does not provide a formal rate-distortion framework, discrete marginal drop ($\Delta_i$) collapse breakpoints, or independent cross-model blind vs. reference-guided protocol separation. |
| **Prompt Compression in the Wild**<br>*(2026 Preprint)* | Benchmarks compression across real-world agentic workflows (APIs, tool calls, multi-turn dialogue). | Tool-call accuracy, schema adherence, end-to-end task success. | Evaluates functional downstream task failure induced by lexical pruning. | Empirical benchmarking across existing tools without a theoretical framework testing the attribution faithfulness of token-importance estimators. |

---

## 2. Refined Primary Research Question

Based on the synthesis above, SCULP avoids claiming novelty for basic lexical pruning or embedding similarity. Instead, the investigation centers on the following formal question:

> **Primary Research Question:**  
> *How well do token-importance estimators predict task-critical information loss under progressive prompt compression, and how does their predictive validity vary across compression severity, prompt component categories, and dependency structures?*

### Formal Hypotheses:

1. **H1 (Attribution Faithfulness):** Token importance scores derived from representation-preservation gradient attribution and embedding leave-one-out (LOO) exhibit statistically significant positive rank correlation (Spearman $r_s > 0$, Kendall $\tau > 0$) with empirical deletion damage, outperforming random token ranking.
2. **H2 (Differential Component Vulnerability):** Task-critical constraints and multi-step procedural dependencies suffer semantic collapse at substantially lower compression ratios ($CR \approx 10\text{--}30\%$) than general intent and narrative context ($CR \approx 60\text{--}80\%$).
3. **H3 (Protocol Divergence):** Standalone blind interpretability (Protocol A) degrades much more steeply than reference-guided semantic preservation (Protocol B), demonstrating that readers unable to access the original context experience severe task ambiguity even when compressed prompts retain core keywords.
4. **H4 (Dependency-Aware Pruning Benefit):** Grouping syntactic and semantic dependency spans during pruning preserves significantly more functional downstream task success than independent greedy token deletion at matched actual token budgets.

---

## 3. Scope of Claims and Boundaries

- **Established Techniques Used:** WordPiece subword alignment, Captum Integrated Gradients, Sentence-BERT cosine similarity, Wilcoxon signed-rank tests, Holm-Bonferroni correction.
- **Novel Contributions of SCULP:**
  1. A multi-component decomposition framework explicitly separating Intent, Constraints, Context, Tone, and Dependencies.
  2. Protocol separation between Standalone Blind Interpretability (Protocol A) and Reference-Guided Semantic Preservation (Protocol B).
  3. Discrete marginal gradient ($\Delta_i$) collapse breakpoint analysis coupled with operational thresholds ($\tau_{0.80}, \tau_{0.70}$).
  4. Empirical alignment benchmarking testing whether token-importance attributions faithfully predict empirical deletion damage.
