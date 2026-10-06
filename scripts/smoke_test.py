"""End-to-end sanity smoke test on the specified research sentence:
'That cat chased the mouse across the garden.'
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring.random import RandomScorer
from src.scoring.embedding import EmbeddingLeaveOneOutScorer
from src.scoring.gradient import GradientScorer
from src.scoring.integrated_gradients import IntegratedGradientScorer
from src.pruning.pruner import PromptPruner
from src.interpretation.interpreter import BlindInterpreter
from src.analysis.plots import plot_token_heatmap


def run_smoke_test(sentence: str = "That cat chased the mouse across the garden."):
    print("=" * 70)
    print("SMOKE TEST SANITY CHECK")
    print(f"Sentence: \"{sentence}\"")
    print("=" * 70)

    # 1. Initialize Scorers
    scorers = {
        "Random": RandomScorer(seed=42),
        "Embedding (LOO)": EmbeddingLeaveOneOutScorer(seed=42),
        "Gradient": GradientScorer(seed=42),
        "Integrated Gradients": IntegratedGradientScorer(n_steps=15, seed=42)
    }

    pruner = PromptPruner()
    interpreter = BlindInterpreter(provider="mock")

    for method_name, scorer in scorers.items():
        print(f"\n>>> Method: {method_name}")
        score_res = scorer.score_prompt(sentence, prompt_id="smoke_cat")

        print("\nTOKEN / WORD ALIGNMENT & SCORES:")
        print(f"{'Word ID':<8} {'Word':<12} {'Raw Score':<12} {'Normalized':<12} {'Rank':<6}")
        print("-" * 52)
        for w in score_res.word_scores:
            print(f"{w.word_id:<8} {w.word:<12} {w.raw_score:<12.4f} {w.normalized_score:<12.4f} {w.rank:<6}")

        sorted_words = sorted(score_res.word_scores, key=lambda x: x.normalized_score, reverse=True)
        top_words = [w.word for w in sorted_words[:3]]
        bottom_words = [w.word for w in sorted_words[-3:]]
        print(f"\nTop important: {', '.join(top_words)}")
        print(f"Bottom important: {', '.join(bottom_words)}")

        print("\nCOMPRESSED VARIANTS:")
        for cr in [0.1, 0.3, 0.5, 0.7]:
            prune_res = pruner.prune(score_res, target_removal=cr)
            recon = interpreter.interpret(prune_res.compressed_prompt)
            print(f"  {int(cr*100)}% removed: \"{prune_res.compressed_prompt}\" (Actual tokens: {prune_res.remaining_tokens}/{prune_res.original_tokens})")
            print(f"       -> Blind Reconstruction: task='{recon.task}' | context={recon.context}")

        # Generate token importance heatmap for this sentence
        if method_name == "Embedding (LOO)":
            words = [w.word for w in score_res.word_scores]
            scores = [w.normalized_score for w in score_res.word_scores]
            plot_token_heatmap(
                words, scores,
                title="Embedding LOO Word Importance: 'That cat chased the mouse...'",
                out_path="results/plots/token_heatmap_smoke.png"
            )
            print("\n[Visual Heatmap saved to results/plots/token_heatmap_smoke.png]")

    print("\n" + "=" * 70)
    print("Smoke test successfully completed for all 4 attribution methods!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
