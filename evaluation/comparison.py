"""Compare custom Skip-Gram/FastText models against official pretrained embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evaluation.analogy import AnalogyLookup, evaluate_analogies
from evaluation.wordsim import EmbeddingLookup, run_wordsim_evaluation
from src.utils import set_seed


@dataclass
class ModelSpec:
    name: str
    vec_path: str | None = None
    gensim_path: str | None = None
    gensim_name: str | None = None
    use_subwords: bool = False
    optional: bool = False


def _load_gensim_model(path: str):
    from gensim.models import KeyedVectors

    path_obj = Path(path)
    if path_obj.suffix == ".bin":
        return KeyedVectors.load_word2vec_format(str(path_obj), binary=True)
    return KeyedVectors.load_word2vec_format(str(path_obj), binary=False)


def _load_gensim_by_name(name: str):
    import gensim.downloader as api

    return api.load(name)


def evaluate_model(spec: ModelSpec) -> Dict[str, float | int | dict]:
    if spec.vec_path and Path(spec.vec_path).exists():
        wordsim_lookup = EmbeddingLookup.from_vec(spec.vec_path)
        analogy_lookup = AnalogyLookup.from_vec(spec.vec_path, use_subwords=spec.use_subwords)
    elif spec.gensim_name:
        model = _load_gensim_by_name(spec.gensim_name)
        wordsim_lookup = EmbeddingLookup.from_gensim(model)
        analogy_lookup = AnalogyLookup.from_gensim(model, use_subwords=spec.use_subwords)
    elif spec.gensim_path and Path(spec.gensim_path).exists():
        model = _load_gensim_model(spec.gensim_path)
        wordsim_lookup = EmbeddingLookup.from_gensim(model)
        analogy_lookup = AnalogyLookup.from_gensim(model, use_subwords=spec.use_subwords)
    else:
        if spec.optional:
            return {"status": "skipped", "model": spec.name, "reason": "missing optional model file"}
        raise FileNotFoundError(f"Model file not found for {spec.name}")

    wordsim_metrics = run_wordsim_evaluation(wordsim_lookup, use_subwords=spec.use_subwords)
    analogy_metrics = evaluate_analogies(analogy_lookup)
    return {
        "status": "ok",
        "model": spec.name,
        **wordsim_metrics,
        "analogy_accuracy": analogy_metrics["accuracy"],
        "analogy_correct": analogy_metrics["correct"],
        "analogy_total": analogy_metrics["total_questions"],
    }


def run_full_comparison(
    skipgram_vec: str = "models/skipgram_final.vec",
    fasttext_vec: str = "models/fasttext_final.vec",
    official_fasttext: str | None = "models/cc.en.300.vec",
    glove_vec: str | None = "models/glove.6B.300d.txt",
    use_gensim_hub: bool = True,
    output_dir: str = "reports",
) -> pd.DataFrame:
    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ModelSpec(name="Skip-Gram (ours)", vec_path=skipgram_vec, use_subwords=False),
        ModelSpec(name="FastText (ours)", vec_path=fasttext_vec, use_subwords=True),
    ]
    if use_gensim_hub:
        specs.extend(
            [
                ModelSpec(
                    name="FastText (official)",
                    gensim_name="fasttext-wiki-news-subwords-300",
                    use_subwords=True,
                    optional=True,
                ),
                ModelSpec(
                    name="GloVe 300d",
                    gensim_name="glove-wiki-gigaword-300",
                    use_subwords=False,
                    optional=True,
                ),
            ]
        )
    else:
        specs.extend(
            [
                ModelSpec(
                    name="FastText (official)",
                    gensim_path=official_fasttext,
                    use_subwords=True,
                    optional=True,
                ),
                ModelSpec(name="GloVe 300d", gensim_path=glove_vec, use_subwords=False, optional=True),
            ]
        )

    rows: List[Dict[str, float | int | str]] = []
    for spec in specs:
        print(f"Evaluating {spec.name}...")
        try:
            metrics = evaluate_model(spec)
        except FileNotFoundError as exc:
            metrics = {"status": "missing", "model": spec.name, "reason": str(exc)}
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "comparison_results.csv", index=False)
    with open(output_dir / "comparison_results.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    ok_df = df[df.get("status", "ok") == "ok"].copy()
    if not ok_df.empty:
        plot_comparison_charts(ok_df, figures_dir)
    return df


def plot_comparison_charts(df: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    metrics = ["wordsim353_spearman", "simlex999_spearman", "analogy_accuracy"]
    melted = df.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="model")
    plt.title("Embedding Benchmark Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(figures_dir / "comparison_barplot.png", dpi=150)
    plt.close()


def print_results_table(df: pd.DataFrame) -> None:
    columns = [
        "model",
        "status",
        "wordsim353_spearman",
        "simlex999_spearman",
        "analogy_accuracy",
        "wordsim353_coverage",
    ]
    available = [col for col in columns if col in df.columns]
    printable = df[available].fillna("-")
    print(printable.to_string(index=False))


if __name__ == "__main__":
    results = run_full_comparison()
    print_results_table(results)
