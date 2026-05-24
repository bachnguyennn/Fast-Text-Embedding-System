# FastText Subword Embeddings Trained from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Repository:** [github.com/bachnguyennn/Fast-Text-Embedding-System](https://github.com/bachnguyennn/Fast-Text-Embedding-System)

A from-scratch **PyTorch** implementation of Skip-Gram with Negative Sampling and **FastText subword n-grams**, trained on the [text8](http://mattmahoney.net/dc/text8.zip) corpus (~17M tokens). The project compares a plain Skip-Gram baseline against FastText on the same data, and benchmarks both against **official pre-trained FastText** and **GloVe 300d** via Gensim.

## Motivation

Word embeddings map tokens to dense vectors that capture semantic similarity. Classic Skip-Gram (Word2Vec) learns one vector per word, which fails on rare or out-of-vocabulary (OOV) tokens. **FastText** extends Skip-Gram by representing each word as the sum of its word vector and its character n-gram vectors (typically 3–6 characters). That makes embeddings more robust to morphology, typos, and unseen words.

This repository implements the full pipeline—tokenization, vocabulary, subsampling, negative sampling, efficient MPS training, intrinsic evaluation, and visualization—so the effect of subword information can be measured directly.

## Architecture

```
Input word "where"
  ├── word embedding:     E_word["where"]
  └── subword embeddings: E_sub["<wh"], E_sub["<whe"], ... , E_sub["ere>"]
  => input vector = E_word + mean(E_sub[ngrams])

Skip-Gram + Negative Sampling:
  maximize log σ(v_center · v_context) + Σ log σ(-v_center · v_negative)
```

## Project Structure

```
fasttext_embeddings_from_scratch/
├── src/                       # tokenizer, vocab, dataset, model, trainer, device
├── evaluation/                # WordSim, analogy, 4-model comparison
├── scripts/
│   ├── train_portfolio_models.py   # 300d, 10 epochs, full text8
│   ├── export_checkpoint.py        # recover .vec after training (OOM-safe)
│   ├── download_baselines.py
│   ├── generate_report_figures.py
│   └── post_train_pipeline.sh      # evaluate + figures (no extra training)
├── notebook/                  # interactive report
├── reports/figures/           # README visuals
├── models/                    # .vec / .pt checkpoints (gitignored)
├── train.py / evaluate.py / visualize.py
└── README.md
```

## Training decision — why we stopped at 10 epochs

**Final reported models use 10 epochs on full text8 (300d).** We intentionally did **not** continue to 25 epochs.

| Factor | What we found |
|--------|----------------|
| **Quality** | After 10 epochs, WordSim-353 ρ reached **~0.60** (Skip-Gram) and **~0.56** (FastText)—a large jump from the 100d pilot (~0.02) and close to GloVe 300d (**0.61**) on the same benchmarks. |
| **Ceiling** | Official FastText (**0.70** WordSim, **~71%** analogies) is trained on Wikipedia + news at web scale. Matching **0.68+** on text8 alone is unrealistic; diminishing returns set in well before that. |
| **Time** | A follow-up run toward 25 epochs averaged **~56 min/epoch** on Apple M1 Pro MPS (~5M samples/epoch, 333k subword vocabulary). Skip-Gram reached epoch 19/25 before we **paused**—estimated **~20+ more hours** for both models was not justified for portfolio goals. |
| **Portfolio goal** | The project demonstrates **implementation**, **controlled comparison** (Skip-Gram vs FastText), and **honest benchmarking** vs official embeddings—not beating Facebook’s pre-trained model. |

We later added **faster training** (`FastCorpusSkipGramDataset`, vectorized collation, MPS AMP, checkpoint resume) for anyone who wants to continue, but the **published results** below are from the completed **10-epoch** run.

## Quick Start

### Clone & environment

```bash
git clone https://github.com/bachnguyennn/Fast-Text-Embedding-System.git
cd Fast-Text-Embedding-System
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Reproduce evaluation & figures (no training required)

Trained `.vec` files are produced locally and gitignored. To regenerate **metrics and plots** from existing vectors:

```bash
bash scripts/post_train_pipeline.sh
```

### Train from scratch (optional, ~20+ hours total on M1 MPS)

```bash
caffeinate -dims python scripts/train_portfolio_models.py
```

If FastText export hits MPS OOM at the end:

```bash
python scripts/export_checkpoint.py --checkpoint models/fasttext_best.pt
```

Training uses **MPS** when available:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Baselines

```bash
python scripts/download_baselines.py
python evaluate.py
```

### Notebook

```bash
jupyter notebook notebook/fasttext_from_scratch_report.ipynb
```

## Training configuration (final — 10 epochs)

| Setting | Value |
|---------|-------|
| Corpus | Full text8 (17,005,279 raw tokens) |
| Subsampling | Word2vec (threshold 1e-5) → ~4.98M tokens/epoch |
| **Epochs (final)** | **10** |
| Embedding dim | **300** |
| Window | 5 |
| Negative samples | 15 |
| Batch size | 1024 |
| Device | **Apple MPS** (M1 Pro) |
| Optimizer | Adam (lr=0.0025) + ReduceLROnPlateau |
| Word vocab | 71,292 | Subword vocab | 333,222 |

**Final training loss (epoch 10):** Skip-Gram **3.05** · FastText **3.12**

## Results

Intrinsic evaluation on **WordSim-353**, **SimLex-999**, and **Google analogy** questions. Custom models: **300d × 10 epochs**, full subsampled text8. Baselines: Gensim `fasttext-wiki-news-subwords-300`, `glove-wiki-gigaword-300`.

| Model | WordSim-353 ρ | SimLex-999 ρ | Analogy accuracy | WordSim coverage |
|-------|---------------|--------------|------------------|------------------|
| **Skip-Gram (ours)** | **0.60** | 0.20 | 8.3% (1,506 / 18,104) | 99.4% |
| **FastText (ours)** | 0.56 | 0.17 | **11.3%** (2,207 / 19,544) | **100%** |
| FastText (official, 300d) | **0.70** | **0.44** | **71.3%** (11,293 / 15,845) | 100% |
| GloVe 300d | 0.61 | 0.37 | 71.7% (14,020 / 19,544) | 100% |

Raw metrics: [`reports/comparison_results.csv`](reports/comparison_results.csv)

### Benchmark comparison

![Bar chart comparing WordSim-353, SimLex-999, and analogy accuracy across all four embedding models](reports/figures/comparison_barplot.png)

### Training loss

![Training loss curves for Skip-Gram and FastText over 10 epochs on full text8](reports/figures/training_loss_curves.png)

### Embedding space (t-SNE)

![t-SNE projection of Skip-Gram word vectors (300d)](reports/figures/tsne_skipgram.png)

![t-SNE projection of FastText word vectors with subword composition (300d)](reports/figures/tsne_fasttext.png)

### Nearest neighbors (qualitative)

![Nearest neighbors for king, queen, computer, beautiful, and related query words for Skip-Gram vs FastText](reports/figures/nearest_neighbors.png)

FastText links inflected forms (e.g. `computer` → `compute`, `computing`) via character n-grams; Skip-Gram captures strong in-vocab similarity on WordSim-353.

## Key insights

1. **Full-corpus 300d training works**: WordSim ρ went from ~0.02 (100d pilot) to **~0.56–0.60**, near GloVe on the same tasks.
2. **FastText improves coverage and analogies**: **100%** WordSim coverage and **11.3%** analogy accuracy vs Skip-Gram **99.4%** / **8.3%** at the same setup.
3. **Skip-Gram can win on WordSim-353**: Slightly higher ρ (0.60 vs 0.56) when both are trained equally; subword averaging helps coverage more than raw WordSim on frequent pairs.
4. **Official models set the ceiling**: ~**71%** analogy accuracy vs ~**11%** for from-scratch text8—data scale, not a bug in our code.
5. **Stopping at 10 epochs is deliberate**: Extra epochs cost days on laptop MPS for marginal gains; the comparison story is already clear.

## Implementation highlights

| Component | Details |
|-----------|---------|
| Tokenizer | Lowercasing, punctuation removal, FastText 3–6 char n-grams |
| Training | Subsampling, fast corpus iterator, vectorized collate, MPS + optional AMP |
| Models | `SkipGramModel`, `FastTextModel` (word + mean subword on input) |
| Evaluation | Vectorized analogy (custom `.vec`); Gensim `most_similar` (official) |
| Export | Word2vec `.vec`; batched CPU export (avoids MPS OOM on 71k FastText vectors) |
| Resume | Checkpoints store optimizer state (`--resume-from`, `--target-epochs`) |

## What is not in Git

**Gitignored:** `data/raw/text8`, `models/*.vec`, `models/*.pt`, `venv/`.  
**Included:** `reports/comparison_results.csv`, training histories, `reports/figures/*.png`.

## Notebook

[`notebook/fasttext_from_scratch_report.ipynb`](notebook/fasttext_from_scratch_report.ipynb) — EDA, training, neighbors, benchmarks, t-SNE.

## License

See [LICENSE](LICENSE).
