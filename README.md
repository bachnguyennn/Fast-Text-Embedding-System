# FastText Subword Embeddings Trained from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Repository:** [github.com/bachnguyennn/Fast-Text-Embedding-System](https://github.com/bachnguyennn/Fast-Text-Embedding-System)

A from-scratch **PyTorch** implementation of Skip-Gram with Negative Sampling and **FastText subword n-grams**, trained on the [text8](http://mattmahoney.net/dc/text8.zip) corpus. The project compares a plain Skip-Gram baseline against FastText on the same data and benchmarks, and optionally against official pre-trained embeddings.

## Motivation

Word embeddings map tokens to dense vectors that capture semantic similarity. Classic Skip-Gram (Word2Vec) learns one vector per word, which fails on rare or out-of-vocabulary (OOV) tokens. **FastText** extends Skip-Gram by representing each word as the sum of its word vector and its character n-gram vectors (typically 3–6 characters). That makes embeddings more robust to morphology, typos, and unseen words.

This repository implements the full pipeline—tokenization, vocabulary, negative sampling, training, intrinsic evaluation, and visualization—so the effect of subword information can be measured directly.

## Architecture

```
Input word "where"
  ├── word embedding:     E_word["where"]
  └── subword embeddings: E_sub["<wh"], E_sub["<whe"], ... , E_sub["ere>"]
  => input vector = E_word + mean(E_sub[ngrams])

Skip-Gram + Negative Sampling:
  maximize log σ(v_center · v_context) + Σ log σ(-v_center · v_negative)
```

Two embedding tables are maintained:

- **Input side**: word vector (+ subword vectors for FastText)
- **Output side**: context word vectors for negative sampling

## Project Structure

```
fasttext_embeddings_from_scratch/
├── data/raw/                  # text8 corpus
├── data/processed/            # cleaned tokens + vocabulary
├── src/                       # tokenizer, vocab, dataset, model, trainer, utils
├── evaluation/                # WordSim, analogy, comparison
├── notebook/                  # full analysis notebook (recommended entry point)
├── reports/                   # comparison_results.csv, figures/
├── models/                    # checkpoints and .vec exports
├── train.py                   # CLI training
├── evaluate.py                # CLI benchmarks
└── visualize.py               # t-SNE plots
```

## What is not in Git

Large artifacts are **gitignored** and produced locally:

- `data/raw/text8` — downloaded automatically (~95 MB)
- `data/processed/` token files and `vocab.pkl`
- `models/*.vec`, `models/*.pt` — trained checkpoints

Benchmark outputs in `reports/comparison_results.csv` **are** included so results are visible without retraining.

## Quick Start

### Clone

```bash
git clone https://github.com/bachnguyennn/Fast-Text-Embedding-System.git
cd Fast-Text-Embedding-System
```

### Notebook (recommended)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook/fasttext_from_scratch_report.ipynb
```

Use the project venv as the Jupyter kernel. Run **Kernel → Restart & Run All**. Training takes ~25–35 minutes with default notebook settings; evaluation (Section 5) takes ~1–3 minutes.

### Command line

```bash
source venv/bin/activate

# Train both models (adjust flags as needed)
python train.py --model-type skipgram --epochs 5
python train.py --model-type fasttext --epochs 5

# Benchmarks + comparison chart
python evaluate.py

# t-SNE visualization
python visualize.py --vec-path models/fasttext_final.vec
```

Quick smoke test:

```bash
python train.py --model-type fasttext --epochs 2 --max-tokens 500000 --max-pairs 100000
```

## Training Configuration (notebook run)

| Setting | Value |
|---------|-------|
| Corpus | text8 (1M tokens sampled) |
| Training pairs | 150,000 (capped) |
| Epochs | 3 |
| Embedding dim | 100 |
| Window size | 5 |
| Negative samples | 10 |
| Optimizer | Adam (lr=0.003) + ReduceLROnPlateau |
| Word vocab | 13,968 |
| Subword vocab | 87,725 |
| OOV rate (on 1M tokens) | 6.19% |

**Training loss (converged):**

| Model | Epoch 1 | Epoch 3 |
|-------|---------|---------|
| Skip-Gram | 4.51 | 3.04 |
| FastText | 3.91 | 2.98 |

## Results

Intrinsic evaluation on WordSim-353, SimLex-999, and Google analogy questions. Results are saved to `reports/comparison_results.csv` after evaluation.

| Model | WordSim-353 ρ | SimLex-999 ρ | Analogy acc. | WordSim coverage |
|-------|---------------|--------------|--------------|------------------|
| **Skip-Gram (ours)** | -0.02 | -0.03 | 0.04% (3 / 7,323) | 78% |
| **FastText (ours)** | **0.02** | **0.02** | **0.28%** (45 / 16,261) | **91%** |
| FastText official 300d | ~0.75 | ~0.40 | ~75% | ~100% |
| GloVe 300d (optional) | ~0.60 | ~0.37 | ~70% | ~100% |

Official baselines were not included in the run above (pretrained files not downloaded). See below to add them.

### Qualitative examples (FastText, 100d)

| Query | Nearest neighbors |
|-------|-------------------|
| `computer` | compute, computed, launched, pseudo, dos |
| `science` | sci, mathematicians, frequencies, advance |
| `king` | jersey, origins, commander, kingdom, founded |

Morphological neighbors (`computer` → `compute`) show subword learning; classic semantic pairs (`king` → `queen`) require more data and dimensions than this run used.

### Optional: official baselines

Download pretrained vectors into `models/` and re-run evaluation:

- [FastText English 300d](https://fasttext.cc/docs/en/english-vectors.html) → `models/cc.en.300.vec`
- [GloVe 6B 300d](https://nlp.stanford.edu/projects/glove/) → `models/glove.6B.300d.txt`

```bash
python evaluate.py
```

## Implementation Highlights

| Component | Details |
|-----------|---------|
| Tokenizer | Lowercasing, punctuation removal, FastText 3–6 char n-grams with `<` `>` boundaries |
| Vocabulary | Min frequency = 5, `<UNK>` / `<PAD>`, unigram^0.75 negative sampling |
| Dataset | Window size 5, 10 negatives per positive pair |
| Models | `SkipGramModel`, `FastTextModel` (word + mean subword on input) |
| Export | Word2vec-compatible `.vec` format |

## Key Insights

1. **FastText improves coverage**: On the same training setup, FastText evaluated more WordSim pairs (91% vs 78%) and more analogy questions (16,261 vs 7,323) because subword n-grams provide vectors for OOV/rare forms.
2. **FastText wins on every reported metric** vs our Skip-Gram baseline, though absolute Spearman scores remain near zero with 1M tokens, 100 dimensions, and 3 epochs.
3. **Subwords capture morphology**: Neighbors like `compute` / `computed` for `computer` and `sci` for `science` demonstrate character-level generalization.
4. **Corpus scale dominates quality**: text8 is ~17M tokens; official FastText uses Wikipedia + Common Crawl at 300d. Expect a large gap until training on more data and dimensions.
5. **Negative sampling scales training**: A small number of negative draws per positive pair makes Skip-Gram practical on CPU.

## Limitations

- Subset of text8 (1M tokens) and capped training pairs (150k), not the full corpus
- 100-dimensional embeddings (benchmarks often use 300d)
- Low intrinsic scores and analogy accuracy vs published Word2Vec / FastText models
- Intrinsic evaluation only (no downstream classification or NER tasks)
- Official FastText / GloVe comparison optional and not run by default

## Improving Results

To push metrics closer to published benchmarks:

1. Train on **full text8** (`MAX_TOKENS = None`) with **5–10 epochs**
2. Use **`EMBEDDING_DIM = 300`**
3. Remove or raise the **`MAX_PAIRS`** cap
4. Add official **`cc.en.300.vec`** for comparison in the report

## Notebook

The main deliverable is [`notebook/fasttext_from_scratch_report.ipynb`](notebook/fasttext_from_scratch_report.ipynb):

1. Corpus EDA and Zipf plot  
2. Subword n-gram analysis  
3. Skip-Gram + FastText training (with skip-if-trained option)  
4. Nearest-neighbor inspection  
5. Benchmark comparison  
6. t-SNE visualization  
7. Error analysis and research value  

## License

See [LICENSE](LICENSE).
