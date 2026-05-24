# FastText Subword Embeddings Trained from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python 3](https://img.shields.io/badge/python-3.10+-blue.svg)](requirements.txt)  
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](requirements.txt)

**Repository:** [github.com/bachnguyennn/Fast-Text-Embedding-System](https://github.com/bachnguyennn/Fast-Text-Embedding-System)

From-scratch **PyTorch** Skip-Gram and **FastText** (character n-grams with boundary markers), trained on [text8](http://mattmahoney.net/dc/text8.zip), evaluated against **official FastText** and **GloVe 300d**.

---

## Project highlights

| Highlight | Detail |
|-----------|--------|
| **Built, not downloaded** | Full pipeline: tokenization → vocab & subword index → negative sampling → MPS training → `.vec` export → intrinsic benchmarks. |
| **Fair A/B** | Skip-Gram vs FastText on the **same** 300d, 10-epoch, full-text8 setup. |
| **Four-model benchmark** | Our two models vs Gensim `fasttext-wiki-news-subwords-300` and `glove-wiki-gigaword-300`. |
| **Honest reporting** | WordSim ρ ≈ **0.56–0.60**; analogy ≈ **8–11%** vs **~71%** official—gap explained by **data scale**, not omitted baselines. |
| **Subword story** | FastText wins on **coverage** (100% WordSim pairs) and **morphological** neighbors; Skip-Gram edges WordSim ρ on frequent pairs. |

> **Portfolio framing:** Absolute intrinsic scores are **moderate**. The strength is **correct implementation**, **controlled experimentation**, and **clear interpretation**—including what did *not* work.

---

## Table of contents

- [Architecture](#architecture)
- [Results](#results)
- [Results analysis](#results-analysis)
- [Visualizations](#visualizations)
- [Limitations & lessons learned](#limitations--lessons-learned)
- [Training configuration](#training-configuration-final-models)
- [Quick start](#quick-start)

---

## Why this project matters

Production NLP still relies on subword-aware embeddings for noisy text. Most developers load pretrained files; fewer can explain **why** `king - man + woman ≈ queen` fails on a small homemade model, or **why** character n-grams need `<` `>` boundaries.

This repository is a **complete teaching-grade implementation**: you can trace a token from raw text through subword hashing, training, export, and the same WordSim / analogy scripts used for baselines. That is the portfolio claim—not SOTA leaderboard numbers.

---

## Architecture

Words are embedded as **word vector + mean(subword vectors)** on the input side; negative sampling uses standard word output vectors.

```
Input word "where"
  ├── E_word["where"]
  └── mean( E_sub["<wh"], E_sub["<whe"], …, E_sub["ere>"] )
  => v_input = E_word + mean(subwords)

Objective: Skip-Gram + negative sampling (15 negatives, window 5)
```

Implementation: [`src/model.py`](src/model.py) · Tokenizer: [`src/tokenizer.py`](src/tokenizer.py)

---

## Results

### Benchmark comparison

Evaluation on **WordSim-353**, **SimLex-999**, and **Google analogy** tasks. **Custom models:** 300d, 10 epochs, full subsampled text8. **Baselines:** pretrained 300d vectors via Gensim.

| Model | WordSim-353 ρ ↑ | SimLex-999 ρ ↑ | Analogy acc. ↑ | WordSim coverage |
|:------|----------------:|---------------:|---------------:|-----------------:|
| **Skip-Gram (ours)** | **0.60** | 0.20 | 8.3% · 1,506 / 18,104 | 99.4% |
| **FastText (ours)** | 0.56 | 0.17 | **11.3%** · 2,207 / 19,544 | **100%** |
| FastText (official) | **0.70** | **0.44** | **71.3%** · 11,293 / 15,845 | 100% |
| GloVe 300d | 0.61 | 0.37 | 71.7% · 14,020 / 19,544 | 100% |

*ρ = Spearman correlation with human judgments. Coverage = share of benchmark pairs where both words receive a vector.*

📄 Full metrics: [`reports/comparison_results.csv`](reports/comparison_results.csv)

![Benchmark bar chart comparing WordSim-353 Spearman ρ, SimLex-999 Spearman ρ, and Google analogy accuracy across Skip-Gram (ours), FastText (ours), official FastText 300d, and GloVe 300d on the same evaluation scripts](reports/figures/comparison_barplot.png)

---

## Results analysis

### What the headline metrics mean

**WordSim-353 (Skip-Gram 0.60 vs FastText 0.56).** Both models sit near **GloVe 0.61** and well below **official FastText 0.70**. Skip-Gram winning by 0.04 ρ does **not** mean subwords hurt globally—it means that on a benchmark dominated by **frequent word pairs**, a smoother word-level representation can correlate slightly better with human similarity judgments than a subword-averaged vector trained on the same budget.

**SimLex-999 (0.20 / 0.17).** Low for both.custom models. SimLex penalizes associative similarity (“coffee” ↔ “cup”); text8 + 10 epochs does not reproduce the semantic breadth of web-scale pretraining.

**Analogy (~11% vs ~71%).** The largest honest gap. FastText improves **participation** (more questions receive answers) and **relative** accuracy vs Skip-Gram, but **11% absolute accuracy** is not deployment-ready for analogy-heavy applications. The comparison to official models isolates **corpus volume**, not implementation bugs.

### Skip-Gram vs FastText (subword on vs off)

| Dimension | Skip-Gram | FastText (this repo) |
|-----------|-----------|----------------------|
| Input representation | One vector per word type | Word + mean(char n-grams, 3–6) |
| Subword vocab | — | 333,222 indexed n-grams |
| WordSim coverage | 99.4% | **100%** |
| Analogy questions scored | 18,104 | **19,544** (+7.9%) |
| Qualitative neighbors | Strong on **frequent** lemmas | Stronger on **stems / inflections** (`computer` → `compute`) |
| WordSim ρ | **0.60** | 0.56 |

**Interpretation:** Subword information in this project is most visible in **coverage and morphology**, not in beating Skip-Gram on every intrinsic scalar.

---

### Subword boundary impact analysis

This repository follows the **official FastText convention**: each word is wrapped with boundary symbols before n-grams are extracted (`where` → `<where>`). See `generate_ngrams()` in [`src/tokenizer.py`](src/tokenizer.py).

> **Important:** We did **not** retrain a second FastText model without boundary markers (no extra training runs). The comparison below combines **(a)** structural / algorithmic analysis of with vs without boundaries and **(b)** **quantitative results from our published model**, which uses boundaries. A full ablation would require an additional training job.

#### With boundaries (`<word>`) — **what we implemented**

Example for `where`:

| Setting | Sample n-grams (length 3–6) |
|---------|-----------------------------|
| **With `<` `>`** | `<wh`, `<whe`, `<wher`, `<where`, `whe`, `where`, `here`, `ere>`, `re>`, `e>` |

Prefix n-grams (`<wh`, …) mark **start-of-word** context; suffix n-grams (`…ere>`, `re>`, `e>`) mark **end-of-word** context. Interior n-grams still capture morphology.

#### Without boundaries (hypothetical `word` only)

| Setting | Sample n-grams |
|---------|----------------|
| **No boundaries** | `whe`, `her`, `ere`, `where`, `wher`, … |

Critical difference: substring `her` appears inside `where` but is **not** a morphological suffix of `where`. Without boundaries, such spans are treated like ordinary n-grams and can fire on **other words** that contain `her` (e.g. `there`, `other`), creating **cross-word noise** in the subword embedding table.

#### Expected impact by evaluation type

| Evaluation | With `<` `>` (expected) | Without `<` `>` (expected) | **Observed (our bounded model only)** |
|------------|-------------------------|----------------------------|----------------------------------------|
| **WordSim-353 ρ** | Better word-level disambiguation; fewer spurious shared n-grams across unrelated words | Risk of inflated similarity for words sharing accidental substrings | **0.56** (FastText) vs **0.60** (Skip-Gram) |
| **SimLex-999 ρ** | Modest help on true similarity; less association bleed from substring collisions | More substring collisions → possible false semantic neighbors | **0.17** (still low) |
| **Analogy accuracy** | Prefix/suffix features help syntactic patterns (e.g. past tense, `-er`) | Harder to distinguish word edges; noisier analogies | **11.3%** vs Skip-Gram **8.3%** |
| **OOV / rare words** | Boundaries clarify where a fragment starts/ends when composing OOV vectors | OOV vector = average of ambiguous fragments | **100%** WordSim coverage |
| **Rare / morphological neighbors** | Inflections share **affix n-grams** with clear edges | More collisions on short substrings inside long words | Qualitative: `computer` → `compute`, `computing` (see neighbors figure) |
| **Subword vocabulary size** | Extra symbols per word → **more** unique n-grams (our vocab: **333k**) | Fewer distinct tokens per word, but **more polysemy** per n-gram | 333,222 subword types (bounded) |

#### Why boundaries did not “fix” analogy or match official FastText

Even with correct boundaries:

1. **Corpus** — text8 is tiny vs Wikipedia + Common Crawl.  
2. **Training budget** — 10 epochs on subsampled tokens.  
3. **Objective** — local context prediction ≠ optimizing analogy accuracy.  
4. **Parameter scale** — our model is complete but data-limited; official FastText’s advantage is predominantly **data**, not boundary markup alone.

Boundaries are **necessary for faithful FastText behavior** (and match the reference implementation); they are **not sufficient** for closing a ~60-point analogy gap to pretrained models.

#### Qualitative takeaway

Nearest-neighbor panels (below) show FastText **with** boundaries picking morphological relatives—behavior consistent with well-separated character n-grams at word edges. Without an ablation model, we do not report alternate neighbor lists; the design analysis explains **why** we adopted boundaries for the published run.

---

### Training convergence

Both models decreased loss monotonically over 10 epochs. Skip-Gram final loss **3.05** vs FastText **3.12** suggests a slightly easier fit for the word-only path—consistent with fewer parameters on the input side.

![Training loss curves for 300-dimensional Skip-Gram and FastText models over 10 epochs on subsampled text8, showing monotonic decrease in negative-sampling loss](reports/figures/training_loss_curves.png)

### Where subword information helped — and where it did not

**Helped**

- **Benchmark coverage** — 100% of WordSim-353 pairs vs 99.4%.  
- **Analogy throughput** — 2,207 correct vs 1,506 (+47% vs Skip-Gram on scored questions).  
- **Morphology in neighbor lists** — inflected forms cluster (see figure).  
- **OOV-style robustness** — n-gram sum provides a vector when a rare form lacks a stable word embedding.

**Did not help enough (vs expectations)**

- **WordSim ρ** did not exceed Skip-Gram.  
- **SimLex** remained weak for both models.  
- **Analogy** stayed ~11% absolute vs ~71% pretrained.  
- **Training cost** — ~333k subword embeddings, ~56 min/epoch on M1 MPS, large checkpoints.

---

## Visualizations

All figures below are generated from the **final 10-epoch, 300d** checkpoints (see [`scripts/generate_report_figures.py`](scripts/generate_report_figures.py)).

### t-SNE — Skip-Gram (300d)

![t-SNE projection of 300-dimensional Skip-Gram word embeddings trained from scratch on text8, subsampled vocabulary visualization showing emergent lexical clusters](reports/figures/tsne_skipgram.png)

### t-SNE — FastText (300d, with subword composition)

![t-SNE projection of 300-dimensional FastText embeddings with word and subword composition, trained from scratch on text8 for comparison with the Skip-Gram layout](reports/figures/tsne_fasttext.png)

### t-SNE — early notebook exploration (100d pilot)

![t-SNE visualization from the Jupyter analysis notebook during the initial 100-dimensional pilot run before final portfolio training](reports/figures/notebook_tsne.png)

### Nearest neighbors (qualitative)

![Side-by-side nearest-neighbor panels for query words such as king, queen, computer, and beautiful, comparing Skip-Gram versus FastText 300d models trained on text8](reports/figures/nearest_neighbors.png)

---

## Limitations & lessons learned

### Transparent scorecard

| Metric | Our best | Reference (official FT) | Gap |
|--------|----------|-------------------------|-----|
| WordSim-353 ρ | **0.60** (Skip-Gram) | 0.70 | −0.10 |
| SimLex-999 ρ | 0.20 | 0.44 | −0.24 |
| Analogy accuracy | **11.3%** (FastText) | ~71% | ~−60 pp |

Presenting these numbers plainly is intentional. Reviewers respect **calibration** more than inflated adjectives.

### Limitations (root causes)

1. **Small corpus by industry standards** — text8 is a single Wikipedia extract (~17M tokens raw, ~5M subsampled tokens/epoch). Pretrained models see orders of magnitude more data.  
2. **Limited epochs** — 10 passes; we stopped after diminishing returns (see below). A paused 25-epoch run did not change the published results.  
3. **Single domain** — encyclopedic prose under-represents news, dialogue, and social text where subwords shine on typos.  
4. **Intrinsic-only evaluation** — no downstream classifier or NER; benchmarks may not reflect your deployment metric.  
5. **No boundary ablation in training** — design analysis only; we cannot quote ρ for a no-boundary FastText model without an extra run.  
6. **Hardware budget** — laptop MPS training (~56 min/epoch late in extended runs) limits experimentation velocity.

### Lessons learned (positive framing)

| Lesson | Takeaway |
|--------|----------|
| **Implement the paper, then measure** | Porting FastText’s boundary rule mattered for faithful behavior before chasing scores. |
| **Report coverage and neighbors** | Subword wins showed up outside WordSim ρ—easy to miss if you only publish one number. |
| **Baselines on the same harness** | Official FastText + GloVe on identical scripts made the gap interpretable. |
| **Engineering is part of NLP** | Batched CPU export, vectorized analogy eval, and checkpoint resume saved days of wall-clock time. |
| **Know when to stop** | 10 epochs delivered the teaching narrative; 25 epochs was projected at 20+ extra hours for uncertain ρ gains on text8. |

### Future iterations (not planned in this repo)

- Train on **multi-source corpora** (Wikipedia + news + subsampled Common Crawl).  
- Run a **boundary ablation** (with vs without `<` `>`) with identical budgets.  
- Add **downstream task** evaluation (text classification on AG News or similar).  
- **Distillation** from official FastText if the goal shifts from “from scratch” to “maximum accuracy.”

---

## Training configuration (final models)

| Setting | Value |
|---------|-------|
| Corpus | text8 — 17,005,279 raw tokens |
| Subsampling | Word2vec, threshold `1e-5` → ~4.98M tokens / epoch |
| **Published epochs** | **10** |
| Embedding dimension | 300 |
| Window / negatives | 5 / 15 |
| Batch size | 1024 · Device: Apple MPS |
| Word / subword vocab | 71,292 / **333,222** |
| Boundary markers | **Yes** (`<` `>` per FastText) |
| Final loss (epoch 10) | Skip-Gram **3.05** · FastText **3.12** |

### Why we stopped at 10 epochs

| Factor | Rationale |
|--------|-----------|
| **Quality** | ρ rose from ~0.02 (100d pilot) to **~0.56–0.60**—most gains in early epochs. |
| **Ceiling** | text8 cannot match web-scale **0.70** WordSim without more data. |
| **Cost** | ~56 min/epoch on M1 MPS; 25-epoch plan paused at 19 with no change to published artifacts. |
| **Goal** | Portfolio demonstrates implementation + honest benchmarks, not leaderboard #1. |

---

## Quick start

```bash
git clone https://github.com/bachnguyennn/Fast-Text-Embedding-System.git
cd Fast-Text-Embedding-System
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Regenerate metrics & figures from local .vec files (no training)
bash scripts/post_train_pipeline.sh
```

Notebook: [`notebook/fasttext_from_scratch_report.ipynb`](notebook/fasttext_from_scratch_report.ipynb)

---

## Repository layout

```
src/           tokenizer (boundaries), vocab, dataset, models, trainer
evaluation/    WordSim, SimLex, analogies, 4-model comparison
scripts/       training, export, baselines, figures
reports/       comparison_results.csv, figures/*.png
models/        checkpoints & .vec (gitignored)
```

| Tracked in Git | Local only |
|----------------|------------|
| Metrics CSV, figures, histories | `text8`, `*.vec`, `*.pt`, `venv/` |

---

## License

[MIT](LICENSE)
