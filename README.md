# Hybrid Datasets for Emotion Classification

Code accompanying the paper **"Hybrid Datasets for Emotion Classification: How Much Synthetic Data Is Too Much?"** (Maznyk & Terekhova).

This repository investigates how the proportion of synthetic (LLM-generated) training data affects the performance of a text-based emotion classifier, by training DistilRoBERTa-base on hybrid datasets mixed from real and synthetic sources at five real/synthetic ratios (100/0, 75/25, 50/50, 25/75, 0/100), evaluated only on real data.

> This project was developed as part of the seminar **EAITHD26** at **Osnabrück University**, Summer Semester 2026.

## Data

- **Real data:** [DAIR-AI Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) — English tweets labeled with one of six emotions (anger, fear, joy, love, sadness, surprise).
- **Synthetic data:** [ELSA](https://huggingface.co/datasets/joyspace-ai/ELSA-Emotion-and-Language-Style-Alignment-Dataset) — LLM-generated rewrites of DAIR-AI texts in four styles; this study uses the *conversational* style.

Neither dataset is bundled in this repo, both are downloaded automatically via the `datasets` library when the notebook is run.

## Method

- **Classes:** `sadness`, `joy`, `love`, `anger`, `fear`, `surprise` (`dair-ai/emotion`'s native label set).
- **Ratios tested:** 100% real / 75_25 / 50_50 / 25_75 / 0% real, each fixed at 10,000 training examples, stratified by label so class balance is constant across conditions.
- **val/test sets** are built from real data only, split off *before* any ratio mixing, so evaluation always reflects real-world performance regardless of what the model was trained on.
- **One model per ratio**, each fine-tuned from the same pretrained checkpoint (not continued from the previous ratio's weights), for 3 epochs.
- **Evaluation:** overall accuracy/macro-F1, per-emotion recall, and paired significance testing across ratios on the same test set.


## Project structure

```
Hybrid_datasets.ipynb   # full pipeline: data loading -> mixing -> training -> stats -> figures
```

## Requirements

```
transformers
datasets
torch
scikit-learn
statsmodels
pandas
numpy
matplotlib
```

## Running

Designed for Google Colab with a GPU runtime (uses `distilroberta-base` fine-tuning, ~3 epochs per ratio, 5 ratios total). Run top to bottom; later cells depend on variables defined earlier (`real_train_pool`, `label2id`, `mixed_datasets`, etc.), so partial/out-of-order execution can raise `NameError`s or silently reuse stale data.


## Known limitations

- Single random seed per ratio condition; no estimate of run-to-run variance.
- Single model architecture (`distilroberta-base`) and fixed hyperparameters; not swept.
- Synthetic data from one generator (ELSA) only.
- Rare classes (`surprise`, `love`) have limited test-set support, so per-emotion significance tests have low power to detect small effect.

## Citation

If you use this code, please cite it — see [`CITATION.cff`](./CITATION.cff)

## Data use note

DAIR-AI and ELSA are used under their respective licenses/terms for research purposes. DAIR-AI is derived from public Twitter/X posts; ELSA is derived from DAIR-AI via LLM rewriting.
