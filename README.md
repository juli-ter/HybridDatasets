# Hybrid Datasets for Emotion Classification

Code accompanying the paper **"Hybrid Datasets for Emotion Classification: How Much Synthetic Data Is Too Much?"** (Maznyk & Terekhova).

This repository investigates how the proportion of synthetic (LLM-generated) training data affects the performance of a text-based emotion classifier, by training DistilRoBERTa-base on hybrid datasets mixed from real and synthetic sources at five real/synthetic ratios (100/0, 75/25, 50/50, 25/75, 0/100), evaluated only on real data.

> This project was developed as part of the seminar **EAITHD26** at **Osnabrück University**, Summer Semester 2026.

## Data

- **Real data:** [DAIR-AI Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) — English tweets labeled with one of six emotions (anger, fear, joy, love, sadness, surprise).
- **Synthetic data:** [ELSA](https://huggingface.co/datasets/joyspace-ai/ELSA-Emotion-and-Language-Style-Alignment-Dataset) — LLM-generated rewrites of DAIR-AI texts in four styles; this study uses the *conversational* style.

Neither dataset is bundled in this repo, both are downloaded automatically via the `datasets` library when the notebook is run.

## Running the experiments

Open `Hybrid_datasets.ipynbb` and run all cells top to bottom. This will:

1. Load DAIR-AI and ELSA, and split the real data into training/validation/test sets (stratified, `random_state=42`).
2. Build five hybrid training sets of 10,000 samples each, at real/synthetic ratios of 100/0, 75/25, 50/50, 25/75, and 0/100.
3. Fine-tune DistilRoBERTa-base on each mixture (5 runs per condition) and evaluate on the real-only test set.
4. Reproduce the tables and figures reported in the paper.

Random seeds are fixed for data sampling and splitting; results may vary slightly between runs due to non-determinism in model training.

## Citation

If you use this code, please cite it — see [`CITATION.cff`](./CITATION.cff)

## Data use note

DAIR-AI and ELSA are used under their respective licenses/terms for research purposes. DAIR-AI is derived from public Twitter/X posts; ELSA is derived from DAIR-AI via LLM rewriting.
