# SMM4H-HeaRD 2026 — Task 1 (ADE Detection) Project Report

## 1) Problem statement
We worked on **SMM4H-HeaRD 2026 Task 1**, a **binary text classification** problem:

- **Input**: a short text (social/media-style) with a **language code**.
- **Output**: `predicted_label ∈ {0,1}` where **1 = ADE (Adverse Drug Event mentioned)**, 0 = not ADE.
- **Official ranking metric**: **macro-F1 averaged *unweighted* across languages** (each language contributes equally, regardless of sample count).

This repo contains a complete training + evaluation + inference pipeline to train multilingual transformer models, calibrate them, and generate a submission zip.

## 2) Datasets used and how we split them
### 2.1 Main shared-task data (primary)
Files (workspace root):
- `train_data_SMM4H_2026_Task_1.csv` (main training set)
- `dev_data_SMM4H_2026_Task_1.csv` (main dev set used for evaluation and calibration)

We **do not** split dev further; it is used as the consistent held-out evaluation set.

### 2.2 CADEC translated data (auxiliary pretraining)
Files (workspace root):
- `train_data_cadec_translated.csv`
- `dev_data_cadec_translated.csv`

In our training code this is used as **Stage 1 pretraining** (domain adaptation), before fine-tuning on the main shared-task data.

### 2.3 Cross-validation on main train
For robust training and checkpoint ensembling, Stage 2 trains with **stratified K-fold CV** on the main training set:

- **# folds**: 5 (`n_folds = 5`)
- **Stratification target**: **(label, language)** combined, so each fold preserves label balance *within* each language when possible.
- **Fallback**: if some (label, language) strata are too small for 5-fold, it falls back to stratifying on **label only**.

Implementation reference:
- `smm4h_ade/data_pipeline.py`: `kfold_indices()` and `stratify_labels_for_kfold()`
- `smm4h_ade/train.py`: Stage 2 loop over folds

## 3) Languages considered
### 3.1 Supported language codes in codebase
`smm4h_ade/config.py` defines:

- `SUPPORTED_LANGS = ("de", "fr", "ru", "en", "zh", "ja", "fa")`
- plus an additional **unknown/OOD bucket** used internally for language embeddings.

Notes:
- **`fa` (Farsi)** is included to support potential **zero-shot test** language handling.
- The dev results we currently report are for the languages present in the provided dev CSV: **de, en, fr, ja, ru, zh**.

### 3.2 Language handling features we used
We used multiple language-aware mechanisms:

- **Language embedding**: a small learned embedding per language (plus an `unk` row), projected and added to the CLS pooled representation.
- **Metadata tokens** (optional): prefix tokens like `[LANG_xx]`, `[TYPE_*]`, `[ORIGIN_*]` if such columns exist.
- **Prompt formatting** (optional): an ADE-oriented instruction prompt wrapper.
- **Reasoning format** (optional): "Text:/Reason:" formatting (without leaking labels).

These toggles appear in `smm4h_ade/config.py` and are wired through training/eval/infer.

## 4) Model architectures used
### 4.1 Backbone transformers
We used two multilingual transformer backbones:

1) **XLM-RoBERTa Large** (`xlm-roberta-large`)
- Used as the main high-capacity multilingual model.
- Trained with **5-fold CV**; we keep `fold_0 … fold_4` checkpoints and ensemble them at inference/evaluation time.

2) **mDeBERTa v3 base** (`microsoft/mdeberta-v3-base`)
- Trained as an additional backbone for diversity (used in later-stage ensembling experiments).

### 4.2 Classification head (our architecture on top of the backbone)
Core model (`smm4h_ade/modeling.py`, class `ADEClassifier`):

- Input: tokenized text
- Backbone: HuggingFace `AutoModel`
- Pooling: **CLS token** (`last_hidden_state[:,0]`)
- Optional: **language embedding** (lang-id → embedding → linear projection → added to pooled vector)
- Then: LayerNorm → Dropout → Linear classifier → 2 logits

### 4.3 Parameter-efficient fine-tuning (LoRA)
We used **LoRA** via `peft` (`smm4h_ade/modeling.py: apply_lora()`):

- Targets attention projection layers:
  - RoBERTa/XLM-R style: `query`, `value`
  - DeBERTa style: `query_proj`, `value_proj`
- Default hyperparameters in config:
  - `r = 16`, `alpha = 32`, `dropout = 0.05`

Why: this lets us fine-tune strong backbones with fewer trainable parameters and typically better stability under limited compute.

## 5) Training approach (what we did)
Training code lives in `smm4h_ade/train.py` and is organized as:

### Stage 1 — CADEC pretraining (domain adaptation)
- Train on `train_data_cadec_translated.csv`
- Validate on `dev_data_cadec_translated.csv`
- Goal: adapt the backbone to ADE-style language and improve downstream transfer.

### Stage 2 — Main shared-task training with CV
- Train on `train_data_SMM4H_2026_Task_1.csv` using **5-fold stratified CV**.
- Evaluate each fold on **main dev** (`dev_data_SMM4H_2026_Task_1.csv`) using the official metric (macro-F1 unweighted by language).
- Save best checkpoint per fold as `fold_k/best.pt` with `fold_k/metrics.json`.

### Sampling / imbalance handling
The training loader supports:
- **Weighted sampling** (default enabled): positives get higher sampling weight than negatives (see `_build_train_loader()` in `smm4h_ade/train.py`).
- Optional language-balancing samplers (round-robin / inverse frequency) are implemented but not always enabled in runs.

## 6) Calibration and decision thresholds (important for the leaderboard metric)
Because the leaderboard metric is **language-macro-F1**, the best decision rule is usually not a single global 0.5 threshold.

We implemented (and used in our final stack):

### 6.1 Temperature scaling
In `smm4h_ade/evaluate.py`:
- Fit temperature on dev logits:
  - **Global** temperature, or
  - **Per-language** temperature (recommended when languages differ in calibration)

### 6.2 Threshold tuning
Also in `smm4h_ade/evaluate.py` (and used in step4 scripts):
- Tune **global threshold** that maximizes official macro-F1.
- Tune **per-language thresholds** that maximize official macro-F1 (language-wise).

These calibration artifacts are stored as JSON in a calibration output directory.

## 7) Ensembling (how many models we used)
### 7.1 In-fold / across-fold ensemble (XLM-R)
We used **5 checkpoints** from 5-fold CV for XLM-R:
- `results/train_fullstack_20260411_130700/fold_0/best.pt`
- `results/train_fullstack_20260411_130700/fold_1/best.pt`
- `results/train_fullstack_20260411_130700/fold_2/best.pt`
- `results/train_fullstack_20260411_130700/fold_3/best.pt`
- `results/train_fullstack_20260411_130700/fold_4/best.pt`

Ensemble method: **mean of logits** (then calibrated + thresholded).

### 7.2 Cross-backbone ensemble (XLM-R + mDeBERTa)
We also ran a **two-model ensemble experiment**:
- XLM-R dev logits + mDeBERTa dev logits
- Searched a weight \(w\in[0,1]\) to combine probabilities / logits and maximize dev macro-F1.

Pipeline driver:
- `results/step4_ensemble_bg.sh`

## 8) Results (accuracy / F1)
Important: the shared task uses **macro-F1** (unweighted by language). “Accuracy” is not the official metric, so we report the official metric first.

### 8.1 Best dev official metric (full-stack XLM-R setup)
From:
- `results/submission_fullstack_20260411/dev_eval_official_metric.json`

**Official dev macro-F1 (unweighted by language): 0.7882**

Per-language breakdown (dev):

| language | precision | recall | F1 | n |
|----------|-----------|--------|----|---|
| de | 0.8667 | 0.7429 | 0.8000 | 634 |
| en | 0.7313 | 0.8033 | 0.7656 | 888 |
| fr | 1.0000 | 0.9667 | 0.9831 | 418 |
| ja | 0.5393 | 0.6667 | 0.5963 | 3045 |
| ru | 0.7216 | 0.6765 | 0.6983 | 2669 |
| zh | 0.8537 | 0.9211 | 0.8861 | 379 |
| **macro (unweighted)** | 0.7854 | 0.7962 | **0.7882** | 8033 |

### 8.2 Example single-fold checkpoint score (for context)
Example:
- `results/train_fullstack_20260411_130700/fold_0/metrics.json` shows **macro-F1 ≈ 0.7172** (single fold, before full calibration/stacking).

## 9) End-to-end pipeline (how to reproduce)
### 9.1 Training
Entry point:
- `python3 train.py ...` (wrapper that calls `smm4h_ade/train.py`)

This runs:
- Stage 1 (CADEC) then Stage 2 (main train, K-fold), producing `fold_k/best.pt` and `metrics.json`.

### 9.2 Dev evaluation + calibration bundle
Entry point:
- `python3 evaluate.py --calibration_bundle ...`

This exports:
- temperature JSON(s)
- best threshold JSON(s)
- metrics + error analysis + confusion matrices

### 9.3 Test inference + submission zip
Entry point:
- `python3 scripts/build_submission_zip.py ...`

This runs `infer.py` and produces:
- `submission.csv` (id, predicted_label)
- `submission.csv.zip` suitable for upload

## 10) Notes / practical engineering work we did
- **Language robustness**: included a language-embedding table with an `unk` bucket; optional fastText language validation.
- **Better ranking metric alignment**: tuned thresholds (and optionally temperature) to directly maximize **macro-F1 across languages**.
- **Compute constraints**: used **LoRA** and cautious checkpoint loading patterns for stability during evaluation.

