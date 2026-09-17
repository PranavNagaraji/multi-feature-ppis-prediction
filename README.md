# Multi Feature PPIS Prediction (ESM-2 Feature Fusion Extension)

GTE-PPIS is a structure-based protein-protein interaction site (PPIS) predictor combining a Graph Transformer (GT) and an Equivariant Graph Neural Network (EGNN). This repository extends the published GTE-PPIS model (Wang et al., 2025, *Bioinformatics*) with a **Feature Fusion Module (FFM)** that integrates ESM-2 (650M) protein language model embeddings with classical handcrafted features via a learned, biophysics-supervised gating mechanism.

---

## System Requirements

Developed and tested under Linux with:

| Package | Version |
|---|---|
| Python | 3.10.14 |
| PyTorch | 2.4.0 |
| DGL | 2.4.0+cu121 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.0 |
| transformers | 5.12.1 |
| freesasa | (for RSA feature generation) |

Full pinned dependencies: [`requirements.txt`](requirements.txt)

---

## Architecture Overview

```
Raw Node Features (61d per residue):
  DSSP (14d) + PSSM (20d) + HMM (20d) + resAF (7d)

ESM-2 embeddings (1280d, pre-extracted)
         │
         ▼
Feature Fusion Module (FFM)  ← --fusion_mode {none|concat|gated}
  • none:   bypass, use classical 61d features as-is
  • concat: project each to d=128, concatenate → 256d + DSSP+resAF = 277d
  • gated:  scalar gate gᵢ = σ(Linear([c_proj, p_proj]→1))
            fused = gᵢ·c_proj + (1-gᵢ)·p_proj → 128d + DSSP+resAF = 149d
            (supervised by RSA target: gᵢ* = 1 - RSAᵢ)
         │
         ▼
    ┌────┴────┐
    │         │
  EGNN(10L)  GT(4L, residual=True)
    │         │
    └────┬────┘
       avg
         │
       output logits (2-class)
```

**Loss**: Focal Loss (`gamma`, default 2.0) + optional auxiliary losses:
- `--lambda_gate 0.1`: RSA-supervised gate MSE loss
- `--lambda_agree 0.1`: EGNN–GT branch agreement regularisation (MSE between branch softmax outputs)

---

## Dataset and Features

Datasets (`Train_335`, `Test_60`, `Test_315-28`, `UBtest_31-6`) are stored in `./Dataset/` as Python pickles:
```python
Dataset[ID] = [sequence_str, label_array]
```

Feature directories under `./Feature/`:

| Subdirectory | Content | Shape |
|---|---|---|
| `dssp/` | DSSP secondary structure | L × 14 |
| `pssm/` | PSSM evolutionary profile | L × 20 |
| `hmm/` | HMM profile | L × 20 |
| `resAF/` | Residue atom features | L × 7 |
| `distance_map_SC/` | Cα–Cβ distance maps | L × L |
| `psepos/` | Pseudo-position coordinates | per-protein pkl |
| `esm2/` | ESM-2 (650M) embeddings (fp16) | L × 1280 |
| `rsa/` | FreeSASA relative solvent accessibility | L (float32) |

Distance maps and PDB files are hosted externally due to GitHub size limits:
- [Distance maps (Google Drive)](https://drive.google.com/drive/folders/1UKX1dIrzrEPQpGyYcvIJlrm6KAKvNEpp?usp=drive_link)
- [PDB files (Google Drive)](https://drive.google.com/drive/folders/1eTjFtxsP4mnzyg5CXs96c3w4AxzYNW5N?usp=drive_link)

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# DGL must be installed separately with the correct CUDA suffix:
# pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

---

## Pre-processing (one-time, required for fusion modes)

**Generate ESM-2 embeddings** (required for `concat`/`gated` modes):
```bash
python generate_esm2_embeddings.py
# Output: Feature/esm2/{ID}.npy (float16, L×1280)
# GPU fp16 with automatic CPU fallback on OOM
```

**Generate RSA features** (required for gated mode RSA supervision):
```bash
python generate_rsa_features.py
# Requires: PDB/ directory populated
# Output: Feature/rsa/{ID}.npy (float32, L-dim, range [0,1])
```

---

## Running GTE-PPIS

### Training

```bash
# Classical backbone (no ESM-2, replicates published GTE-PPIS architecture)
python train.py --fusion_mode none --focal_gamma 2.0

# Gated ESM-2 fusion with RSA supervision and branch regularisation (full proposed method)
python train.py --fusion_mode gated --d_proj 128 --focal_gamma 2.0 \
    --lambda_gate 0.1 --lambda_agree 0.1

# Naive concat fusion
python train.py --fusion_mode concat --d_proj 128 --focal_gamma 2.0
```

Key CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--fusion_mode` | `none` | `none` / `concat` / `gated` |
| `--d_proj` | `128` | Projection dimension for FFM |
| `--focal_gamma` | `2.0` | Focal Loss γ (0 = weighted CE) |
| `--lambda_gate` | `0.1` | RSA gate supervision weight |
| `--lambda_agree` | `0.1` | Branch agreement regularisation weight |
| `--smoke_test` | off | 2-sample, 1-fold, 1-epoch quick check |

Checkpoints and logs saved to: `./Log/fusion_<mode>_d<d_proj>_<timestamp>/`

### Testing

```bash
python test.py \
    --fusion_mode gated \
    --d_proj 128 \
    --model_dir Log/fusion_gated_d128_<timestamp>/model/
```

Evaluates on `Test_60`, `Test_315-28`, and `UBtest_31-6`. Threshold is locked from the validation fold (no test-label leakage). In `gated` mode, per-residue gate values are written to `<model_dir>/<checkpoint>_gate_records.csv`.

---

## Directory Structure

```
GTE-PPIS/
├── train.py                   # Training entry point (5-fold CV + full model)
├── test.py                    # Evaluation entry point (all three test sets)
├── final_model.py             # FinalModel: EGNN + GT branches, FFM integration
├── fusion_module.py           # FeatureFusionModule (none / concat / gated)
├── loss.py                    # FocalLoss + compute_pos_weight
├── data_generator.py          # ProDataset, graph construction, collate function
├── EGNN_model.py              # Equivariant GNN (E-GCL blocks, 10 layers)
├── GraphTransformer_Block.py  # Graph Transformer (4 layers, residual=True)
├── generate_esm2_embeddings.py  # One-time ESM-2 embedding extraction
├── generate_rsa_features.py     # One-time RSA feature extraction (FreeSASA)
├── generate_psepos.py           # Pseudo-position coordinate generation
├── run_experiments.sh           # Sequential experiment runner script
├── requirements.txt
├── Dataset/                   # Train_335, Test_60, Test_315-28, UBtest_31-6 pickles
├── Feature/                   # All per-protein feature arrays
│   ├── dssp/, pssm/, hmm/, resAF/   # Classical handcrafted features
│   ├── distance_map_SC/             # Distance matrices
│   ├── psepos/                      # Pseudo-position pickles
│   ├── esm2/                        # ESM-2 embeddings (generated)
│   └── rsa/                         # RSA values (generated)
├── PDB/                       # PDB structure files (external download)
└── Log/                       # Training runs (auto-created)
    └── fusion_<mode>_d<d>_<timestamp>/
        ├── training.log
        └── model/             # Fold1..5 + Full model checkpoints
```

---

## Current Experimental Status

| Run | Mode | `gamma` | GT residual | Gate supervision | Test_60 AUROC | Test_60 AUPRC | UBtest AUROC | UBtest AUPRC |
|---|---|---|---|---|---|---|---|---|
| No-ESM2 baseline (Aug 25) | `none` | 2.0 | True | — | 0.8537 | 0.5518 | 0.7611 | 0.3115 |
| Unsupervised gate (Aug 13) | `gated` | 2.0 | False† | None | 0.8236 | 0.5056 | 0.8191 | 0.4182 |
| RSA-supervised gate (Sep 11) | `gated` | 0.0 | True | RSA-MSE λ=0.1 | 0.8116 | 0.4863 | 0.8159 | 0.4248 |

† Aug 13 model trained before `transformer_residual=True` fix (commit `d0c93b5`). All results use validation-locked thresholds (no test-label leakage). Published GTE-PPIS paper baseline: Test_60 AUROC 0.873, AUPRC 0.611 (Wang et al. 2025).

See [`rsa_supervision_ablation_report.tex`](rsa_supervision_ablation_report.tex) for full per-fold tables and delta analysis.
