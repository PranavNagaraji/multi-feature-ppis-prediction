# GTE-PPIS Modified — Project Context
**Last updated: 2026-10-01**

---

## 1. Who

B.Tech 4th-year (final-year) student. This is the B.Tech final-year project.

---

## 2. Project Overview

A research project built on top of **GTE-PPIS** (Wang et al., *Bioinformatics* 2025, PMC12199915), a graph-based dual-branch model (EGNN + Graph Transformer) for residue-level Protein-Protein Interaction Site (PPIS) prediction.

The core novel contribution is a **Feature Fusion Module (FFM)** that learns to combine two evolutionary/sequence feature streams — handcrafted PSSM+HMM and pre-trained ESM-2 650M embeddings — using learned gating before the GNN branches process node features. Two additional **auxiliary biophysical training objectives** further guide the gate using structural priors.

---

## 3. Architecture — Current Actual Implementation

### Input node features (61d total, unchanged from GTE-PPIS)
- DSSP: 14d (structural/secondary structure)
- PSSM: 20d (evolutionary, handcrafted)
- HMM: 20d (evolutionary, handcrafted)
- resAF: 7d (residue atom features)

### Feature Fusion Module (FFM) — `fusion_module.py`

Placed **between** raw node features and GNN branches. Acts **only on the evolutionary streams** (PSSM 20d + HMM 20d = 40d), leaving DSSP and resAF untouched.

**Input streams per residue i:**
1. `classical_i`: PSSM+HMM (40d)
2. `plm_i`: ESM-2 650M per-residue embeddings (1280d), cached as float32 from float16

**Projection (shared dimension `d_proj=128`):**
- `classical_proj = LayerNorm(Linear(40→128)(classical_i))`
- `plm_proj = LayerNorm(Linear(1280→128)(plm_i))`
- Both: Xavier uniform init (gain=1.0), zero bias

**Fusion modes (selectable via `--fusion_mode`):**

| Mode | Computation | Output dim |
|---|---|---|
| `none` | FFM bypassed entirely | 61d (original) |
| `concat` | `[c_proj ‖ p_proj]` → 256d | 256 + 14 + 7 = **277d** |
| `gated` | `g_i = σ(Linear(256→1))`, `f_i = g_i·c_proj + (1-g_i)·p_proj` | 128 + 14 + 7 = **149d** |

> `cross_attn` mode was **removed** (commit `d6864e3`, Sep 12 2026) — confirmed unused in any production run and had a silent `gate_val=None` bug.

**Re-concatenation:** Final node features = `[fused_i ‖ DSSP_i ‖ resAF_i]`

### GNN Branches (`final_model.py`)
- **EGNN** (`EGNN_model.py`): 10 layers, `residual=True`, `tanh=True`, `attention=True`
- **Graph Transformer** (`GraphTransformer_Block.py`): 4 layers, `transformer_residual=True`
- Final prediction: element-wise average of EGNN and GT logits → `(x1 + x2) / 2`

### Loss — `loss.py` + `final_model.py`

```
L_total = L_focal + λ_gate · L_gate + λ_agree · L_agree
```

- **L_focal**: Focal Loss (Lin et al. 2017), α = [1.0, neg/pos ratio per fold], γ configurable (`--focal_gamma`, default 2.0; γ=0 → weighted CE)
- **L_gate** (Idea 1, Biophysics-Supervised Gate): MSE between predicted gate and `τ_i = 1 - [α_RSA · RSA_i + (1 - α_RSA) · B_norm_i]`
  - Buried residues (low RSA, low B-factor) → τ→1 → trust classical features
  - Surface/flexible residues (high RSA or B-factor) → τ→0 → trust PLM features
  - `ALPHA_RSA` in `final_model.py` (line 27): currently `0.5` (joint RSA+Bfactor supervision)
- **L_agree** (Idea 2, Branch Agreement Regularisation): MSE between softmax(EGNN logits) and softmax(GT logits)

### Key constants (canonical values as of Oct 1, 2026)
| Constant | Value | Source |
|---|---|---|
| `LEARNING_RATE` | `1e-4` | `data_generator.py` L38, explicitly re-set in `final_model.py` L15 and `train.py` L23 (wildcard import collision fix) |
| `WEIGHT_DECAY` | `1e-4` | `data_generator.py` L39 |
| `SEED` | `2024` | `data_generator.py` L15 |
| `BATCH_SIZE` | `1` | `data_generator.py` L40 |
| `NUMBER_EPOCHS` | `50` | `data_generator.py` L42 |
| `MAP_CUTOFF` | `14` Å | `data_generator.py` L28 |
| `ALPHA_RSA` | `0.5` | `final_model.py` L27 |
| `_DEFAULT_POS_WEIGHT` | `5.4056` | `final_model.py` L21 |

---

## 4. Datasets

| Dataset | Description | Residues | % Positive |
|---|---|---|---|
| `Train_335.pkl` | Training set (335 proteins, −2j3rA → 334 used) | 66,208 | ~15.6% |
| `Test_60.pkl` | Bound test set | 13,144 | ~15.6% |
| `Test_315-28.pkl` | Larger bound test set | 60,376 | — |
| `UBtest_31-6.pkl` | **Unbound** test set (harder) | 5,917 | — |

- `2j3rA` excluded from Train_335 (ESM-2 embedding unavailable at time of extraction).
- Cross-validation: 5-fold, SEED=2024.

---

## 5. Feature Files (on disk)

| Feature | Directory | Generator script |
|---|---|---|
| DSSP | `Feature/dssp/` | — (pre-existing) |
| PSSM | `Feature/pssm/` | — (pre-existing) |
| HMM | `Feature/hmm/` | — (pre-existing) |
| resAF | `Feature/resAF/` | — (pre-existing) |
| Distance maps | `Feature/distance_map_SC/` | — (pre-existing) |
| Pseudopositions | `Feature/psepos/` | `generate_psepos.py` |
| ESM-2 embeddings | `Feature/esm2/` | `generate_esm2_embeddings.py` |
| RSA | `Feature/rsa/` | `generate_rsa_features.py` (FreeSASA) |
| B-factor | `Feature/bfactor/` | `generate_bfactor_features.py` |

**RSA fallback:** 0.5 for proteins without PDB (`2j3rA` is the only known case in Train_335).

**B-factor fallback (0.5):** 4 proteins — `3zeuD` (PDB/sequence length mismatch, 6 interior gaps), `1cdbA`, `1ci5A`, `1qndA` (all-zero B-factor distributions, degenerate).

---

## 6. Evaluation Protocol

- 5-fold cross-validation
- **Validation-locked threshold**: threshold chosen on validation fold (maximises val F1), then locked before evaluating test set. No test-label leakage.
- Primary metrics: AUPRC, MCC, AUROC
- Secondary: F1, Precision, Recall, Accuracy

---

## 7. CLI Flags (train.py / test.py)

| Flag | Default | Description |
|---|---|---|
| `--fusion_mode` | `none` | `none` / `concat` / `gated` |
| `--d_proj` | `128` | Shared projection dimension |
| `--focal_gamma` | `2.0` | Focal Loss γ (0 = weighted CE) |
| `--lambda_gate` | `0.1` | RSA+Bfactor gate supervision weight |
| `--lambda_agree` | `0.1` | Branch agreement regularisation weight |
| `--smoke_test` | off | 2-sample, 1-fold, 1-epoch verification |
| `--model_time` | — | train.py: override log folder name |
| `--model_dir` | — | test.py: path to checkpoint directory (required) |

---

## 8. Comparison Baselines (published)

**GTE-PPIS paper** (Wang et al., *Bioinformatics* 2025, Table 2–3, PMC12199915):
- Test_60: AUROC=0.873, AUPRC=0.611, MCC=0.471 (paper's MCC=0.500 uses a different threshold protocol)
- Test_315-28: AUPRC=0.598, MCC=0.511

**DHEG** (*Briefings in Bioinformatics* 2026):
- Test_60: AUROC=0.900, AUPRC=0.686, MCC=0.580, F1=0.648

**Important caveat:** Paper threshold protocols are unconfirmed — they may include test-label tuning. Our results use strictly validation-locked thresholds.

---

## 9. Working Rules for Antigravity

1. **Repository files and logs are the source of truth.** This context file is background, not ground truth.
2. Before answering questions about metrics, architecture, or training state — inspect actual files/logs.
3. If this file contradicts the actual code, trust the code and explain the discrepancy.
4. Never fabricate metrics, experiments, or conclusions.
5. Make minimal changes. Do not rename variables unnecessarily. Do not rewrite entire files unless explicitly requested.
6. Clearly separate: observed result vs expected behaviour vs hypothesis/speculation.
7. Early stopping is **currently commented out** in `train.py` (lines 245, 275–277 as of the last ablation setup).
