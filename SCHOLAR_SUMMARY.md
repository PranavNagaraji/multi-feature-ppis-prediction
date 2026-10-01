# GTE-PPIS Feature Fusion — Research Scholar Summary
**Last updated: 2026-10-01 | Based on actual log files on disk**

---

## 1. Research Problem and Motivation

**Task:** Residue-level Protein-Protein Interaction Site (PPIS) prediction.

**Base model:** GTE-PPIS (Wang et al., *Bioinformatics* 2025, PMC12199915) — a dual-branch GNN (EGNN + Graph Transformer) operating on a protein residue-contact graph.

**Specific gap addressed:** Prior work (e.g., DHEG, *Briefings in Bioinformatics* 2026) showed that naively concatenating or substituting PSSM/HMM handcrafted features with ESM-2 PLM embeddings underperforms the handcrafted-only baseline (MCC as low as 0.086 for standalone PLMs on PPIS). The open question is whether a **learned fusion mechanism** — rather than static concatenation or replacement — enables PLM embeddings to contribute a genuinely complementary signal, and whether **biophysical structural priors** (residue solvent accessibility, crystallographic B-factors) can meaningfully guide that mechanism.

---

## 2. Architectural Contribution — Feature Fusion Module (FFM)

The FFM is inserted between raw node features and both GNN branches. It acts exclusively on the evolutionary streams (PSSM 20d + HMM 20d = 40d classical, plus ESM-2 1280d), leaving structural streams (DSSP 14d, resAF 7d) untouched.

### Fusion modes

| Mode | Computation | Node dim into GNNs |
|---|---|---|
| `none` | FFM bypassed | 61d |
| `concat` | `[c_proj ‖ p_proj]` | 277d |
| `gated` | `g_i = σ(Linear([c,p]→1))`, `f_i = g_i·c + (1-g_i)·p` | 149d |

Both streams projected to shared d=128, followed by LayerNorm. Xavier uniform init.

---

## 3. Novel Auxiliary Training Objectives

### Idea 1 — Biophysics-Supervised Gate Loss (λ_gate)

The gate is taught to encode structural burial information:

$$\tau_i = 1 - [\alpha_\text{RSA} \cdot \text{RSA}_i + (1 - \alpha_\text{RSA}) \cdot B^\text{norm}_i]$$
$$\mathcal{L}_\text{gate} = \frac{1}{N}\sum_i (\hat{g}_i - \tau_i)^2$$

- **RSA_i**: Relative Solvent Accessibility, computed via FreeSASA from PDB structures.
- **B_norm_i**: Per-residue normalised crystallographic B-factor (percentile-robust: p5/p95 normalisation). Encodes thermal flexibility / crystallographic uncertainty.
- `ALPHA_RSA = 0.5`: Joint RSA+Bfactor target (current setting, `final_model.py` L27).
- Interpretation: buried+rigid residues → τ→1 → trust classical PSSM/HMM; surface+flexible residues → τ→0 → trust ESM-2.
- **B-factor novelty confirmed** (Sep 11 novelty search): no prior PPIS work uses crystallographic B-factors as a reliability gate for PLM–structure fusion.

### Idea 2 — Branch Agreement Regularisation (λ_agree)

$$\mathcal{L}_\text{agree} = \text{MSE}(\text{softmax}(\mathbf{x}_\text{EGNN}),\; \text{softmax}(\mathbf{x}_\text{GT}))$$

Discourages the two GNN branches from making contradictory predictions, encouraging consensus on interface residue identity.

### Total loss

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{focal} + \lambda_\text{gate} \cdot \mathcal{L}_\text{gate} + \lambda_\text{agree} \cdot \mathcal{L}_\text{agree}$$

L_focal = Focal Loss (Lin et al. 2017), α = [1.0, neg/pos ratio computed per fold], γ configurable (default 2.0; γ=0 → weighted CE).

---

## 4. Quantitative Results

**All numbers below are 5-fold CV averages, validation-locked threshold protocol.**
Source: actual log files on disk. Numbers from `reference_test_logs/` (Aug runs) and `Log/` (Sep–Oct runs).

### Test_60 (bound, 13,144 residues)

| System | Config | AUROC | AUPRC | MCC | F1 |
|---|---|---|---|---|---|
| GTE-PPIS (Wang et al. 2025)† | — | 0.873 | 0.611 | 0.471 | 0.582 |
| DHEG (*Briefings in Bioinf.* 2026)† | — | 0.900 | 0.686 | 0.580 | 0.648 |
| **Ours: No-ESM2 backbone** | `none`, γ=2, LR=1e-3 (Aug 25) | **0.848** | **0.538** | **0.440** | **0.532** |
| **Ours: Unsupervised gate** | `gated`, γ=2, LR=1e-3, λ_gate=0 (Aug 13) | **0.824** | **0.506** | **0.411** | **0.508** |
| **Ours: RSA-supervised gate** | `gated`, γ=0, LR=1e-3, λ_gate=0.1, ALPHA=1.0 (Sep 11) | **0.812** | **0.486** | **0.387** | **0.493** |
| **Ours: Joint RSA+Bfactor gate** | `gated`, γ=0, LR=1e-4, ALPHA=0.5, early-stop (Sep 25) | **0.826** | **0.513** | **0.413** | **0.509** |
| **Ours: RSA-only arm 1** | `gated`, γ=2, LR=1e-4, ALPHA=1.0, no early-stop (Oct 1) | **0.806** | **0.480** | **0.379** | **0.485** |

### Test_315-28 (bound, 60,376 residues)

| System | AUROC | AUPRC | MCC | F1 |
|---|---|---|---|---|
| GTE-PPIS (paper)† | — | 0.598 | 0.511 | — |
| No-ESM2 backbone (Aug 25) | 0.859 | 0.518 | 0.435 | 0.518 |
| Unsupervised gate (Aug 13) | 0.812 | 0.438 | 0.360 | 0.459 |
| RSA-supervised gate (Sep 11) | 0.803 | 0.435 | 0.350 | 0.449 |
| Joint RSA+Bfactor gate (Sep 25) | 0.817 | 0.449 | 0.371 | 0.468 |
| RSA-only arm 1 (Oct 1) | 0.806 | 0.428 | 0.348 | 0.450 |

### UBtest_31-6 (unbound, 5,917 residues)

| System | AUROC | AUPRC | MCC | F1 |
|---|---|---|---|---|
| No-ESM2 backbone (Aug 25) | 0.761 | 0.312 | 0.270 | 0.353 |
| Unsupervised gate (Aug 13) | 0.819 | 0.417 | 0.372 | 0.447 |
| RSA-supervised gate (Sep 11) | 0.816 | 0.425 | 0.369 | 0.450 |
| Joint RSA+Bfactor gate (Sep 25) | 0.813 | 0.411 | 0.363 | 0.439 |
| RSA-only arm 1 (Oct 1) | 0.816 | 0.420 | 0.371 | 0.452 |

†Paper numbers from Wang et al. (2025), PMC12199915, Tables 2–3. Paper threshold protocol unconfirmed (may include test-label tuning). Our UBtest_31-6 ≠ paper's UBtest_25 (different set sizes); not directly comparable.

---

## 5. Key Findings So Far

1. **No-ESM2 backbone outperforms our gated fusion on Test_60 and Test_315-28.** The fusion module has not yet matched the baseline on bound test sets.
2. **Gated fusion (both supervised and unsupervised) consistently improves UBtest (unbound) performance** vs no-ESM2 backbone: +0.105 AUPRC, +0.102 MCC on UBtest (unsupervised gate, Aug 13).
3. **RSA gate supervision (Sep 11) does not clearly improve over unsupervised gate.** Three confounders changed simultaneously (RSA supervision ON, `transformer_residual` fix, γ changed 2→0), making direct attribution impossible.
4. **Joint RSA+Bfactor gate (Sep 25, early-stop)** partially recovers Test_60 performance compared to Sep 11, suggesting some benefit of B-factor in the joint target, but still below no-ESM2 backbone on Test_60.
5. **LR=1e-4 with γ=2 (arm 1, Oct 1)** is worse than LR=1e-4 with γ=0 (Sep 25) — focal loss with γ=2 is likely too aggressive on this small dataset.

---

## 6. Active Ablation Plan

| Arm | Config | Status | Log |
|---|---|---|---|
| Arm 1: RSA-only | `gated`, γ=2, LR=1e-4, ALPHA=1.0, no ES | ✅ Complete, tested | `fusion_gated_d128_2026-10-01-13-41-39_rsa_only_lr1e4` |
| Arm 2: Joint RSA+Bfactor | `gated`, γ=0, LR=1e-4, ALPHA=0.5, no ES | 🔄 Training (epoch ~42/50, Oct 1) | `fusion_gated_d128_2026-10-01-21-39-58` |

**Arm 2** isolates the effect of joint RSA+Bfactor supervision at the correct LR and without focal loss — a clean comparison to Arm 1 and the Sep 25 run.

---

## 7. Experiment Status Table (all runs)

| Run | Mode | LR | γ | λ_gate | ALPHA_RSA | ES | Status |
|---|---|---|---|---|---|---|---|
| Aug 25 (no-ESM2) | `none` | 1e-3 | 2.0 | 0 | — | Off | ✅ Complete + tested |
| Aug 13 (unsup gate) | `gated` | 1e-3 | 2.0 | 0 | — | Off | ✅ Complete + tested |
| Sep 11 (RSA-sup) | `gated` | 1e-3 | 0.0 | 0.1 | 1.0 | Off | ✅ Complete + tested |
| Sep 25 (joint bfac+RSA, ES) | `gated` | 1e-3 | 0.0 | 0.1 | 0.5 | On (p=8) | ✅ Complete + tested |
| Oct 1 Arm 1 (RSA-only) | `gated` | 1e-4 | 2.0 | 0.1 | 1.0 | Off | ✅ Complete + tested |
| Oct 1 Arm 2 (joint, γ=0) | `gated` | 1e-4 | 0.0 | 0.1 | 0.5 | Off | 🔄 Training |
| `concat` full run | `concat` | — | — | — | — | — | 🔲 Not run |

---

## 8. Open Issues / Next Steps

1. **Confounder isolation:** Sep 11 changed three axes simultaneously (RSA supervision, transformer_residual fix, γ change). Arm 2 (Oct 1) is the cleanest isolation run.
2. **Multi-seed validation:** All runs use SEED=2024. Observed deltas (ΔAUPRC ≤ 0.03) are within expected single-seed variance. N≥3 seeds needed for significance.
3. **λ sweep:** `--lambda_gate` and `--lambda_agree` both fixed at 0.1 in all runs. Sweep over {0.01, 0.05, 0.1, 0.2} needed.
4. **Gate inspection:** Compute correlation between predicted gate ĝ_i and RSA/B-factor to verify whether auxiliary loss is directing gate behaviour.
5. **`concat` run:** No full training run exists yet.
6. **Gap to baseline:** Best Test_60 AUPRC (0.538, no-ESM2 Aug 25) is 0.073 below GTE-PPIS paper (0.611) and 0.148 below DHEG (0.686). Ablation is not yet finished.

---

## 9. Files Added / Modified vs Original GTE-PPIS

### Added
- `fusion_module.py`: `FeatureFusionModule` (`none`, `concat`, `gated`)
- `loss.py`: `FocalLoss`, `compute_pos_weight`
- `generate_esm2_embeddings.py`: ESM-2 650M per-residue embeddings → `Feature/esm2/`
- `generate_rsa_features.py`: FreeSASA RSA → `Feature/rsa/`
- `generate_bfactor_features.py`: Crystallographic B-factors (percentile-robust norm) → `Feature/bfactor/`

### Modified
- `final_model.py`: FFM integration, `compute_auxiliary_losses`, `ALPHA_RSA`, `LEARNING_RATE` fix, `residual=True`, `transformer_residual=True`
- `data_generator.py`: Loads ESM-2, RSA, B-factor features; updated `graph_collate` (now unpacks 13 items)
- `train.py`: CLI flags, per-fold `pos_weight`, early stopping class, `LEARNING_RATE` re-declaration, `ALPHA_RSA` print
- `test.py`: CLI flags, validation-locked threshold protocol, gate CSV export

### Untouched (do not modify)
- `test_original.py`: Original GTE-PPIS test script (uses old 11-item `graph_collate`, will break with current `ProDataset`)
- `EGNN_model.py`, `GraphTransformer_Block.py`: Unchanged from original GTE-PPIS
