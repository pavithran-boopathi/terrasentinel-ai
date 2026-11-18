# Experiments

Every row here must link to a reproducible run (git commit, config file, MLflow run ID, data split).
No results are recorded yet: no model has been trained.

## Template

### EXP-NNN — <short title>
- **Date:**
- **Commit / config:** `<sha>` · `ml/configs/<file>.yaml`
- **MLflow run:** `<run_id>`
- **Data:** dataset, split (official / geographic hold-out), # chips train/val/test
- **Model:** architecture, params, input bands, pretrained weights
- **Training:** device, batch size, epochs (early-stopped at), loss, LR schedule, augmentation, seed
- **Results (test, flood class, nodata ignored):**

| Split | IoU | Dice | Precision | Recall | F1 | Latency (ms / 512² chip, device) |
|---|---|---|---|---|---|---|

- **Observations / failure cases:**
- **Next:**

## Log

## EXP-001 — Classical baselines vs. U-Net (from scratch), Sen1Floods11 S1 hand-labeled

- **Date:** 2026-11-18
- **Code:** U-Net runs at commit `1a9aecf` (first run at `5024de1`; ML code identical — `git diff 5024de1 1a9aecf -- ml/` is empty). Runs flagged "dirty" had only `scripts/summarize_experiments.py` modified, not `ml/`. Commit messages were rewritten on 2026-11-18 (contents unchanged), so run metadata in `results.json` / MLflow records the earlier IDs: `d1c8708` = `1a9aecf`, `d2c099e` = `5024de1`.
- **Configs:** `ml/configs/unet_official.yaml`, `ml/configs/unet_geo_holdout.yaml`; seeds 42, 1, 2
- **MLflow:** experiment `sen1floods11-segmentation` in `data/mlflow/mlflow.db` (local)
- **Data:** Sen1Floods11 v1.1 hand-labeled S1 (VV, VH dB). `official` split sha `0511f84471e9` (train 252 / val 89 / test 89 / Bolivia 15; Paraguay_34417 excluded — all-NaN S1). `geo_holdout_v1` sha `ae7f157c7f23` (train 289 / val 58 Pakistan+Spain / test 98 Bolivia+Mekong+Ghana).
- **Model:** U-Net, 4 levels, base 32 channels, decoder dropout 0.1, 7,762,753 params, random init, 2-band input
- **Training:** Apple M5 Pro, MPS, batch 8, random 256² crops + dihedral flips/rot90 + light Gaussian noise, AdamW lr 1e-3 wd 1e-4, 3-epoch warmup + cosine, BCE (pos_weight = sqrt(dry/water) = 3.08–3.10) + soft Dice, ignore index 255, early stopping on val IoU (patience 15), threshold 0.5 (not tuned), ~4.6 s/epoch, full 512² chips at evaluation
- **Baselines:** per-chip Otsu on VH; single VH dB threshold grid-searched for train IoU; gradient-boosted trees (HistGradientBoosting) on VV, VH, VV−VH and 5×5 / 15×15 local mean/std, 3,000 px per train chip; and the dataset's own `S1OtsuLabelHand` masks as a **reference only** (likely produced with more context than a single chip — not a like-for-like competitor)
- **Metrics:** flood-class IoU / F1 / precision / recall pooled over all valid pixels of a split; 95% CI by bootstrap over chips (2,000 resamples)

### Results

#### `official`

| Method | Split | IoU [95% CI] | F1 | Precision | Recall | Chips |
|---|---|---|---|---|---|---|
| otsu_vh_per_chip | val | 0.220 [0.159, 0.287] | 0.361 | 0.236 | 0.760 | 89 |
| otsu_vh_per_chip | test | 0.231 [0.163, 0.304] | 0.376 | 0.253 | 0.729 | 89 |
| otsu_vh_per_chip | bolivia | 0.404 [0.189, 0.603] | 0.576 | 0.432 | 0.863 | 15 |
| dataset_s1_otsu_labels | val | 0.507 [0.406, 0.599] | 0.673 | 0.634 | 0.716 | 89 |
| dataset_s1_otsu_labels | test | 0.554 [0.426, 0.660] | 0.713 | 0.682 | 0.748 | 89 |
| dataset_s1_otsu_labels | bolivia | 0.718 [0.517, 0.824] | 0.836 | 0.803 | 0.870 | 15 |
| global_vh_threshold(-23.75dB) | val | 0.491 [0.403, 0.569] | 0.659 | 0.717 | 0.610 | 89 |
| global_vh_threshold(-23.75dB) | test | 0.530 [0.419, 0.616] | 0.692 | 0.754 | 0.640 | 89 |
| global_vh_threshold(-23.75dB) | bolivia | 0.522 [0.371, 0.612] | 0.686 | 0.873 | 0.565 | 15 |
| pixel_hgb | val | 0.598 [0.493, 0.688] | 0.749 | 0.886 | 0.648 | 89 |
| pixel_hgb | test | 0.639 [0.502, 0.752] | 0.780 | 0.880 | 0.700 | 89 |
| pixel_hgb | bolivia | 0.438 [0.294, 0.518] | 0.609 | 0.921 | 0.455 | 15 |
| U-Net (unet-official-20261118-135044) | val | 0.619 [0.527, 0.689] | 0.764 | 0.754 | 0.775 | 89 |
| U-Net (unet-official-20261118-135044) | bolivia | 0.657 [0.470, 0.769] | 0.793 | 0.831 | 0.758 | 15 |
| U-Net (unet-official-20261118-135044) | test | 0.646 [0.537, 0.733] | 0.785 | 0.756 | 0.817 | 89 |
| U-Net (unet-official-20261118-140345) | test | 0.651 [0.530, 0.745] | 0.788 | 0.801 | 0.776 | 89 |
| U-Net (unet-official-20261118-140345) | val | 0.619 [0.525, 0.691] | 0.764 | 0.791 | 0.740 | 89 |
| U-Net (unet-official-20261118-140345) | bolivia | 0.553 [0.338, 0.707] | 0.713 | 0.846 | 0.615 | 15 |
| U-Net (unet-official-20261118-141253) | test | 0.637 [0.520, 0.729] | 0.778 | 0.753 | 0.805 | 89 |
| U-Net (unet-official-20261118-141253) | val | 0.609 [0.515, 0.682] | 0.757 | 0.746 | 0.768 | 89 |
| U-Net (unet-official-20261118-141253) | bolivia | 0.718 [0.535, 0.807] | 0.836 | 0.844 | 0.828 | 15 |

#### `geo_holdout_v1`

| Method | Split | IoU [95% CI] | F1 | Precision | Recall | Chips |
|---|---|---|---|---|---|---|
| otsu_vh_per_chip | val | 0.220 [0.136, 0.312] | 0.361 | 0.237 | 0.759 | 58 |
| otsu_vh_per_chip | test | 0.279 [0.193, 0.362] | 0.436 | 0.305 | 0.764 | 98 |
| dataset_s1_otsu_labels | val | 0.342 [0.221, 0.471] | 0.510 | 0.400 | 0.703 | 58 |
| dataset_s1_otsu_labels | test | 0.687 [0.596, 0.755] | 0.815 | 0.914 | 0.735 | 98 |
| global_vh_threshold(-23.25dB) | val | 0.431 [0.296, 0.556] | 0.602 | 0.526 | 0.705 | 58 |
| global_vh_threshold(-23.25dB) | test | 0.644 [0.554, 0.708] | 0.783 | 0.912 | 0.687 | 98 |
| pixel_hgb | val | 0.580 [0.418, 0.713] | 0.734 | 0.736 | 0.733 | 58 |
| pixel_hgb | test | 0.664 [0.559, 0.744] | 0.798 | 0.933 | 0.698 | 98 |
| U-Net (unet-geo-holdout-20261118-135703) | test | 0.663 [0.552, 0.751] | 0.797 | 0.929 | 0.699 | 98 |
| U-Net (unet-geo-holdout-20261118-135703) | val | 0.596 [0.430, 0.730] | 0.747 | 0.777 | 0.720 | 58 |
| U-Net (unet-geo-holdout-20261118-140936) | test | 0.644 [0.525, 0.732] | 0.783 | 0.922 | 0.681 | 98 |
| U-Net (unet-geo-holdout-20261118-140936) | val | 0.572 [0.410, 0.701] | 0.728 | 0.725 | 0.731 | 58 |
| U-Net (unet-geo-holdout-20261118-141659) | val | 0.552 [0.402, 0.683] | 0.712 | 0.656 | 0.778 | 58 |
| U-Net (unet-geo-holdout-20261118-141659) | test | 0.720 [0.614, 0.795] | 0.837 | 0.890 | 0.791 | 98 |

#### Across seeds (pooled flood IoU)

| Config | Split | Seeds | Mean | Std | Min | Max |
|---|---|---|---|---|---|---|
| unet-geo-holdout | test | 3 | 0.676 | 0.040 | 0.644 | 0.720 |
| unet-geo-holdout | val | 3 | 0.574 | 0.022 | 0.552 | 0.596 |
| unet-official | bolivia | 3 | 0.643 | 0.083 | 0.553 | 0.718 |
| unet-official | test | 3 | 0.645 | 0.007 | 0.637 | 0.651 |
| unet-official | val | 3 | 0.615 | 0.006 | 0.609 | 0.619 |

Latency (forward pass, 512×512×2, MPS, synchronised, 30 iters after warm-up): **18.5–19.7 ms/chip at batch 1**, 17.3–18.0 ms/chip at batch 8 across the six runs. Baselines on CPU: see `ms_per_chip_cpu` in the baseline `results.json`.

Per-event charts: `docs/figures/per_event_iou_*.png` (seed-42 runs).

### Findings
1. **In-distribution (official test), the U-Net does not clearly beat a strong pixel baseline:** 0.645 ± 0.007 (3 seeds) vs. gradient-boosted pixels 0.639, with heavily overlapping CIs. Per-chip Otsu (0.231) and a global VH threshold (0.530) are clearly worse.
2. **On an unseen event (Bolivia), the U-Net generalises better than the pixel model but is unstable:** 0.643 ± 0.083 vs. 0.438 (pixel GBT) and 0.522 (global threshold). With only 15 chips, seed variance is large (0.553–0.718).
3. **Whole-event hold-out (`geo_holdout_v1`):** U-Net 0.676 ± 0.040 vs. pixel GBT 0.664 and global threshold 0.644 — again a small, not significant margin. Per event, every method does well on Mekong (U-Net 0.83–0.85) and poorly on Ghana (0.36–0.43); Bolivia swings 0.32–0.63 between seeds.
4. **Out-of-event validation is a poor model-selection signal:** the seed with the lowest val IoU (0.552) had the highest test IoU (0.720), and two geo-holdout seeds early-stopped at epochs 10 and 13. Val events (Pakistan, Spain) differ too much from test events.
5. **Errors are mostly missed water, not false alarms:** on held-out events precision is 0.89–0.93 but recall 0.68–0.79. The worst chips (Ghana_141271, Ghana_83200, Bolivia_195474, Bolivia_312675) are narrow channels and labelled water that appears bright rather than dark in SAR — consistent with flooded vegetation (double-bounce) and sub-resolution rivers. This hypothesis is from visual inspection only and is not yet quantified.
6. Water fraction correlates with chip IoU (Pearson r = 0.50–0.69): chips with little water score poorly, so pooled IoU is dominated by water-rich events such as Mekong.

### Limitations
- Small test sets (89 / 98 / 15 chips) → wide CIs; differences of ~0.01–0.03 IoU between the U-Net and the pixel GBT are not meaningful.
- Threshold fixed at 0.5; no calibration. No pretrained encoder. MPS kernels are not bit-deterministic, so seeds fix data order/initialisation but not every floating-point result.
- Hand labels contain large "not valid" regions (up to 34% of pixels for Ghana) which are excluded from all metrics.

### Next
- Threshold selection on val and probability calibration (needed before uncertainty work)
- Pretrained encoder and SegFormer (Phase 4), compared on the same splits and seeds
- Leave-one-event-out CV (Phase 8) to replace the single, noisy out-of-event val split

## EXP-002 — Pretrained encoders and SegFormer vs. U-Net (Phase 4)

- **Date:** 2026-11-18 · **Code:** `e616cb1` (model factory, val-tuned threshold) · configs `ml/configs/{unet_resnet34,segformer_b0,segformer_b2}_{official,geo_holdout}.yaml` · seeds 42/1/2 · MLflow experiment `sen1floods11-segmentation`
- **Data / protocol:** identical to EXP-001 (same splits, crops, augmentation, BCE+Dice, early stopping on val IoU). Pretrained encoders: ImageNet ResNet-34 (smp) and MiT-B0/B2 (`nvidia/mit-b*`); the RGB stem is replaced by a 2-channel conv initialised from the mean RGB filter. lr 3e-4 (ResNet-34) / 2e-4 (SegFormer).
- **Val-tuned threshold:** after training, the threshold maximising val IoU (grid 0.05–0.95) is also reported — selected on validation only. (Phase 3 U-Nets predate this feature.)
- **Generated by:** `scripts/export_experiments_json.py` → `docs/results/experiments.json`

#### `official` — pooled flood IoU, mean ± std over seeds 42/1/2 (threshold 0.5)

| Model | Params | test IoU | bolivia IoU | test IoU @ val-tuned thr. | bolivia IoU @ val-tuned thr. |
|---|---|---|---|---|---|
| U-Net (scratch) | 7.8M | 0.645 ± 0.007 | 0.643 ± 0.083 | — (not run) | — (not run) |
| U-Net ResNet-34 (ImageNet) | 24.4M | 0.648 ± 0.010 | 0.672 ± 0.043 | 0.648 ± 0.011 | 0.672 ± 0.040 |
| SegFormer-B0 | 3.7M | 0.652 ± 0.013 | 0.711 ± 0.022 | 0.653 ± 0.013 | 0.714 ± 0.024 |
| SegFormer-B2 | 27.3M | 0.658 ± 0.005 | 0.708 ± 0.028 | 0.667 ± 0.003 | 0.731 ± 0.002 |
| *pixel GBT baseline* | — | 0.639 | 0.438 | — | — |

#### `geo_holdout_v1` — pooled flood IoU, mean ± std over seeds 42/1/2 (threshold 0.5)

| Model | Params | test IoU | test IoU @ val-tuned thr. |
|---|---|---|---|
| U-Net (scratch) | 7.8M | 0.676 ± 0.040 | — (not run) |
| U-Net ResNet-34 (ImageNet) | 24.4M | 0.670 ± 0.042 | 0.653 ± 0.049 |
| SegFormer-B0 | 3.7M | 0.652 ± 0.042 | 0.647 ± 0.031 |
| SegFormer-B2 | 27.3M | 0.700 ± 0.048 | 0.682 ± 0.030 |
| *pixel GBT baseline* | — | 0.664 | — |

Per-event test IoU (`official`, mean over seeds):

| Model | Ghana | India | Mekong | Nigeria | Pakistan | Paraguay | Somalia | Spain | Sri-Lanka | USA |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net (scratch) | 0.54 | 0.71 | 0.84 | 0.89 | 0.19 | 0.56 | 0.46 | 0.70 | 0.82 | 0.51 |
| U-Net ResNet-34 (ImageNet) | 0.54 | 0.68 | 0.85 | 0.87 | 0.18 | 0.57 | 0.46 | 0.70 | 0.85 | 0.53 |
| SegFormer-B0 | 0.58 | 0.66 | 0.81 | 0.88 | 0.18 | 0.59 | 0.49 | 0.72 | 0.83 | 0.55 |
| SegFormer-B2 | 0.61 | 0.69 | 0.81 | 0.88 | 0.22 | 0.61 | 0.53 | 0.71 | 0.77 | 0.57 |

Per-event test IoU (`geo_holdout_v1`, mean over seeds):

| Model | Bolivia | Ghana | Mekong |
|---|---|---|---|
| U-Net (scratch) | 0.44 | 0.38 | 0.84 |
| U-Net ResNet-34 (ImageNet) | 0.42 | 0.38 | 0.84 |
| SegFormer-B0 | 0.30 | 0.38 | 0.84 |
| SegFormer-B2 | 0.50 | 0.41 | 0.86 |

Per-event charts: `docs/figures/per_event_iou_*.png`.

### Findings
1. **In-distribution, gains are small.** Official test: SegFormer-B2 0.658 ± 0.005 vs. scratch U-Net 0.645 ± 0.007 (+0.013), with the pixel GBT baseline at 0.639. All four deep models are within ~0.013 IoU.
2. **Pretraining mainly buys robustness on the unseen Bolivia event:** SegFormer-B0/B2 reach 0.71 with low seed variance (± 0.02–0.03) vs. 0.643 ± 0.083 for the scratch U-Net and 0.438 for the pixel GBT. SegFormer-B0 does this with 3.7 M parameters — half the scratch U-Net.
3. **Whole-event hold-out:** SegFormer-B2 is best (0.700 ± 0.048) but seed spreads overlap; not a significant win on 98 chips from 3 events.
4. **Training-event diversity matters more than architecture.** Bolivia is unseen in both schemes, yet every model scores 0.64–0.71 on it when trained on all 10 other events (official) and only 0.30–0.50 when trained on 6 (`geo_holdout_v1`).
5. **Threshold tuning on validation helps only when validation resembles test:** official (in-distribution val) SegFormer-B2 0.658 → 0.667 test, 0.708 → 0.731 Bolivia; geo-holdout (val = different events) it *hurts*, 0.700 → 0.682.
6. **Pakistan is hard for every model** (IoU 0.18–0.22 on the official test) — consistent with EXP-001's observation that errors are dominated by water that is not dark in SAR.
7. **Model selected for serving:** SegFormer-B2, official split, seed chosen by **validation** IoU (seed 42, val 0.670). It happens to have the lowest test IoU of the three seeds (0.653 vs. 0.657 / 0.663) — reported as is; picking by test would be leakage. Registered as `flood-segformer-b2:1` (default); serves a 512² chip in 2.3 s on the Docker CPU worker.

## EXP-003 — Calibration and predictive uncertainty (Phase 8)

- **Date:** 2026-11-18 · **Script:** `scripts/evaluate_uncertainty.py` → `docs/results/uncertainty.json`, `docs/figures/reliability_*.png`
- **Runs:** seed-42 official runs of the scratch U-Net and SegFormer-B2. Temperature fitted on each run's validation split (LBFGS on NLL). 20,000 random valid pixels per chip (so "IoU @ 100%" differs slightly from the pooled all-pixel IoU above). TTA = 8 dihedral transforms; MC dropout = 10 passes (both models contain dropout). Uncertainty = predictive entropy of the mean probability.

| Model (official, seed 42) | Split | ECE raw → T-scaled | Brier raw → T | NLL raw → T | Error-AUROC TTA / MC | IoU @ coverage 100% → 90% → 80% (TTA) |
|---|---|---|---|---|---|---|
| U-Net (scratch) (T = 1.08) | test | 0.027 → 0.031 | 0.0458 → 0.0455 | 0.175 → 0.173 | 0.824 / 0.818 | 0.640 → 0.762 → 0.811 |
| U-Net (scratch) (T = 1.08) | bolivia | 0.026 → 0.029 | 0.0509 → 0.0505 | 0.182 → 0.182 | 0.876 / 0.878 | 0.660 → 0.770 → 0.840 |
| SegFormer-B2 (T = 1.32) | test | 0.034 → 0.032 | 0.0451 → 0.0437 | 0.181 → 0.161 | 0.854 / 0.862 | 0.656 → 0.792 → 0.854 |
| SegFormer-B2 (T = 1.32) | bolivia | 0.068 → 0.077 | 0.0582 → 0.0564 | 0.209 → 0.194 | 0.876 / 0.876 | 0.688 → 0.811 → 0.887 |

### Findings
1. **Low ECE hides a systematic bias.** ECE is small (≈ 0.03 in-distribution) because most pixels are predicted near 0 or 1, but the reliability curves show mid-range probabilities over-predict water (predicted 0.5 → observed ≈ 0.25–0.30). On Bolivia SegFormer-B2 is worse calibrated (ECE 0.068) — overconfidence under distribution shift.
2. **Temperature scaling does not fix it:** NLL/Brier improve slightly, ECE gets worse in 3 of 4 cases. A single temperature cannot remove an asymmetric bias; a calibrator with a bias term (Platt) or isotonic regression is the indicated next step.
3. **Uncertainty is informative:** entropy separates misclassified from correct pixels with AUROC 0.82–0.88. Deferring the 10% most uncertain pixels raises IoU on the remainder from ≈ 0.65 to ≈ 0.76–0.81 — useful for routing pixels to human review.
4. **MC dropout ≈ TTA** (AUROC within 0.01); TTA needs no dropout layers, so it applies to any architecture.

