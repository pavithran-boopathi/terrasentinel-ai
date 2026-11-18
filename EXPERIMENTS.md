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
- **Code:** U-Net runs at commit `d1c8708` (first run at `d2c099e`; ML code identical — `git diff d2c099e d1c8708 -- ml/` is empty). Runs flagged "dirty" had only `scripts/summarize_experiments.py` modified, not `ml/`.
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

