# Model card — `flood-segformer-b2:1`

| | |
|---|---|
| Task | Binary flood (open water) segmentation from Sentinel-1 GRD backscatter |
| Input | 2 bands, VV and VH in dB, 10 m, any CRS; clipped to [−50, 5] dB and standardised with training statistics stored in the checkpoint |
| Output | P(water) per pixel; mask at threshold 0.5; polygons with geodesic areas |
| Architecture | SegFormer-B2 (MiT-B2 encoder, ImageNet-pretrained; RGB stem replaced by a 2-channel conv initialised from the mean RGB filter), all-MLP decoder, 27.3 M parameters |
| Training data | Sen1Floods11 v1.1 hand-labelled chips, official train split (252 chips, 10 flood events); Paraguay_34417 excluded (all-NaN S1) |
| Training | AdamW lr 2e-4, cosine with 3-epoch warm-up, batch 8, random 256² crops, dihedral flips/rotations, light noise, BCE (pos_weight 3.08) + soft Dice, early stopping on val IoU; Apple M5 Pro (MPS) |
| Selection | Of 3 seeds, the one with the highest **validation** IoU (seed 42) — not chosen on test |
| Run | `segformer-b2-official-20261118-152455` (commit `e616cb1`), sha256 in the registry |

## Evaluation (this checkpoint, pooled flood IoU, 95 % chip-bootstrap CI in `results.json`)
| Split | IoU [95 % CI] | F1 | Precision | Recall |
|---|---|---|---|---|
| Official test (89 chips) | 0.653 [0.543, 0.739] | 0.790 | 0.759 | 0.823 |
| Bolivia (15 chips, unseen event) | 0.677 [0.491, 0.768] | 0.807 | 0.704 | 0.947 |

Across 3 seeds the configuration scores 0.658 ± 0.005 (test) and 0.708 ± 0.028 (Bolivia); see EXPERIMENTS.md EXP-002.

## Calibration & uncertainty (EXP-003)
ECE 0.034 (test) / 0.068 (Bolivia); mid-range probabilities over-predict water. Temperature scaling (T = 1.32) improves NLL but not ECE. TTA entropy detects errors with AUROC 0.85–0.88.

## Intended use and limits
- Mapping **open** flood water in SAR scenes for situational awareness; outputs are estimates.
- Misses flooded vegetation, urban flooding and narrow channels more often than it raises false alarms (held-out events: precision ≈ 0.9, recall ≈ 0.7).
- Not validated outside the 11 Sen1Floods11 events, for other sensors, polarisations or radiometric processing chains.
- Does **not** assess damage. "Potentially exposed" infrastructure = overlap with predicted water.
