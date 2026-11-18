# ml/

Training and evaluation code. Install with `pip install -e ".[ml,geo]"`.

- `data/` — dataset classes, split builders, normalisation (Phase 2)
- `models/` — U-Net, SegFormer adapters, damage model (Phases 3–4, 8)
- `training/` — training loop, losses, early stopping, MLflow logging (Phase 3)
- `evaluation/` — metrics, latency benchmarks, calibration, error analysis (Phases 3, 8)
- `configs/` — YAML experiment configs

Empty until Phase 2/3 — no placeholder implementations are kept here.
