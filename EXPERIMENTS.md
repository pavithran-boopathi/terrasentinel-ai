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
_(empty — first entries arrive in Phase 3)_
