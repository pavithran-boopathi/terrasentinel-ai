# Data (git-ignored)

Nothing in this folder is committed. Planned layout:

```
data/
  raw/sen1floods11/      # Phase 2 — hand-labeled split first
  raw/spacenet8/         # Phase 6
  raw/xbd/               # Phase 8 (requires registration)
  processed/             # chips, normalisation stats, split manifests
  storage/               # API object storage (uploads, masks, reports)
  terrasentinel-dev.db   # local SQLite dev database
```

| Dataset | Use | Source | License / access (verify before download) |
|---|---|---|---|
| Sen1Floods11 | S1 flood segmentation | github.com/cloudtostreet/Sen1Floods11 (`gs://sen1floods11`) | **No license declared** (checked 2026-11-18). Research use with citation; not redistributed. Downloaded: hand-labeled S1 + labels, 0.73 GB |
| SpaceNet 8 | flooded roads/buildings | github.com/SpaceNetChallenge/SpaceNet8 (AWS S3) | to verify in Phase 6 |
| xBD / xView2 | building damage | xview2.org (registration) | to verify in Phase 8; believed non-commercial |

Downloads are never automatic: check `df -h` and get approval first.
