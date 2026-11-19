# Evidence-backed resume bullets

Every number is in EXPERIMENTS.md / docs/results/ and was produced by this repository.

**TerraSentinel AI — satellite flood intelligence platform** (PyTorch, SegFormer, FastAPI, PostGIS, Celery, React, Terraform) · github.com/pavithran-boopathi/terrasentinel-ai

- Built an end-to-end SAR flood-mapping system: Sentinel-1 ingestion and validation, 4 deep segmentation architectures (U-Net, ResNet-34 U-Net, SegFormer-B0/B2) plus classical baselines, trained on an Apple-Silicon GPU in 24 multi-seed runs plus 11-fold cross-validation.
- Designed leakage-aware evaluation (event-held-out splits, 11-fold leave-one-event-out CV, chip-bootstrap CIs); showed pretrained SegFormers lift IoU on an unseen flood event from 0.44 (pixel baseline) to 0.71, while event-to-event variance (0.27–0.83 IoU) exceeds any architecture gain.
- Found and fixed data issues in a public benchmark (an all-NaN test chip, extreme backscatter outliers, event leakage across official splits), documented in a generated data report.
- Quantified predictive uncertainty (MC dropout, test-time augmentation): entropy detects misclassified pixels with AUROC 0.82–0.88; deferring the 10 % least-certain pixels raises IoU from ~0.65 to ~0.78. Showed temperature scaling cannot fix an asymmetric calibration bias.
- Engineered seamless tiled inference for arbitrary scene sizes (Hann blending with zero-weight seams, memory-mapped accumulators): matches whole-image output to 1e-5 and processes ~1,300 km²/s on an M5 Pro with SegFormer-B0 (9.3 ms per 512² chip).
- Delivered a production-style platform: FastAPI with JWT auth and validated GeoTIFF uploads, model registry with provenance, Celery jobs with retries and failure recovery, PostGIS geometries with GiST indexes, Local/S3 storage, PDF/CSV/COG/GeoJSON reports, and a React/MapLibre/Plotly web app.
- Added multitemporal change detection (model-based new/receded water + log-ratio cross-check) and flood-exposure analysis of buildings/roads in local UTM, keeping exposure strictly separate from damage assessment.
- Reported a negative result honestly: an end-to-end infrastructure model failed on 131 SpaceNet 8 tiles; an object-level classifier with known geometry detected flooded roads with F1 0.60 / AUROC 0.87.
- MLOps: 139 automated tests (128 Python, 11 frontend), 6-job GitHub Actions CI (real PostGIS, Docker builds, Terraform validate), MLflow tracking, Docker Compose stack, and an AWS Terraform design (ECS Fargate, RDS, ElastiCache, S3, CloudWatch alarms).
