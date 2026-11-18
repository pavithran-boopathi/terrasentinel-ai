# Demo script (≈ 5 minutes)

Prerequisites: `colima start && docker compose up -d`, a registered default model
(`scripts/register_model.py`), and a Sentinel-1 GeoTIFF with VV/VH in dB — e.g. any
`data/raw/sen1floods11/.../S1Hand/Bolivia_*_S1Hand.tif` (held out from training).

1. **Dashboard** (http://127.0.0.1:5173) — API/DB health, compute device, job counts and durations, default model with its test IoU read from `results.json`.
2. **AI Analysis** — upload the GeoTIFF (validated: bands, dtype, CRS, size), keep the default model, *Start inference*. Watch the progress bar (Celery job). Result: SAR false-colour + predicted water on the map, flooded/valid km², inference time and device.
   - Optional: upload an earlier scene of the same area and select it as *pre-event* → new vs receded water and the log-ratio cross-check.
3. **Disaster Impact** — pick the analysis; upload an OSM buildings and/or roads GeoJSON; *Compute exposure* → exposed buildings (red on the map), flooded road km. Point out the label: *potential exposure, not damage*, and the limitations panel.
4. **Model Laboratory** — registered models with SHA-256 and provenance; Plotly charts of IoU per model and per event, baselines with CIs, latency; SpaceNet 8 results (including the negative deep-model result).
5. **Reports** — download the PDF assessment, CSV summary, mask/probability COGs and GeoJSON.
6. **Behind the scenes** — `http://127.0.0.1:8000/api/v1/docs` (OpenAPI), `mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db`, GitHub Actions (6 jobs incl. real PostGIS and Terraform validate), `infrastructure/terraform`.

Talking points: tiled inference is seamless (matches whole-image inference to 1e-5); the PostGIS area of the flood extent agrees with the raster-derived area to 0.01 %; uncertainty flags errors with AUROC ≈ 0.85; held-out-event evaluation exposes what random splits hide.
