"""Downloadable reports for an analysis: CSV summary and a PDF assessment."""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LIMITATIONS = [
    "Flood extent is predicted from Sentinel-1 SAR backscatter by a statistical model; it is an "
    "estimate, not a survey.",
    "SAR detects open water well; flooded vegetation, flooded urban areas and narrow channels are "
    "frequently missed (see EXPERIMENTS.md, EXP-001 findings).",
    "'Potentially exposed' means geometric overlap with predicted water. It is NOT damage "
    "assessment; structural damage requires a separate model and imagery.",
    "Reported model metrics come from held-out Sen1Floods11 chips and may not transfer to other "
    "regions, sensors or seasons.",
]


def _rows(job: Any) -> list[tuple[str, Any]]:
    s = job.summary or {}
    mv = job.model_version
    rows: list[tuple[str, Any]] = [
        ("analysis_id", str(job.id)),
        ("image", job.image.filename),
        ("model", f"{mv.name}:{mv.version} ({mv.arch})"),
        ("threshold", s.get("threshold")),
        ("status", job.status.value if hasattr(job.status, "value") else job.status),
        ("flooded_area_km2", s.get("flooded_area_km2")),
        ("valid_area_km2", s.get("valid_area_km2")),
        ("flooded_fraction_of_valid", s.get("flooded_fraction_of_valid")),
        ("flood_polygons", s.get("polygons")),
        ("inference_seconds", s.get("seconds")),
        ("inference_device", s.get("device")),
    ]
    test = (mv.metrics or {}).get("test", {})
    if test:
        rows.append(("model_test_iou", test.get("iou")))
        rows.append(("model_test_f1", test.get("f1")))
    exp = s.get("exposure")
    if exp:
        if "buildings" in exp:
            rows += [
                ("buildings_total", exp["buildings"]["total"]),
                ("buildings_potentially_exposed", exp["buildings"]["potentially_exposed"]),
            ]
        if "roads" in exp:
            rows += [
                ("roads_total_km", exp["roads"]["total_km"]),
                ("roads_flooded_km", exp["roads"]["flooded_km"]),
            ]
    return rows


def summary_csv(job: Any) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["field", "value"])
    w.writerows(_rows(job))
    return buf.getvalue()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.4f}"
    return "—" if v is None else str(v)


def pdf_report(job: Any, sar_png: bytes | None, mask_png: bytes | None) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.07, 0.95, "TerraSentinel AI — Flood assessment", fontsize=16, weight="bold")
        fig.text(
            0.07,
            0.925,
            f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
            fontsize=9,
            color="#555",
        )
        if sar_png:
            ax = fig.add_axes((0.07, 0.52, 0.86, 0.38))
            ax.imshow(Image.open(io.BytesIO(sar_png)))
            if mask_png:
                ax.imshow(Image.open(io.BytesIO(mask_png)))
            ax.set_title("Sentinel-1 (VV, VH, ratio) with predicted flood water (blue)", fontsize=9)
            ax.axis("off")
        rows = [(k, _fmt(v)) for k, v in _rows(job)]
        tab = fig.add_axes((0.07, 0.22, 0.86, 0.27))
        tab.axis("off")
        t = tab.table(
            cellText=rows,
            colLabels=["Field", "Value"],
            loc="upper left",
            colWidths=[0.4, 0.6],
            cellLoc="left",
        )
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        y = 0.19
        fig.text(0.07, y, "Limitations", fontsize=11, weight="bold")
        for line in LIMITATIONS:
            y -= 0.035
            fig.text(0.07, y, "• " + line, fontsize=7.5, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


def write_csv(path: Path, job: Any) -> None:
    path.write_text(summary_csv(job))
