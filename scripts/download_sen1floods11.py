#!/usr/bin/env python3
"""Download the Sen1Floods11 v1.1 hand-labeled subset from the public GCS bucket over HTTPS.

No gsutil / Google account needed. Each file is verified against the MD5 that GCS reports, existing
verified files are skipped, and a manifest (name, size, md5) is written for reproducibility.

    python scripts/download_sen1floods11.py                  # S1 + labels (~0.75 GB)
    python scripts/download_sen1floods11.py --with-s2        # also Sentinel-2 (~1 GB more)
    python scripts/download_sen1floods11.py --dry-run        # list and size only
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "sen1floods11"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"
ROOT = "v1.1/"
HAND = "v1.1/data/flood_events/HandLabeled/"

BASE_PREFIXES = [
    HAND + "S1Hand/",
    HAND + "LabelHand/",
    HAND + "JRCWaterHand/",
    HAND + "S1OtsuLabelHand/",
    "v1.1/splits/flood_handlabeled/",
]
EXTRA_FILES = ["v1.1/Sen1Floods11_Metadata.geojson"]
S2_PREFIX = HAND + "S2Hand/"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https host
        return json.load(r)


def list_objects(prefix: str) -> list[dict]:
    out, token = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            q["pageToken"] = token
        d = _get_json(f"{API}?{urllib.parse.urlencode(q)}")
        out += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return out


def md5_b64(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check against GCS, not security
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def fetch(obj: dict, dest_root: Path) -> tuple[str, str]:
    rel = obj["name"].removeprefix(ROOT)
    dest = (dest_root / rel).resolve()
    if dest_root.resolve() not in dest.parents:
        raise ValueError(f"unsafe object path: {obj['name']}")
    if dest.exists() and md5_b64(dest) == obj["md5Hash"]:
        return rel, "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = MEDIA + urllib.parse.quote(obj["name"])
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f)
    if md5_b64(tmp) != obj["md5Hash"]:
        tmp.unlink()
        raise OSError(f"MD5 mismatch for {obj['name']}")
    tmp.replace(dest)
    return rel, "downloaded"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path("data/raw/sen1floods11"))
    ap.add_argument("--with-s2", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    prefixes = BASE_PREFIXES + ([S2_PREFIX] if args.with_s2 else [])
    objs = [o for p in prefixes for o in list_objects(p)]
    objs += [o for f in EXTRA_FILES for o in list_objects(f) if o["name"] == f]
    total = sum(int(o["size"]) for o in objs)
    print(f"{len(objs)} files, {total / 1e9:.2f} GB")
    if args.dry_run:
        return 0

    free = shutil.disk_usage(args.dest.parent if args.dest.parent.exists() else Path(".")).free
    if free < total * 2:
        print(f"Refusing: only {free / 1e9:.1f} GB free", file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0}
    errors: list[str] = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, o, args.dest): o for o in objs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                _, status = fut.result()
                counts[status] += 1
            except Exception as exc:  # noqa: BLE001 - collect and report all failures
                errors.append(f"{futures[fut]['name']}: {exc}")
            if i % 200 == 0 or i == len(objs):
                print(f"  {i}/{len(objs)}", flush=True)

    manifest = sorted(
        (
            {"path": o["name"].removeprefix(ROOT), "size": int(o["size"]), "md5": o["md5Hash"]}
            for o in objs
        ),
        key=lambda m: m["path"],
    )
    (args.dest / "MANIFEST.json").write_text(
        json.dumps({"bucket": BUCKET, "files": manifest}, indent=1)
    )
    print(f"downloaded={counts['downloaded']} skipped={counts['skipped']} errors={len(errors)}")
    for e in errors[:20]:
        print("  ERROR", e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
