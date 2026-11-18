#!/usr/bin/env python3
"""Download a SpaceNet 8 AOI (public AWS bucket, HTTPS, no AWS account) with integrity checks.

    python scripts/download_spacenet8.py --aoi Germany_Training_Public [--dry-run]

Files are verified against the S3 ETag when it is a plain MD5 (single-part uploads) and against
the listed size otherwise. A MANIFEST.json is written for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "https://spacenet-dataset.s3.amazonaws.com/"
PREFIX = "spacenet/SN8_floods/"


def list_objects(prefix: str) -> list[dict]:
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q), timeout=60) as r:  # noqa: S310
            x = r.read().decode()
        for key, etag, size in re.findall(
            r"<Key>([^<]+)</Key>.*?<ETag>([^<]+)</ETag>.*?<Size>(\d+)</Size>", x, re.S
        ):
            if not key.endswith("/"):
                out.append(
                    {"key": key, "etag": etag.replace("&quot;", "").strip('"'), "size": int(size)}
                )
        m = re.search(r"<NextContinuationToken>([^<]+)<", x)
        if not m:
            return out
        token = m.group(1)


def verify(path: Path, obj: dict) -> bool:
    if not path.exists() or path.stat().st_size != obj["size"]:
        return False
    if "-" in obj["etag"]:  # multipart ETag is not an MD5; size check only
        return True
    h = hashlib.md5()  # noqa: S324 - integrity vs S3 ETag
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == obj["etag"]


def fetch(obj: dict, dest_root: Path) -> str:
    rel = obj["key"].removeprefix(PREFIX)
    dest = (dest_root / rel).resolve()
    if dest_root.resolve() not in dest.parents:
        raise ValueError(f"unsafe key {obj['key']}")
    if verify(dest, obj):
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with (
        urllib.request.urlopen(BUCKET + urllib.parse.quote(obj["key"]), timeout=300) as r,  # noqa: S310
        tmp.open("wb") as f,
    ):  # noqa: S310
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    if not verify(dest, obj):
        dest.unlink()
        raise OSError(f"integrity check failed: {obj['key']}")
    return "downloaded"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="Germany_Training_Public")
    ap.add_argument("--dest", type=Path, default=Path("data/raw/spacenet8"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    objs = list_objects(f"{PREFIX}{a.aoi}/") + [
        o for o in list_objects(f"{PREFIX}resolutions.txt") if o["key"].endswith("resolutions.txt")
    ]
    total = sum(o["size"] for o in objs)
    print(f"{len(objs)} files, {total / 1e9:.2f} GB")
    if a.dry_run:
        return 0
    if shutil.disk_usage(Path(".")).free < 2 * total:
        print("not enough free disk", file=sys.stderr)
        return 1
    a.dest.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"downloaded": 0, "skipped": 0}
    errors = []
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(fetch, o, a.dest): o for o in objs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                counts[f.result()] += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{futs[f]['key']}: {exc}")
            if i % 100 == 0 or i == len(objs):
                print(f"  {i}/{len(objs)}", flush=True)
    manifest = sorted(
        (
            {"path": o["key"].removeprefix(PREFIX), "size": o["size"], "etag": o["etag"]}
            for o in objs
        ),
        key=lambda m: m["path"],
    )
    (a.dest / f"MANIFEST_{a.aoi}.json").write_text(json.dumps({"files": manifest}, indent=1))
    print(f"downloaded={counts['downloaded']} skipped={counts['skipped']} errors={len(errors)}")
    for e in errors[:10]:
        print("  ERROR", e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
