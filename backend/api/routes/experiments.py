"""Serve the benchmark tables produced by ``scripts/summarize_experiments.py --json``.

The file is generated from run ``results.json`` files and committed under ``docs/results/`` —
the web app never shows hand-typed metrics.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, status

from backend.api.deps import CurrentUser
from backend.core.config import REPO_ROOT

router = APIRouter(prefix="/experiments", tags=["experiments"])
SUMMARY = REPO_ROOT / "docs" / "results" / "experiments.json"


@router.get("")
def experiments(_: CurrentUser) -> dict[str, Any]:
    if not SUMMARY.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No experiment summary generated yet")
    return json.loads(SUMMARY.read_text())
