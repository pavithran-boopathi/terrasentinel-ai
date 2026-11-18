"""STAC search for Sentinel-1 scenes (Element 84 Earth Search by default, no API key)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
S1_COLLECTION = "sentinel-1-grd"


@dataclass(frozen=True)
class SceneSummary:
    id: str
    datetime: str
    bbox: tuple[float, ...]
    platform: str | None
    orbit_state: str | None
    instrument_mode: str | None
    polarizations: tuple[str, ...]
    assets: tuple[str, ...]


def validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    minx, miny, maxx, maxy = bbox
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise ValueError(f"invalid lon/lat bbox: {bbox}")


def summarize(item: Any) -> SceneSummary:
    p = item.properties
    return SceneSummary(
        id=item.id,
        datetime=str(p.get("datetime")),
        bbox=tuple(item.bbox or ()),
        platform=p.get("platform"),
        orbit_state=p.get("sat:orbit_state"),
        instrument_mode=p.get("sar:instrument_mode"),
        polarizations=tuple(p.get("sar:polarizations", ())),
        assets=tuple(sorted(item.assets)),
    )


def search_sentinel1(
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    *,
    catalog_url: str = EARTH_SEARCH,
    collection: str = S1_COLLECTION,
    max_items: int = 20,
    client: Any = None,
) -> list[SceneSummary]:
    """Find Sentinel-1 GRD scenes intersecting ``bbox`` between ``start`` and ``end`` (inclusive).
    ``client`` may be injected for tests; otherwise a ``pystac_client.Client`` is opened."""
    validate_bbox(bbox)
    if end < start:
        raise ValueError("end date before start date")
    if client is None:
        from pystac_client import Client

        client = Client.open(catalog_url)
    search = client.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        max_items=max_items,
    )
    return [summarize(it) for it in search.items()]
