"""Flood *exposure* of buildings and roads: geometric overlap with predicted flood extent.

Exposure is not damage. A building is "potentially exposed" when a minimum fraction of its
footprint lies inside predicted flood water; roads report the length inside the flood extent.
All measurements are made in a local UTM projection (metres), never in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

WGS84 = "EPSG:4326"


def utm_crs_for(geom: BaseGeometry, crs: str = WGS84) -> str:
    """Local UTM zone for the centroid of ``geom`` (given in ``crs``)."""
    c = gpd.GeoSeries([geom], crs=crs).to_crs(WGS84).iloc[0].centroid
    zone = int((c.x + 180) // 6) + 1
    return f"EPSG:{32600 + zone if c.y >= 0 else 32700 + zone}"


@dataclass(frozen=True)
class ExposureParams:
    min_building_overlap: float = 0.25  # fraction of footprint area inside flood → exposed
    road_buffer_m: float = 0.0  # optional tolerance around flood extent for roads
    flood_buffer_m: float = 0.0  # dilate flood extent (e.g. to account for mask resolution)


def _prepare(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("layer has no CRS")
    out = gdf.to_crs(crs)
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    out["geometry"] = out.geometry.make_valid()
    return out


def building_exposure(
    buildings: gpd.GeoDataFrame, flood: BaseGeometry, crs: str, params: ExposureParams
) -> gpd.GeoDataFrame:
    b = buildings.copy()
    area = b.geometry.area
    inter = b.geometry.intersection(flood).area if not flood.is_empty else 0.0 * area
    b["footprint_m2"] = area
    b["flooded_m2"] = inter
    b["flooded_fraction"] = np.where(area > 0, inter / area, 0.0)
    b["potentially_exposed"] = b["flooded_fraction"] >= params.min_building_overlap
    return b


def road_exposure(
    roads: gpd.GeoDataFrame, flood: BaseGeometry, crs: str, params: ExposureParams
) -> gpd.GeoDataFrame:
    r = roads.copy()
    zone = (
        flood.buffer(params.road_buffer_m) if params.road_buffer_m and not flood.is_empty else flood
    )
    length = r.geometry.length
    inside = r.geometry.intersection(zone).length if not zone.is_empty else 0.0 * length
    r["length_m"] = length
    r["flooded_length_m"] = inside
    r["potentially_exposed"] = r["flooded_length_m"] > 0
    return r


def exposure_report(
    flood_polygons: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None = None,
    roads: gpd.GeoDataFrame | None = None,
    params: ExposureParams | None = None,
    crs: str | None = None,
) -> dict[str, Any]:
    """Return summary statistics plus per-feature tables (GeoDataFrames in ``crs``)."""
    params = params or ExposureParams()
    layers = [g for g in (flood_polygons, buildings, roads) if g is not None and len(g)]
    if crs is None:
        if not layers:
            raise ValueError("no input features")
        ref = layers[0].to_crs(WGS84).geometry.union_all()
        crs = utm_crs_for(ref)
    flood_gdf = _prepare(flood_polygons, crs) if len(flood_polygons) else flood_polygons
    flood = unary_union(list(flood_gdf.geometry)) if len(flood_gdf) else unary_union([])
    if params.flood_buffer_m and not flood.is_empty:
        flood = flood.buffer(params.flood_buffer_m)
    out: dict[str, Any] = {
        "crs": crs,
        "params": params.__dict__,
        "flood_area_km2": float(flood.area / 1e6) if not flood.is_empty else 0.0,
        "label": "potential exposure (overlap with predicted flood water) — not damage",
    }
    if buildings is not None:
        b = building_exposure(_prepare(buildings, crs), flood, crs, params)
        out["buildings"] = {
            "total": int(len(b)),
            "potentially_exposed": int(b["potentially_exposed"].sum()),
            "exposed_footprint_m2": float(b.loc[b["potentially_exposed"], "footprint_m2"].sum()),
        }
        out["building_features"] = b
    if roads is not None:
        r = road_exposure(_prepare(roads, crs), flood, crs, params)
        out["roads"] = {
            "total_km": float(r["length_m"].sum() / 1000),
            "flooded_km": float(r["flooded_length_m"].sum() / 1000),
            "segments_exposed": int(r["potentially_exposed"].sum()),
        }
        out["road_features"] = r
    return out


def summary_only(report: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if not k.endswith("_features")}


def features_to_geojson(gdf: gpd.GeoDataFrame, columns: list[str]) -> dict[str, Any]:
    keep = gdf[[*columns, "geometry"]].to_crs(WGS84)
    for c in columns:
        if pd.api.types.is_float_dtype(keep[c]):
            keep[c] = keep[c].round(3)
    return keep.__geo_interface__
