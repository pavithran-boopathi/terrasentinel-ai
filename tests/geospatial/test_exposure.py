import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from geospatial.exposure import ExposureParams, exposure_report, utm_crs_for

UTM = "EPSG:32632"  # Germany


def gdf(geoms, crs=UTM):
    return gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=geoms, crs=crs)


def test_utm_zone_selection():
    assert utm_crs_for(box(6.9, 50.4, 7.0, 50.5)) == "EPSG:32632"
    assert utm_crs_for(box(-65, -15, -64, -14)) == "EPSG:32720"


def test_building_and_road_exposure_in_metres():
    flood = gdf([box(0, 0, 100, 100)])
    buildings = gdf(
        [
            box(10, 10, 20, 20),  # fully inside
            box(95, 10, 105, 20),  # 50% inside
            box(98, 50, 108, 60),  # 20% inside  → below 25% threshold
            box(200, 200, 210, 210),  # outside
        ]
    )
    roads = gdf([LineString([(-50, 50), (150, 50)]), LineString([(300, 0), (300, 100)])])
    rep = exposure_report(flood, buildings, roads, crs=UTM)
    assert rep["flood_area_km2"] == pytest.approx(0.01)
    assert rep["buildings"] == {
        "total": 4,
        "potentially_exposed": 2,
        "exposed_footprint_m2": pytest.approx(200.0),
    }
    f = rep["building_features"]["flooded_fraction"].round(3).tolist()
    assert f == [1.0, 0.5, 0.2, 0.0]
    assert rep["roads"]["total_km"] == pytest.approx(0.3)
    assert rep["roads"]["flooded_km"] == pytest.approx(0.1)
    assert rep["roads"]["segments_exposed"] == 1
    assert "not damage" in rep["label"]


def test_reprojects_wgs84_inputs_and_threshold_param():
    x0, y0 = 360000, 5600000  # near the Ahr valley, Germany (UTM 32N)
    flood = gdf([box(x0, y0, x0 + 100, y0 + 100)]).to_crs("EPSG:4326")
    bld = gdf([box(x0 + 95, y0 + 10, x0 + 105, y0 + 20)]).to_crs("EPSG:4326")
    strict = exposure_report(flood, bld, params=ExposureParams(min_building_overlap=0.6))
    assert strict["crs"] == UTM and strict["buildings"]["potentially_exposed"] == 0
    loose = exposure_report(flood, bld, params=ExposureParams(min_building_overlap=0.4))
    assert loose["buildings"]["potentially_exposed"] == 1


def test_no_flood_means_no_exposure_and_missing_crs_rejected():
    rep = exposure_report(gdf([]), gdf([box(0, 0, 1, 1)]), crs=UTM)
    assert rep["buildings"]["potentially_exposed"] == 0 and rep["flood_area_km2"] == 0
    with pytest.raises(ValueError, match="CRS"):
        exposure_report(
            gdf([box(0, 0, 1, 1)]), gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]), crs=UTM
        )
