import "maplibre-gl/dist/maplibre-gl.css";
import type { GeoJSONSource, Map as MLMap, StyleSpecification } from "maplibre-gl";
import { useEffect, useRef, useState } from "react";
import { API_BASE, tokenStore } from "../lib/api";

export interface RasterOverlay {
  id: string;
  url: string; // absolute URL of a WGS84 PNG
  bbox: [number, number, number, number]; // west, south, east, north
  opacity?: number;
}

export interface VectorOverlay {
  id: string;
  data: GeoJSON.FeatureCollection | GeoJSON.Geometry;
  kind: "fill" | "line";
  color: string;
  /** Optional property-driven colour: [property, valueIfTrue, valueIfFalse] for booleans. */
  byBool?: [string, string, string];
  opacity?: number;
}

const BASEMAP: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.6 } }],
};

const corners = ([w, s, e, n]: RasterOverlay["bbox"]): [[number, number], [number, number], [number, number], [number, number]] => [
  [w, n],
  [e, n],
  [e, s],
  [w, s],
];

export function MapView({
  rasters = [],
  vectors = [],
  fitTo,
  height = 480,
}: {
  rasters?: RasterOverlay[];
  vectors?: VectorOverlay[];
  fitTo?: [number, number, number, number] | null;
  height?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!ref.current) return;
    let cancelled = false;
    let map: MLMap | undefined;
    import("maplibre-gl").then(({ Map, NavigationControl, ScaleControl }) => {
      if (cancelled || !ref.current) return;
      map = new Map({
        container: ref.current,
        style: BASEMAP,
        center: [10, 20],
        zoom: 1.5,
        // Protected API resources (previews) need the bearer token.
        transformRequest: (url) => {
          const t = tokenStore.get();
          return url.includes(API_BASE) && t ? { url, headers: { Authorization: `Bearer ${t}` } } : { url };
        },
      });
      map.addControl(new NavigationControl(), "top-right");
      map.addControl(new ScaleControl({ unit: "metric" }), "bottom-left");
      map.on("load", () => {
        mapRef.current = map!;
        setReady(true);
      });
    });
    return () => {
      cancelled = true;
      mapRef.current = null;
      map?.remove();
    };
  }, []);

  // Sync raster overlays.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const wanted = new Set(rasters.map((r) => `r-${r.id}`));
    for (const layer of map.getStyle().layers ?? []) {
      if (layer.id.startsWith("r-") && !wanted.has(layer.id)) {
        map.removeLayer(layer.id);
        map.removeSource(layer.id);
      }
    }
    for (const r of rasters) {
      const id = `r-${r.id}`;
      if (!map.getSource(id)) {
        map.addSource(id, { type: "image", url: r.url, coordinates: corners(r.bbox) });
        map.addLayer({ id, type: "raster", source: id, paint: { "raster-fade-duration": 0 } });
      }
      map.setPaintProperty(id, "raster-opacity", r.opacity ?? 1);
    }
  }, [ready, rasters]);

  // Sync vector overlays (always drawn above rasters).
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const wanted = new Set(vectors.map((v) => `v-${v.id}`));
    for (const layer of map.getStyle().layers ?? []) {
      if (layer.id.startsWith("v-") && !wanted.has(layer.id)) {
        map.removeLayer(layer.id);
        map.removeSource(layer.id);
      }
    }
    for (const v of vectors) {
      const id = `v-${v.id}`;
      const color = v.byBool
        ? (["case", ["==", ["get", v.byBool[0]], true], v.byBool[1], v.byBool[2]] as unknown as string)
        : v.color;
      if (map.getSource(id)) {
        (map.getSource(id) as GeoJSONSource).setData(v.data);
      } else {
        map.addSource(id, { type: "geojson", data: v.data });
        if (v.kind === "fill") {
          map.addLayer({
            id,
            type: "fill",
            source: id,
            paint: { "fill-color": color, "fill-opacity": v.opacity ?? 0.5, "fill-outline-color": "#ffffff" },
          });
        } else {
          map.addLayer({ id, type: "line", source: id, paint: { "line-color": color, "line-width": 2 } });
        }
      }
    }
  }, [ready, vectors]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !fitTo) return;
    map.fitBounds(
      [
        [fitTo[0], fitTo[1]],
        [fitTo[2], fitTo[3]],
      ],
      { padding: 40, duration: 0, maxZoom: 16 },
    );
  }, [ready, fitTo]);

  return (
    <div
      ref={ref}
      style={{ height }}
      className="overflow-hidden rounded-lg border border-slate-800"
      data-testid="map"
    />
  );
}
