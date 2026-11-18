import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { MapView } from "../components/MapView";
import { Card, PageHeader, SignInPrompt, Stat, btn, fmt, select } from "../components/ui";
import { apiUrl, resources } from "../lib/api";
import { useAuth } from "../lib/auth";

export default function Impact() {
  const { user } = useAuth();
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const id = params.get("analysis") ?? "";
  const analyses = useQuery({ queryKey: ["analyses"], queryFn: resources.analyses, enabled: !!user });
  const images = useQuery({ queryKey: ["images"], queryFn: resources.images, enabled: !!user });
  const job = analyses.data?.find((j) => j.id === id);
  const image = images.data?.find((i) => i.id === job?.image_id);
  const [buildings, setBuildings] = useState<File | null>(null);
  const [roads, setRoads] = useState<File | null>(null);
  const [overlap, setOverlap] = useState(0.25);

  const run = useMutation({
    mutationFn: () => resources.exposure(id, buildings, roads, overlap),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["analyses"] });
      qc.invalidateQueries({ queryKey: ["exposed", id] });
    },
  });
  const exposure = run.data ?? job?.summary?.exposure;
  const exposed = useQuery({
    queryKey: ["exposed", id],
    queryFn: () => resources.exposedBuildings(id),
    enabled: !!job?.files.includes("exposed_buildings.geojson") || run.isSuccess,
  });
  const rasters = useMemo(
    () =>
      job && image
        ? [
            { id: `img-${image.id}`, url: apiUrl(`/images/${image.id}/preview.png`), bbox: image.bbox },
            { id: `mask-${job.id}`, url: apiUrl(`/analyses/${job.id}/preview.png`), bbox: image.bbox, opacity: 0.85 },
          ]
        : [],
    [job, image],
  );
  const vectors = useMemo(
    () =>
      exposed.data
        ? [{ id: `bld-${id}`, kind: "fill" as const, color: "#94a3b8", byBool: ["potentially_exposed", "#f43f5e", "#94a3b8"] as [string, string, string], data: exposed.data, opacity: 0.8 }]
        : [],
    [exposed.data, id],
  );
  const s = job?.summary;

  if (!user)
    return (
      <>
        <PageHeader title="Disaster Impact" subtitle="Flood extent and potentially exposed infrastructure" />
        <div className="p-8"><SignInPrompt what="assess impact" /></div>
      </>
    );

  return (
    <>
      <PageHeader title="Disaster Impact" subtitle="Flood extent and potentially exposed infrastructure — exposure, not damage" />
      <div className="grid gap-5 p-8 lg:grid-cols-3">
        <div className="space-y-5">
          <Card title="Analysis">
            <select className={`${select} w-full`} value={id} onChange={(e) => setParams({ analysis: e.target.value })} aria-label="Analysis">
              <option value="">Select a finished analysis…</option>
              {analyses.data?.filter((j) => j.status === "succeeded").map((j) => (
                <option key={j.id} value={j.id}>
                  {j.image_filename} · {new Date(j.created_at).toLocaleString()}
                </option>
              ))}
            </select>
          </Card>
          {job && (
            <Card title="Infrastructure layers (GeoJSON)">
              <p className="mb-3 text-xs text-slate-400">
                Building footprints (polygons) and/or road centrelines (lines), e.g. exported from OpenStreetMap.
              </p>
              <label className="block text-sm">
                Buildings
                <input type="file" accept=".geojson,.json" onChange={(e) => setBuildings(e.target.files?.[0] ?? null)} className="mt-1 block w-full text-xs" aria-label="Buildings GeoJSON" />
              </label>
              <label className="mt-3 block text-sm">
                Roads
                <input type="file" accept=".geojson,.json" onChange={(e) => setRoads(e.target.files?.[0] ?? null)} className="mt-1 block w-full text-xs" aria-label="Roads GeoJSON" />
              </label>
              <label className="mt-3 block text-sm">
                <span className="text-slate-400">Building counts as exposed when ≥ {Math.round(overlap * 100)}% of footprint is in predicted water</span>
                <input type="range" min={0.05} max={1} step={0.05} value={overlap} onChange={(e) => setOverlap(+e.target.value)} className="mt-1 w-full" aria-label="Overlap threshold" />
              </label>
              <button className={`${btn} mt-3`} disabled={(!buildings && !roads) || run.isPending} onClick={() => run.mutate()}>
                {run.isPending ? "Computing…" : "Compute exposure"}
              </button>
              {run.isError && <p role="alert" className="mt-2 text-sm text-rose-300">{run.error.message}</p>}
            </Card>
          )}
          <Card title="Uncertainty & limitations">
            <ul className="list-disc space-y-1 pl-4 text-xs text-slate-400">
              <li>Flood water is a model prediction from SAR backscatter, not a survey.</li>
              <li>On held-out flood events the models miss water more often than they invent it (recall 0.68–0.79 vs precision 0.89–0.93 in EXP-001), so exposure is likely <em>under</em>-estimated.</li>
              <li>Flooded vegetation, urban flooding and narrow channels are frequently not visible as dark SAR returns.</li>
              <li>“Potentially exposed” = geometric overlap with predicted water. It says nothing about structural damage.</li>
            </ul>
          </Card>
        </div>
        <div className="space-y-5 lg:col-span-2">
          {s && (
            <div className="grid gap-4 sm:grid-cols-4">
              <Stat label="Flood water" value={`${fmt(s.flooded_area_km2)} km²`} />
              <Stat label="Flood polygons" value={fmt(s.polygons, 0)} />
              <Stat label="Exposed buildings" value={exposure?.buildings ? `${exposure.buildings.potentially_exposed} / ${exposure.buildings.total}` : "—"} hint="potential exposure" />
              <Stat label="Flooded roads" value={exposure?.roads ? `${fmt(exposure.roads.flooded_km, 2)} km` : "—"} hint={exposure?.roads ? `of ${fmt(exposure.roads.total_km, 2)} km` : undefined} />
            </div>
          )}
          <MapView rasters={rasters} vectors={vectors} fitTo={image?.bbox ?? null} height={520} />
          {exposure && <p className="text-xs text-slate-500">{exposure.label} · measured in {exposure.crs}</p>}
        </div>
      </div>
    </>
  );
}
