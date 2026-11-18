import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Card, PageHeader, SignInPrompt, StatusBadge, btnGhost, fmt } from "../components/ui";
import { downloadFile, resources } from "../lib/api";
import { useAuth } from "../lib/auth";

const FILES: [string, string][] = [
  ["report.pdf", "PDF report"],
  ["report.csv", "CSV summary"],
  ["files/flood_mask.tif", "Mask GeoTIFF"],
  ["files/probability.tif", "Probability GeoTIFF"],
  ["files/flood_polygons.geojson", "Flood GeoJSON"],
  ["files/exposed_buildings.geojson", "Exposed buildings"],
  ["files/exposed_roads.geojson", "Exposed roads"],
  ["files/new_water.tif", "New-water GeoTIFF"],
];

export default function Reports() {
  const { user } = useAuth();
  const analyses = useQuery({ queryKey: ["analyses"], queryFn: resources.analyses, enabled: !!user });
  const [err, setErr] = useState<string | null>(null);

  const available = (files: string[], path: string) =>
    path.startsWith("report") ? true : files.includes(path.replace("files/", ""));

  return (
    <>
      <PageHeader title="Reports" subtitle="PDF assessments, CSV summaries, GeoTIFF and GeoJSON outputs" />
      <div className="p-8">
        {!user ? (
          <SignInPrompt what="download reports" />
        ) : (
          <Card title="Analyses">
            {err && <p role="alert" className="mb-2 text-sm text-rose-300">{err}</p>}
            {analyses.data?.length ? (
              <table className="w-full text-left text-sm">
                <thead className="text-xs text-slate-400">
                  <tr><th className="py-1">Scene</th><th>Model</th><th>Status</th><th>Flood km²</th><th>Downloads</th></tr>
                </thead>
                <tbody>
                  {analyses.data.map((j) => (
                    <tr key={j.id} className="border-t border-slate-800 align-top">
                      <td className="py-2">{j.image_filename}<div className="text-xs text-slate-500">{new Date(j.created_at).toLocaleString()}</div></td>
                      <td>{j.model_name}</td>
                      <td><StatusBadge status={j.status} /></td>
                      <td>{fmt(j.summary?.flooded_area_km2)}</td>
                      <td className="py-2">
                        {j.status === "succeeded" ? (
                          <div className="flex flex-wrap gap-1.5">
                            {FILES.filter(([p]) => available(j.files, p)).map(([p, name]) => (
                              <button
                                key={p}
                                className={`${btnGhost} text-xs`}
                                onClick={() => downloadFile(`/analyses/${j.id}/${p}`, `${j.id.slice(0, 8)}_${p.split("/").pop()}`).catch((e) => setErr(String(e)))}
                              >
                                {name}
                              </button>
                            ))}
                          </div>
                        ) : (
                          <span className="text-xs text-slate-500">{j.error ?? "not ready"}</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-sm text-slate-400">No analyses yet. Run one from AI Analysis.</p>
            )}
          </Card>
        )}
      </div>
    </>
  );
}
