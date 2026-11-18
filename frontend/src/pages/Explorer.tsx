import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { MapView } from "../components/MapView";
import { Card, PageHeader, SignInPrompt, fmt, select } from "../components/ui";
import { apiUrl, resources } from "../lib/api";
import { useAuth } from "../lib/auth";

export default function Explorer() {
  const { user } = useAuth();
  const images = useQuery({ queryKey: ["images"], queryFn: resources.images, enabled: !!user });
  const analyses = useQuery({ queryKey: ["analyses"], queryFn: resources.analyses, enabled: !!user });
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [mix, setMix] = useState(0.5);
  const [showFlood, setShowFlood] = useState(true);

  const imgA = images.data?.find((i) => i.id === a);
  const imgB = images.data?.find((i) => i.id === b);
  const floodFor = (id?: string) =>
    analyses.data?.find((j) => j.image_id === id && j.status === "succeeded");

  const rasters = useMemo(() => {
    const out = [];
    if (imgA) out.push({ id: `a-${imgA.id}`, url: apiUrl(`/images/${imgA.id}/preview.png`), bbox: imgA.bbox, opacity: imgB ? 1 - mix : 1 });
    if (imgB) out.push({ id: `b-${imgB.id}`, url: apiUrl(`/images/${imgB.id}/preview.png`), bbox: imgB.bbox, opacity: mix });
    const job = floodFor(imgB?.id ?? imgA?.id);
    const ref = imgB ?? imgA;
    if (showFlood && job && ref) out.push({ id: `f-${job.id}`, url: apiUrl(`/analyses/${job.id}/preview.png`), bbox: ref.bbox, opacity: 0.85 });
    return out;
  }, [imgA, imgB, mix, showFlood, analyses.data]);

  const footprints = useMemo(
    () =>
      images.data?.length
        ? [{
            id: "footprints",
            kind: "line" as const,
            color: "#f59e0b",
            data: {
              type: "FeatureCollection" as const,
              features: images.data.map((i) => ({ type: "Feature" as const, properties: { name: i.filename }, geometry: i.footprint })),
            },
          }]
        : [],
    [images.data],
  );
  const fit = (imgB ?? imgA)?.bbox ?? null;

  return (
    <>
      <PageHeader title="Satellite Explorer" subtitle="Your uploaded scenes, footprints, flood overlays and before/after comparison" />
      <div className="grid gap-5 p-8 lg:grid-cols-4">
        <div className="space-y-5">
          {!user ? (
            <SignInPrompt what="explore your imagery" />
          ) : (
            <Card title="Compare scenes">
              <label className="block text-sm">
                <span className="text-slate-400">A (e.g. before)</span>
                <select className={`${select} mt-1 w-full`} value={a} onChange={(e) => setA(e.target.value)} aria-label="Scene A">
                  <option value="">—</option>
                  {images.data?.map((i) => <option key={i.id} value={i.id}>{i.filename}</option>)}
                </select>
              </label>
              <label className="mt-3 block text-sm">
                <span className="text-slate-400">B (e.g. after)</span>
                <select className={`${select} mt-1 w-full`} value={b} onChange={(e) => setB(e.target.value)} aria-label="Scene B">
                  <option value="">—</option>
                  {images.data?.map((i) => <option key={i.id} value={i.id}>{i.filename}</option>)}
                </select>
              </label>
              {imgA && imgB && (
                <label className="mt-4 block text-sm">
                  <span className="text-slate-400">A ← blend → B</span>
                  <input type="range" min={0} max={1} step={0.01} value={mix} onChange={(e) => setMix(+e.target.value)} className="mt-2 w-full" aria-label="Blend" />
                </label>
              )}
              <label className="mt-4 flex items-center gap-2 text-sm">
                <input type="checkbox" checked={showFlood} onChange={(e) => setShowFlood(e.target.checked)} />
                Show predicted flood water
              </label>
            </Card>
          )}
          {(imgB ?? imgA) && (
            <Card title="Scene">
              <dl className="grid grid-cols-2 gap-y-1 text-xs">
                <dt className="text-slate-400">Size</dt><dd>{(imgB ?? imgA)!.width}×{(imgB ?? imgA)!.height}</dd>
                <dt className="text-slate-400">CRS</dt><dd>{(imgB ?? imgA)!.crs}</dd>
                <dt className="text-slate-400">File size</dt><dd>{fmt((imgB ?? imgA)!.size_bytes / 1e6, 1)} MB</dd>
              </dl>
            </Card>
          )}
        </div>
        <div className="lg:col-span-3">
          <MapView rasters={rasters} vectors={footprints} fitTo={fit} height={600} />
          <p className="mt-2 text-xs text-slate-500">
            SAR shown as false colour (R = VV, G = VH, B = VV−VH); open water appears dark. Orange outlines are scene footprints.
          </p>
        </div>
      </div>
    </>
  );
}
