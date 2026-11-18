import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { MapView } from "../components/MapView";
import { Card, PageHeader, Stat, StatusBadge, btn, fmt, select } from "../components/ui";
import { apiUrl, resources, type Analysis as Job } from "../lib/api";

export default function Analysis() {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: resources.models });
  const images = useQuery({ queryKey: ["images"], queryFn: resources.images });
  const [file, setFile] = useState<File | null>(null);
  const [imageId, setImageId] = useState("");
  const [modelId, setModelId] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);

  const upload = useMutation({
    mutationFn: (f: File) => resources.uploadImage(f),
    onSuccess: (img) => {
      setImageId(img.id);
      setFile(null);
      qc.invalidateQueries({ queryKey: ["images"] });
    },
  });
  const start = useMutation({
    mutationFn: () => resources.createAnalysis(imageId, modelId || undefined),
    onSuccess: (job) => setJobId(job.id),
  });
  const job = useQuery({
    queryKey: ["analysis", jobId],
    queryFn: () => resources.analysis(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => {
      const s = (q.state.data as Job | undefined)?.status;
      return s === "succeeded" || s === "failed" ? false : 1000;
    },
  });
  const flood = useQuery({
    queryKey: ["flood-geojson", jobId],
    queryFn: () => resources.floodGeojson(jobId!),
    enabled: job.data?.status === "succeeded",
  });

  const image = images.data?.find((i) => i.id === (job.data?.image_id ?? imageId));
  const rasters = useMemo(() => {
    if (!image) return [];
    const r = [{ id: `img-${image.id}`, url: apiUrl(`/images/${image.id}/preview.png`), bbox: image.bbox }];
    if (job.data?.status === "succeeded")
      r.push({ id: `mask-${job.data.id}`, url: apiUrl(`/analyses/${job.data.id}/preview.png`), bbox: image.bbox });
    return r;
  }, [image, job.data?.status, job.data?.id]);
  const vectors = useMemo(
    () =>
      flood.data ? [{ id: `poly-${jobId}`, data: flood.data, kind: "line" as const, color: "#38bdf8" }] : [],
    [flood.data, jobId],
  );
  const s = job.data?.summary;

  return (
    <>
      <PageHeader title="AI Analysis" subtitle="Upload a Sentinel-1 scene, choose a model, run flood segmentation" />
      <div className="grid gap-5 p-8 lg:grid-cols-3">
        <div className="space-y-5">
          <Card title="1 · Sentinel-1 GeoTIFF">
            <p className="mb-3 text-xs text-slate-400">2 bands (VV, VH) in dB, any CRS. Validated on upload.</p>
            <input
              type="file"
              accept=".tif,.tiff"
              aria-label="GeoTIFF file"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm file:mr-3 file:rounded file:border-0 file:bg-slate-800 file:px-3 file:py-1.5 file:text-slate-200"
            />
            <button className={`${btn} mt-3`} disabled={!file || upload.isPending} onClick={() => file && upload.mutate(file)}>
              {upload.isPending ? "Uploading…" : "Upload"}
            </button>
            {upload.isError && <p role="alert" className="mt-2 text-sm text-rose-300">{upload.error.message}</p>}
            {!!images.data?.length && (
              <label className="mt-4 block text-sm">
                <span className="text-slate-400">or use an uploaded image</span>
                <select className={`${select} mt-1 w-full`} value={imageId} onChange={(e) => setImageId(e.target.value)}>
                  <option value="">Select…</option>
                  {images.data.map((i) => (
                    <option key={i.id} value={i.id}>
                      {i.filename} · {i.width}×{i.height}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </Card>
          <Card title="2 · Model">
            {models.data?.length ? (
              <select className={`${select} w-full`} value={modelId} onChange={(e) => setModelId(e.target.value)} aria-label="Model">
                <option value="">Default model</option>
                {models.data.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}:{m.version} ({m.arch}){m.is_default ? " · default" : ""}
                  </option>
                ))}
              </select>
            ) : (
              <p className="text-sm text-slate-400">No models registered. Run scripts/register_model.py.</p>
            )}
          </Card>
          <Card title="3 · Run">
            <button className={btn} disabled={!imageId || !models.data?.length || start.isPending} onClick={() => start.mutate()}>
              Start inference
            </button>
            {start.isError && <p role="alert" className="mt-2 text-sm text-rose-300">{start.error.message}</p>}
            {job.data && (
              <div className="mt-4 space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <StatusBadge status={job.data.status} />
                  <span className="text-slate-400">attempt {job.data.attempts}</span>
                </div>
                <div className="h-2 overflow-hidden rounded bg-slate-800" role="progressbar" aria-valuenow={Math.round(job.data.progress * 100)}>
                  <div className="h-full bg-sky-500 transition-all" style={{ width: `${job.data.progress * 100}%` }} />
                </div>
                {job.data.error && <p className="text-rose-300">{job.data.error}</p>}
              </div>
            )}
          </Card>
        </div>
        <div className="space-y-5 lg:col-span-2">
          <MapView rasters={rasters} vectors={vectors} fitTo={image?.bbox ?? null} height={460} />
          {s && (
            <>
              <div className="grid gap-4 sm:grid-cols-4">
                <Stat label="Flood water" value={`${fmt(s.flooded_area_km2)} km²`} hint="predicted open water" />
                <Stat label="Valid area" value={`${fmt(s.valid_area_km2)} km²`} />
                <Stat label="Flooded share" value={`${fmt((s.flooded_fraction_of_valid ?? 0) * 100, 1)}%`} />
                <Stat label="Inference" value={`${fmt(s.seconds, 2)} s`} hint={`${s.device} · ${s.model}`} />
              </div>
              <p className="text-xs text-slate-500">
                Predicted from SAR backscatter with threshold {s.threshold}. Next:{" "}
                <Link to={`/impact?analysis=${job.data!.id}`} className="text-sky-400 hover:underline">
                  estimate infrastructure exposure
                </Link>{" "}
                or{" "}
                <Link to="/reports" className="text-sky-400 hover:underline">
                  download reports
                </Link>
                .
              </p>
            </>
          )}
        </div>
      </div>
    </>
  );
}
