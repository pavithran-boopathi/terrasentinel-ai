import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Card, PageHeader, Stat, StatusBadge, StatusDot, fmt } from "../components/ui";
import { api, resources } from "../lib/api";
import { useAuth } from "../lib/auth";

export default function Dashboard() {
  const { user } = useAuth();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const ready = useQuery({ queryKey: ["ready"], queryFn: api.ready });
  const compute = useQuery({ queryKey: ["compute"], queryFn: api.compute });
  const mon = useQuery({ queryKey: ["monitoring"], queryFn: resources.monitoring, enabled: !!user, refetchInterval: 5000 });
  const analyses = useQuery({ queryKey: ["analyses"], queryFn: resources.analyses, enabled: !!user, refetchInterval: 5000 });
  const models = useQuery({ queryKey: ["models"], queryFn: resources.models, enabled: !!user });
  const def = models.data?.find((m) => m.is_default);
  const defTest = (def?.metrics.test ?? {}) as { iou?: number; f1?: number };
  const totalJobs = Object.values(mon.data?.jobs_by_status ?? {}).reduce((a, b) => a + b, 0);

  return (
    <>
      <PageHeader title="Dashboard" subtitle="Live system status, processing jobs and model performance" />
      <div className="space-y-5 p-8">
        <div className="grid gap-5 md:grid-cols-3">
          <Card title="API">
            {health.isPending ? (
              <p className="text-sm text-slate-400">Checking…</p>
            ) : health.isError ? (
              <p className="text-sm text-rose-300"><StatusDot ok={false} /> Unreachable — is the backend running on :8000?</p>
            ) : (
              <p className="text-sm"><StatusDot ok /> <span className="ml-1">Online · v{health.data.version}</span></p>
            )}
          </Card>
          <Card title="Dependencies">
            {ready.data ? (
              <ul className="space-y-1 text-sm">
                {Object.entries(ready.data.checks).map(([k, v]) => (
                  <li key={k} className="flex items-center gap-2"><StatusDot ok={v === "ok"} /> <span className="capitalize">{k}</span><span className="text-slate-400">{v}</span></li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-400">{ready.isError ? "Unavailable" : "Checking…"}</p>
            )}
          </Card>
          <Card title="Compute">
            {compute.data ? (
              <dl className="grid grid-cols-2 gap-y-1 text-sm">
                <dt className="text-slate-400">Device</dt><dd className="font-mono uppercase">{compute.data.recommended_device}</dd>
                <dt className="text-slate-400">PyTorch</dt><dd className="font-mono">{compute.data.torch_version ?? "not installed"}</dd>
                <dt className="text-slate-400">Machine</dt><dd className="font-mono">{compute.data.machine}</dd>
              </dl>
            ) : (
              <p className="text-sm text-slate-400">{compute.isError ? "Unavailable" : "Checking…"}</p>
            )}
          </Card>
        </div>
        {!user ? (
          <p className="text-sm text-slate-400"><Link to="/login" className="text-sky-400 hover:underline">Sign in</Link> to see jobs, models and analyses.</p>
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label="Analyses" value={fmt(totalJobs, 0)} hint={`${mon.data?.succeeded_last_24h ?? 0} succeeded / ${mon.data?.failed_last_24h ?? 0} failed in 24 h`} />
              <Stat label="Job duration" value={mon.data?.duration_seconds.mean != null ? `${fmt(mon.data.duration_seconds.mean, 1)} s` : "—"} hint={mon.data?.duration_seconds.p95 != null ? `p95 ${fmt(mon.data.duration_seconds.p95, 1)} s` : "mean, succeeded jobs"} />
              <Stat label="Default model" value={def ? `${def.name}:${def.version}` : "none"} hint={def?.arch} />
              <Stat label="Default model test IoU" value={fmt(defTest.iou)} hint={def ? `${String(def.training.split_scheme)} split · from results.json` : undefined} />
            </div>
            <Card title="Recent analyses">
              {analyses.data?.length ? (
                <table className="w-full text-left text-sm">
                  <thead className="text-xs text-slate-400"><tr><th className="py-1">Scene</th><th>Model</th><th>Status</th><th>Flood km²</th><th>Exposed buildings</th><th>When</th></tr></thead>
                  <tbody>
                    {analyses.data.slice(0, 8).map((j) => (
                      <tr key={j.id} className="border-t border-slate-800">
                        <td className="py-1.5"><Link className="text-sky-400 hover:underline" to={`/impact?analysis=${j.id}`}>{j.image_filename}</Link></td>
                        <td>{j.model_name}</td>
                        <td><StatusBadge status={j.status} /></td>
                        <td>{fmt(j.summary?.flooded_area_km2)}</td>
                        <td>{j.summary?.exposure?.buildings ? `${j.summary.exposure.buildings.potentially_exposed} / ${j.summary.exposure.buildings.total}` : "—"}</td>
                        <td className="text-xs text-slate-400">{new Date(j.created_at).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-sm text-slate-400">No analyses yet. <Link to="/analysis" className="text-sky-400 hover:underline">Run one</Link>.</p>
              )}
            </Card>
          </>
        )}
      </div>
    </>
  );
}
