import { useQuery } from "@tanstack/react-query";
import type { Data } from "plotly.js";
import { useMemo, useState } from "react";
import { Plot } from "../components/Plot";
import { Card, PageHeader, SignInPrompt, fmt, select } from "../components/ui";
import { resources, type ModelGroup } from "../lib/api";
import { useAuth } from "../lib/auth";

const ARCH_LABEL: Record<string, string> = {
  unet: "U-Net (scratch)",
  unet_resnet34: "U-Net ResNet-34 (ImageNet)",
  segformer_b0: "SegFormer-B0",
  segformer_b2: "SegFormer-B2",
};
const label = (a: string) => ARCH_LABEL[a] ?? a;

export default function ModelLab() {
  const { user } = useAuth();
  const models = useQuery({ queryKey: ["models"], queryFn: resources.models, enabled: !!user });
  const exp = useQuery({ queryKey: ["experiments"], queryFn: resources.experiments, enabled: !!user, retry: false });
  const [scheme, setScheme] = useState("official");

  const groups = useMemo(() => exp.data?.sen1floods11.models.filter((m) => m.split_scheme === scheme) ?? [], [exp.data, scheme]);
  const splits = useMemo(() => Array.from(new Set(groups.flatMap((g) => Object.keys(g.splits)))).filter((s) => s !== "val"), [groups]);
  const baselines = useMemo(() => exp.data?.sen1floods11.baselines.filter((b) => b.split_scheme === scheme) ?? [], [exp.data, scheme]);

  const iouChart: Data[] = useMemo(
    () =>
      splits.map((s) => ({
        type: "bar",
        name: s,
        x: groups.map((g) => label(g.arch)),
        y: groups.map((g) => g.splits[s]?.iou.mean ?? null),
        error_y: { type: "data", array: groups.map((g) => g.splits[s]?.iou.std ?? 0), visible: true },
      })) as Data[],
    [groups, splits],
  );
  const eventChart: Data[] = useMemo(() => {
    const test = groups.filter((g) => g.splits.test);
    const events = Array.from(new Set(test.flatMap((g) => Object.keys(g.splits.test.per_event_iou))));
    return test.map((g: ModelGroup) => ({
      type: "bar",
      name: label(g.arch),
      x: events,
      y: events.map((e) => g.splits.test.per_event_iou[e]?.mean ?? null),
    })) as Data[];
  }, [groups]);

  if (!user)
    return (
      <>
        <PageHeader title="Model Laboratory" subtitle="Trained models, experiments and comparisons" />
        <div className="p-8"><SignInPrompt what="view models and experiments" /></div>
      </>
    );

  return (
    <>
      <PageHeader title="Model Laboratory" subtitle="Every number below is read from run results files (docs/results/experiments.json)" />
      <div className="space-y-5 p-8">
        <Card title="Registered models (serving)">
          {models.data?.length ? (
            <table className="w-full text-left text-sm">
              <thead className="text-xs text-slate-400">
                <tr><th className="py-1">Model</th><th>Arch</th><th>Threshold</th><th>Test IoU</th><th>Trained on</th><th>SHA-256</th></tr>
              </thead>
              <tbody>
                {models.data.map((m) => {
                  const test = (m.metrics.test ?? {}) as { iou?: number };
                  return (
                    <tr key={m.id} className="border-t border-slate-800">
                      <td className="py-1.5">{m.name}:{m.version}{m.is_default && <span className="ml-2 rounded bg-sky-500/15 px-1.5 text-xs text-sky-300">default</span>}</td>
                      <td>{label(m.arch)}</td>
                      <td>{m.threshold}</td>
                      <td>{fmt(test.iou)}</td>
                      <td>{String(m.training.split_scheme ?? "—")}</td>
                      <td className="font-mono text-xs text-slate-500">{m.sha256.slice(0, 12)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <p className="text-sm text-slate-400">No registered models.</p>
          )}
        </Card>

        {exp.isError && <p className="text-sm text-slate-400">No experiment summary yet — run scripts/export_experiments_json.py.</p>}
        {exp.data && (
          <>
            <div className="flex items-center gap-3 text-sm">
              <span className="text-slate-400">Split scheme</span>
              <select className={select} value={scheme} onChange={(e) => setScheme(e.target.value)} aria-label="Split scheme">
                <option value="official">official (Sen1Floods11)</option>
                <option value="geo_holdout_v1">geo_holdout_v1 (unseen events)</option>
              </select>
              <span className="text-xs text-slate-500">generated {exp.data.generated}</span>
            </div>
            <div className="grid gap-5 lg:grid-cols-2">
              <Card title="Flood IoU by model (mean ± std over seeds)">
                <Plot data={iouChart} layout={{ yaxis: { title: { text: "IoU" }, range: [0, 1] }, barmode: "group" }} />
              </Card>
              <Card title="Per-event test IoU (mean over seeds)">
                <Plot data={eventChart} layout={{ yaxis: { title: { text: "IoU" }, range: [0, 1] }, barmode: "group" }} />
              </Card>
            </div>
            <Card title="Deep models">
              <table className="w-full text-left text-sm">
                <thead className="text-xs text-slate-400">
                  <tr><th className="py-1">Model</th><th>Params</th><th>Seeds</th>{splits.map((s) => <th key={s}>{s} IoU</th>)}{splits.map((s) => <th key={s + "f"}>{s} F1</th>)}</tr>
                </thead>
                <tbody>
                  {groups.map((g) => (
                    <tr key={g.arch} className="border-t border-slate-800">
                      <td className="py-1.5">{label(g.arch)}</td>
                      <td>{(g.params / 1e6).toFixed(1)}M</td>
                      <td>{g.seeds.length}</td>
                      {splits.map((s) => <td key={s}>{g.splits[s] ? `${fmt(g.splits[s].iou.mean)} ± ${fmt(g.splits[s].iou.std)}` : "—"}</td>)}
                      {splits.map((s) => <td key={s + "f"}>{fmt(g.splits[s]?.f1.mean)}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
            <Card title="Classical baselines">
              <table className="w-full text-left text-sm">
                <thead className="text-xs text-slate-400"><tr><th className="py-1">Method</th><th>Split</th><th>IoU [95% CI]</th><th>F1</th><th>Precision</th><th>Recall</th></tr></thead>
                <tbody>
                  {baselines.map((b) => (
                    <tr key={b.method + b.split} className="border-t border-slate-800">
                      <td className="py-1.5">{b.method}</td><td>{b.split}</td>
                      <td>{fmt(b.iou)} {b.iou_ci95 && <span className="text-slate-500">[{fmt(b.iou_ci95[0])}, {fmt(b.iou_ci95[1])}]</span>}</td>
                      <td>{fmt(b.f1)}</td><td>{fmt(b.precision)}</td><td>{fmt(b.recall)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
            {exp.data.latency && (
              <Card title={`Inference latency (${exp.data.latency.device}, 512×512 chip)`}>
                <Plot
                  height={260}
                  data={[{ type: "bar", x: exp.data.latency.rows.filter((r) => r.batch === 1).map((r) => label(r.arch)), y: exp.data.latency.rows.filter((r) => r.batch === 1).map((r) => r.ms_per_chip), name: "ms / chip (batch 1)" }] as Data[]}
                  layout={{ yaxis: { title: { text: "ms per chip" } } }}
                />
              </Card>
            )}
            {exp.data.spacenet8.length > 0 && (
              <Card title="Infrastructure flood attribution (SpaceNet 8 Germany, spatial test block)">
                <table className="w-full text-left text-sm">
                  <thead className="text-xs text-slate-400"><tr><th className="py-1">Run</th><th>mIoU (fg)</th><th>Flooded-building F1</th><th>Flooded-road F1</th><th>Building detection</th></tr></thead>
                  <tbody>
                    {exp.data.spacenet8.map((r) => (
                      <tr key={r.run} className="border-t border-slate-800">
                        <td className="py-1.5">{r.run}</td><td>{fmt(r.test.mean_iou_fg)}</td>
                        <td>{fmt(r.test.object.flooded_building.f1)}</td><td>{fmt(r.test.object.flooded_road_length.f1)}</td>
                        <td>{fmt(r.test.object.building_detection_rate)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            )}
          </>
        )}
      </div>
    </>
  );
}
