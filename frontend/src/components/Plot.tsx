import type { Data, Layout } from "plotly.js";
import { useEffect, useRef } from "react";

const BASE: Partial<Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#cbd5e1", size: 11 },
  margin: { l: 50, r: 10, t: 30, b: 60 },
  xaxis: { gridcolor: "#1e293b" },
  yaxis: { gridcolor: "#1e293b", zerolinecolor: "#334155" },
  legend: { orientation: "h", y: -0.25 },
};

/** Thin wrapper around the lazily-loaded Plotly basic bundle. */
export function Plot({ data, layout, height = 320 }: { data: Data[]; layout?: Partial<Layout>; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let alive = true;
    const el = ref.current;
    import("plotly.js-basic-dist-min").then((m) => {
      const Plotly = (m as unknown as { default: typeof import("plotly.js") }).default;
      if (alive && el) Plotly.react(el, data, { ...BASE, ...layout, height }, { displayModeBar: false, responsive: true });
    });
    return () => {
      alive = false;
    };
  }, [data, layout, height]);
  return <div ref={ref} />;
}
