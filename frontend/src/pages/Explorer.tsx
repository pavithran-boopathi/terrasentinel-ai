import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef } from "react";
import { NotYetBuilt, PageHeader } from "../components/ui";

// Public MapLibre demo basemap (no API key). Satellite imagery overlays arrive in Phase 7.
const STYLE_URL = "https://demotiles.maplibre.org/style.json";

export default function Explorer() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    let map: import("maplibre-gl").Map | undefined;
    let cancelled = false;
    import("maplibre-gl").then(({ Map, NavigationControl }) => {
      if (cancelled || !ref.current) return;
      map = new Map({ container: ref.current, style: STYLE_URL, center: [10, 20], zoom: 1.5 });
      map.addControl(new NavigationControl(), "top-right");
    });
    return () => {
      cancelled = true;
      map?.remove();
    };
  }, []);

  return (
    <>
      <PageHeader title="Satellite Explorer" subtitle="Interactive map foundation" />
      <div className="space-y-5 p-8">
        <div ref={ref} className="h-[480px] overflow-hidden rounded-lg border border-slate-800" />
        <NotYetBuilt phase={7} what="Satellite imagery overlays and before/after comparison" />
      </div>
    </>
  );
}
