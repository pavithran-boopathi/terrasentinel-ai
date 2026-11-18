import { NotYetBuilt, PageHeader } from "../components/ui";

export default function Reports() {
  return (
    <>
      <PageHeader title="Reports" subtitle="PDF, GeoJSON, GeoTIFF and CSV exports" />
      <div className="p-8">
        <NotYetBuilt phase={7} what="Report generation" />
      </div>
    </>
  );
}
