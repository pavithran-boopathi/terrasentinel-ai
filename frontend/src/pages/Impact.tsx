import { NotYetBuilt, PageHeader } from "../components/ui";

export default function Impact() {
  return (
    <>
      <PageHeader title="Disaster Impact" subtitle="Flood extent and potentially exposed infrastructure" />
      <div className="p-8">
        <NotYetBuilt phase={6} what="Flood exposure and infrastructure impact estimation" />
      </div>
    </>
  );
}
