import { NotYetBuilt, PageHeader } from "../components/ui";

export default function ModelLab() {
  return (
    <>
      <PageHeader title="Model Laboratory" subtitle="Trained models, experiments and comparisons" />
      <div className="p-8">
        <NotYetBuilt phase={3} what="Model registry and experiment history" />
      </div>
    </>
  );
}
