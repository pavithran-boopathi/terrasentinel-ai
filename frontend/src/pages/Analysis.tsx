import { NotYetBuilt, PageHeader } from "../components/ui";

export default function Analysis() {
  return (
    <>
      <PageHeader title="AI Analysis" subtitle="Upload imagery, pick a model and run inference" />
      <div className="p-8">
        <NotYetBuilt phase={5} what="Image upload and model inference" />
      </div>
    </>
  );
}
