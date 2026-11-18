import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { useAuth } from "./lib/auth";
import Analysis from "./pages/Analysis";
import Dashboard from "./pages/Dashboard";
import Explorer from "./pages/Explorer";
import Impact from "./pages/Impact";
import Login from "./pages/Login";
import ModelLab from "./pages/ModelLab";
import Reports from "./pages/Reports";

function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return <p className="p-8 text-sm text-slate-400">Loading…</p>;
  return user ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="explorer" element={<Explorer />} />
        <Route
          path="analysis"
          element={
            <RequireAuth>
              <Analysis />
            </RequireAuth>
          }
        />
        <Route path="impact" element={<Impact />} />
        <Route path="models" element={<ModelLab />} />
        <Route path="reports" element={<Reports />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
