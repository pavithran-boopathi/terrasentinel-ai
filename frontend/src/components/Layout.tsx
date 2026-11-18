import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../lib/auth";

export const NAV = [
  { to: "/", label: "Dashboard" },
  { to: "/explorer", label: "Satellite Explorer" },
  { to: "/analysis", label: "AI Analysis" },
  { to: "/impact", label: "Disaster Impact" },
  { to: "/models", label: "Model Laboratory" },
  { to: "/reports", label: "Reports" },
];

export function Layout() {
  const { user, logout } = useAuth();
  return (
    <div className="flex h-full">
      <aside className="flex w-60 shrink-0 flex-col border-r border-slate-800 bg-slate-900/60">
        <div className="px-5 py-5">
          <div className="text-lg font-semibold tracking-tight">TerraSentinel AI</div>
          <div className="text-xs text-slate-400">Disaster intelligence</div>
        </div>
        <nav className="flex flex-1 flex-col gap-1 px-3" aria-label="Main">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              className={({ isActive }) =>
                `rounded-md px-3 py-2 text-sm ${
                  isActive ? "bg-sky-500/15 text-sky-300" : "text-slate-300 hover:bg-slate-800"
                }`
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-slate-800 px-5 py-4 text-xs text-slate-400">
          {user ? (
            <div className="flex items-center justify-between gap-2">
              <span className="truncate" title={user.email}>
                {user.email}
              </span>
              <button onClick={logout} className="text-sky-400 hover:underline">
                Sign out
              </button>
            </div>
          ) : (
            <NavLink to="/login" className="text-sky-400 hover:underline">
              Sign in
            </NavLink>
          )}
        </div>
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
