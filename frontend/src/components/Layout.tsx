import { NavLink, Outlet } from 'react-router-dom'
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  FileSpreadsheet,
  FlaskConical,
  History,
  LayoutDashboard,
  Settings,
  Sparkles,
  Stethoscope,
  Table2,
  Upload,
} from 'lucide-react'

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/assessment', label: 'Patient Assessment', icon: Stethoscope },
  { to: '/dataset', label: 'Dataset', icon: Database },
  { to: '/preprocessing', label: 'Preprocessing', icon: FlaskConical },
  { to: '/training', label: 'Training', icon: Sparkles },
  { to: '/comparison', label: 'Comparison', icon: BarChart3 },
  { to: '/ann', label: 'ANN Analytics', icon: BrainCircuit },
  { to: '/batch', label: 'Batch', icon: Upload },
  { to: '/reports', label: 'Reports', icon: FileSpreadsheet },
  { to: '/history', label: 'History', icon: History },
  { to: '/settings', label: 'Settings', icon: Settings },
]

export function Layout() {
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[280px_1fr]">
      <aside className="border-b border-[var(--line)] bg-[linear-gradient(180deg,#0b3d4a_0%,#0f4d5c_48%,#146b78_100%)] text-white lg:min-h-screen lg:border-b-0 lg:border-r">
        <div className="px-5 py-6">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/10 ring-1 ring-white/20">
              <Activity className="h-5 w-5 text-[#7ed0c5]" />
            </div>
            <div>
              <div className="display text-2xl leading-none">Insulin Matrix</div>
              <div className="mt-1 text-xs tracking-[0.14em] text-[#b7e0da] uppercase">
                AI Screening Lab
              </div>
            </div>
          </div>
          <p className="mt-4 text-sm text-[#d5ece8]">
            Synthetic diabetes classification for demonstration and model development.
          </p>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-4 lg:flex-col lg:overflow-visible">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex min-w-max items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition ${
                  isActive
                    ? 'bg-white text-[var(--brand)] shadow-lg shadow-black/10'
                    : 'text-[#d7ebe7] hover:bg-white/10'
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="hidden px-5 pb-6 text-xs text-[#9fc9c3] lg:block">
          <div className="flex items-center gap-2">
            <Table2 className="h-3.5 w-3.5" />
            FastAPI · scikit-learn · Keras
          </div>
        </div>
      </aside>
      <main className="px-4 py-6 md:px-8 md:py-8">
        <Outlet />
      </main>
    </div>
  )
}
