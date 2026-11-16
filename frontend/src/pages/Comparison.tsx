import { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { apiGet, apiSend } from '../api'
import { Button, DisclaimerBanner, ErrorText, Loading, PageHeader, Panel } from '../components/ui'

type Row = {
  model_name: string
  family: string
  accuracy?: number
  f1_macro?: number
  precision_macro?: number
  recall_macro?: number
}

export function ComparisonPage() {
  const [rows, setRows] = useState<Row[]>([])
  const [active, setActive] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function refresh() {
    const data = await apiGet<{ models: Row[]; active_model?: string }>('/api/comparison')
    setRows(data.models || [])
    setActive(data.active_model || null)
  }

  useEffect(() => {
    void refresh().catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  async function activate(name: string) {
    setError(null)
    try {
      await apiSend('/api/models/active', 'POST', { model_name: name })
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const chartData = rows.map((r) => ({
    name: r.model_name,
    accuracy: Number(r.accuracy || 0),
    f1_macro: Number(r.f1_macro || 0),
    precision: Number(r.precision_macro || 0),
    recall: Number(r.recall_macro || 0),
  }))

  return (
    <div className="space-y-6">
      <PageHeader
        title="Comparison"
        subtitle="All metrics come from held-out evaluation of models trained in this session."
        actions={<Button variant="ghost" onClick={() => void refresh()}>Refresh</Button>}
      />
      <DisclaimerBanner />
      <ErrorText message={error} />

      {!rows.length ? (
        <Panel title="No models yet">
          <p className="m-0 text-[var(--ink-soft)]">Train classical models or the ANN to populate this view.</p>
        </Panel>
      ) : (
        <>
          <Panel title="Metric chart">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d5e2e6" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-20} textAnchor="end" height={80} />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#0b3d4a" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="f1_macro" fill="#2a9d8f" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="precision" fill="#c9852c" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="recall" fill="#7aa2a9" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>

          <Panel title="Leaderboard">
            <div className="overflow-auto">
              <table className="min-w-full text-left text-sm">
                <thead>
                  <tr className="bg-[var(--brand)] text-white">
                    <th className="px-3 py-2">Model</th>
                    <th className="px-3 py-2">Family</th>
                    <th className="px-3 py-2">Accuracy</th>
                    <th className="px-3 py-2">F1 macro</th>
                    <th className="px-3 py-2">Precision</th>
                    <th className="px-3 py-2">Recall</th>
                    <th className="px-3 py-2">Active</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.model_name} className="odd:bg-white even:bg-[#f4f8f9]">
                      <td className="px-3 py-2 font-medium">{r.model_name}</td>
                      <td className="px-3 py-2">{r.family}</td>
                      <td className="px-3 py-2">{r.accuracy?.toFixed?.(4) ?? '—'}</td>
                      <td className="px-3 py-2">{r.f1_macro?.toFixed?.(4) ?? '—'}</td>
                      <td className="px-3 py-2">{r.precision_macro?.toFixed?.(4) ?? '—'}</td>
                      <td className="px-3 py-2">{r.recall_macro?.toFixed?.(4) ?? '—'}</td>
                      <td className="px-3 py-2">
                        {active === r.model_name ? (
                          <span className="rounded-lg bg-[var(--accent-soft)] px-2 py-1 text-xs font-semibold text-[var(--brand)]">Active</span>
                        ) : (
                          <Button variant="ghost" className="!px-2 !py-1 text-xs" onClick={() => void activate(r.model_name)}>
                            Set active
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
      {!rows ? <Loading /> : null}
    </div>
  )
}
