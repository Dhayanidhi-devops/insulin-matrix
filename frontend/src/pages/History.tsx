import { useEffect, useState } from 'react'
import { apiGet, apiSend } from '../api'
import { Button, DisclaimerBanner, ErrorText, PageHeader, Panel } from '../components/ui'

type Item = {
  prediction_id: string
  created_at: string
  model_name: string
  predicted_class: string
  source: string
  disclaimer: string
  probabilities?: Record<string, number> | null
}

export function HistoryPage() {
  const [items, setItems] = useState<Item[]>([])
  const [experiments, setExperiments] = useState<Record<string, unknown>[]>([])
  const [error, setError] = useState<string | null>(null)

  async function refresh() {
    const hist = await apiGet<{ items: Item[] }>('/api/history')
    setItems(hist.items)
    const exp = await apiGet<{ items: Record<string, unknown>[] }>('/api/experiments')
    setExperiments(exp.items)
  }

  useEffect(() => {
    void refresh().catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  async function clearAll() {
    await apiSend('/api/history', 'DELETE')
    await refresh()
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="History"
        subtitle="Prediction log and training experiment records stored in SQLite."
        actions={
          <>
            <Button variant="ghost" onClick={() => void refresh()}>Refresh</Button>
            <Button variant="danger" onClick={() => void clearAll()}>Clear predictions</Button>
          </>
        }
      />
      <DisclaimerBanner />
      <ErrorText message={error} />

      <Panel title="Predictions">
        <div className="overflow-auto">
          <table className="min-w-full text-left text-sm">
            <thead>
              <tr className="bg-[var(--brand)] text-white">
                <th className="px-3 py-2">When</th>
                <th className="px-3 py-2">ID</th>
                <th className="px-3 py-2">Model</th>
                <th className="px-3 py-2">Class</th>
                <th className="px-3 py-2">Source</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.prediction_id} className="odd:bg-white even:bg-[#f4f8f9]">
                  <td className="px-3 py-2">{item.created_at}</td>
                  <td className="px-3 py-2">{item.prediction_id}</td>
                  <td className="px-3 py-2">{item.model_name}</td>
                  <td className="px-3 py-2 font-medium">{item.predicted_class}</td>
                  <td className="px-3 py-2">{item.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!items.length ? <p className="text-[var(--ink-soft)]">No predictions yet.</p> : null}
        </div>
      </Panel>

      <Panel title="Training experiments">
        <div className="overflow-auto">
          <table className="min-w-full text-left text-sm">
            <thead>
              <tr className="bg-[var(--brand-2)] text-white">
                <th className="px-3 py-2">When</th>
                <th className="px-3 py-2">Model</th>
                <th className="px-3 py-2">Accuracy</th>
                <th className="px-3 py-2">F1 macro</th>
                <th className="px-3 py-2">Train/Test</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp) => {
                const metrics = (exp.metrics || {}) as Record<string, number>
                return (
                  <tr key={String(exp.run_id)} className="odd:bg-white even:bg-[#f4f8f9]">
                    <td className="px-3 py-2">{String(exp.created_at)}</td>
                    <td className="px-3 py-2">{String(exp.model_name)}</td>
                    <td className="px-3 py-2">{metrics.accuracy?.toFixed?.(4) ?? '—'}</td>
                    <td className="px-3 py-2">{metrics.f1_macro?.toFixed?.(4) ?? '—'}</td>
                    <td className="px-3 py-2">{String(exp.n_train)} / {String(exp.n_test)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  )
}
