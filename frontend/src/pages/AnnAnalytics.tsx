import { useEffect, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { apiGet } from '../api'
import { Button, DisclaimerBanner, ErrorText, Loading, PageHeader, Panel, Stat } from '../components/ui'

export function AnnAnalyticsPage() {
  const [data, setData] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function refresh() {
    setData(await apiGet('/api/ann/analytics'))
  }

  useEffect(() => {
    void refresh().catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  const history = (data?.history || {}) as Record<string, number[]>
  const chartData = (history.loss || history.accuracy || []).map((_, idx) => ({
    epoch: idx + 1,
    loss: history.loss?.[idx],
    val_loss: history.val_loss?.[idx],
    accuracy: history.accuracy?.[idx],
    val_accuracy: history.val_accuracy?.[idx],
  }))

  const metrics = (data?.metrics || {}) as Record<string, number>

  return (
    <div className="space-y-6">
      <PageHeader
        title="ANN Analytics"
        subtitle="Training curves and held-out metrics for the neural network."
        actions={<Button variant="ghost" onClick={() => void refresh()}>Refresh</Button>}
      />
      <DisclaimerBanner />
      <ErrorText message={error} />

      {!data ? (
        <Loading />
      ) : !data.trained ? (
        <Panel title="ANN not trained">
          <p className="m-0 text-[var(--ink-soft)]">
            Train the ANN from the Training page. Backend:{' '}
            {String((data.tensorflow as { backend?: string } | undefined)?.backend || 'unknown')}
          </p>
        </Panel>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-4">
            <Stat label="Backend" value={String(data.backend)} />
            <Stat label="Accuracy" value={metrics.accuracy != null ? metrics.accuracy.toFixed(3) : '—'} />
            <Stat label="F1 macro" value={metrics.f1_macro != null ? metrics.f1_macro.toFixed(3) : '—'} />
            <Stat label="Epochs logged" value={chartData.length} />
          </div>
          <Panel title="Learning curves">
            {chartData.length ? (
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#d5e2e6" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {history.loss ? <Line type="monotone" dataKey="loss" stroke="#0b3d4a" dot={false} /> : null}
                    {history.val_loss ? <Line type="monotone" dataKey="val_loss" stroke="#c9852c" dot={false} /> : null}
                    {history.accuracy ? <Line type="monotone" dataKey="accuracy" stroke="#2a9d8f" dot={false} /> : null}
                    {history.val_accuracy ? <Line type="monotone" dataKey="val_accuracy" stroke="#7aa2a9" dot={false} /> : null}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-[var(--ink-soft)]">No curve history available for this backend.</p>
            )}
          </Panel>
          <Panel title="Metrics JSON">
            <pre className="m-0 max-h-96 overflow-auto rounded-xl bg-[#0b3d4a] p-4 text-xs text-[#d8f0ec]">
              {JSON.stringify(data.metrics, null, 2)}
            </pre>
          </Panel>
        </>
      )}
    </div>
  )
}
