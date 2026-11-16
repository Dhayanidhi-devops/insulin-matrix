import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { apiGet, apiSend, type Overview as OverviewData } from '../api'
import { Button, DisclaimerBanner, ErrorText, Loading, PageHeader, Panel, Stat } from '../components/ui'

export function OverviewPage() {
  const [data, setData] = useState<OverviewData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function refresh() {
    setError(null)
    try {
      setData(await apiGet<OverviewData>('/api/overview'))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  async function loadSample() {
    setBusy(true)
    setError(null)
    try {
      await apiSend('/api/dataset/load-default', 'POST')
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const classData = Object.entries(data?.class_counts || {}).map(([name, value]) => ({
    name,
    value,
  }))

  return (
    <div className="space-y-6">
      <PageHeader
        title="Insulin Matrix"
        subtitle="AI diabetes screening and classification platform for demonstration and model development."
        actions={
          <>
            <Button onClick={() => void refresh()} variant="ghost">
              Refresh
            </Button>
            <Button onClick={() => void loadSample()} disabled={busy}>
              {busy ? 'Loading…' : 'Load synthetic dataset'}
            </Button>
          </>
        }
      />
      <DisclaimerBanner synthetic={data?.is_synthetic} syntheticLabel={data?.synthetic_label} text={data?.disclaimer} />
      <ErrorText message={error} />
      {!data ? (
        <Loading label="Loading overview…" />
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Stat label="Rows loaded" value={data.n_rows} hint={data.dataset_name || 'No dataset'} />
            <Stat label="Trained models" value={data.trained_models.length} hint={data.active_model || 'None active'} />
            <Stat label="Predictions logged" value={data.storage_counts.predictions} />
            <Stat
              label="Best F1 (macro)"
              value={data.best_model?.f1_macro != null ? data.best_model.f1_macro.toFixed(3) : '—'}
              hint={data.best_model?.model_name || 'Train models to compare'}
            />
          </div>

          <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <Panel title="Class distribution">
              {classData.length ? (
                <div className="h-72 w-full min-h-[18rem]">
                  <ResponsiveContainer width="100%" height={288}>
                    <BarChart data={classData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#d5e2e6" />
                      <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} angle={-15} textAnchor="end" height={70} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#146b78" radius={[8, 8, 0, 0]} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="text-[var(--ink-soft)]">Load the synthetic sample to see live class counts.</p>
              )}
              {data.rare_class_warnings?.length ? (
                <ul className="mt-4 space-y-1 text-sm text-[var(--warn)]">
                  {data.rare_class_warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              ) : null}
            </Panel>

            <Panel title="Workflow">
              <ol className="m-0 space-y-3 p-0 text-sm text-[var(--ink-soft)]">
                <li>1. Confirm the synthetic dataset on Dataset.</li>
                <li>2. Run Preprocessing (train-only scaling / encoding).</li>
                <li>3. Train classical models and the ANN.</li>
                <li>4. Compare metrics, then score patients or batches.</li>
                <li>5. Download PDF / Excel reports from real results.</li>
              </ol>
              <div className="mt-5 flex flex-wrap gap-2">
                <Link to="/dataset"><Button variant="secondary">Dataset</Button></Link>
                <Link to="/training"><Button variant="secondary">Training</Button></Link>
                <Link to="/assessment"><Button>Patient Assessment</Button></Link>
              </div>
            </Panel>
          </div>
        </>
      )}
    </div>
  )
}
