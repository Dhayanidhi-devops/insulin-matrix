import { useEffect, useState } from 'react'
import { apiGet, apiSend } from '../api'
import { Button, DisclaimerBanner, ErrorText, Field, Loading, PageHeader, Panel, Stat, inputClass } from '../components/ui'

export function PreprocessingPage() {
  const [status, setStatus] = useState<Record<string, unknown> | null>(null)
  const [featuresText, setFeaturesText] = useState('')
  const [testSize, setTestSize] = useState(0.2)
  const [seed, setSeed] = useState(42)
  const [scale, setScale] = useState(true)
  const [impute, setImpute] = useState('median')
  const [report, setReport] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void (async () => {
      try {
        const ds = await apiGet<{ features?: { default: string[] }; loaded: boolean }>('/api/dataset/status')
        if (ds.loaded && ds.features?.default) setFeaturesText(ds.features.default.join(', '))
        const pre = await apiGet<{ report: Record<string, unknown> | null; config: Record<string, unknown> }>('/api/preprocess/status')
        setStatus(pre)
        if (pre.report) setReport(pre.report)
        if (pre.config?.features) setFeaturesText((pre.config.features as string[]).join(', '))
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      }
    })()
  }, [])

  async function run() {
    setBusy(true)
    setError(null)
    try {
      const features = featuresText.split(',').map((s) => s.trim()).filter(Boolean)
      const result = await apiSend<Record<string, unknown>>('/api/preprocess', 'POST', {
        features,
        scale_numeric: scale,
        numeric_impute: impute,
        test_size: testSize,
        seed,
      })
      setReport(result)
      setStatus(await apiGet('/api/preprocess/status'))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Preprocessing"
        subtitle="Imputation, scaling, and encoding are fit on the training split only."
        actions={<Button onClick={() => void run()} disabled={busy}>{busy ? 'Running…' : 'Run preprocessing'}</Button>}
      />
      <DisclaimerBanner />
      <ErrorText message={error} />

      <Panel title="Configuration">
        <div className="grid gap-4 md:grid-cols-2">
          <Field label="Features (comma-separated)">
            <textarea className={`${inputClass} min-h-28`} value={featuresText} onChange={(e) => setFeaturesText(e.target.value)} />
          </Field>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Test size">
              <input className={inputClass} type="number" step="0.05" min="0.05" max="0.5" value={testSize} onChange={(e) => setTestSize(Number(e.target.value))} />
            </Field>
            <Field label="Seed">
              <input className={inputClass} type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </Field>
            <Field label="Numeric impute">
              <select className={inputClass} value={impute} onChange={(e) => setImpute(e.target.value)}>
                <option value="median">median</option>
                <option value="mean">mean</option>
              </select>
            </Field>
            <Field label="Scale numeric">
              <select className={inputClass} value={scale ? 'yes' : 'no'} onChange={(e) => setScale(e.target.value === 'yes')}>
                <option value="yes">yes</option>
                <option value="no">no</option>
              </select>
            </Field>
          </div>
        </div>
      </Panel>

      {!report && !status ? <Loading /> : null}

      {report ? (
        <>
          <div className="grid gap-4 md:grid-cols-4">
            <Stat label="Train rows" value={String(report.n_train ?? '—')} />
            <Stat label="Test rows" value={String(report.n_test ?? '—')} />
            <Stat label="Stratified" value={report.stratified ? 'Yes' : 'No'} />
            <Stat label="Transformed width" value={String((report.transformed_train_shape as number[] | undefined)?.[1] ?? '—')} />
          </div>
          <Panel title="Split class counts">
            <div className="grid gap-4 md:grid-cols-2">
              <pre className="m-0 overflow-auto rounded-xl bg-[#f7fbfb] p-3 text-sm">{JSON.stringify(report.class_counts_train, null, 2)}</pre>
              <pre className="m-0 overflow-auto rounded-xl bg-[#f7fbfb] p-3 text-sm">{JSON.stringify(report.class_counts_test, null, 2)}</pre>
            </div>
          </Panel>
        </>
      ) : (
        <Panel title="Not run yet">
          <p className="m-0 text-[var(--ink-soft)]">Run preprocessing before training to inspect the held-out split.</p>
        </Panel>
      )}
    </div>
  )
}
