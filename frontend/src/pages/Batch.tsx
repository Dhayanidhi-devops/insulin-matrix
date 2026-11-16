import { useEffect, useState } from 'react'
import { apiGet, apiUpload } from '../api'
import { DisclaimerBanner, ErrorText, Field, PageHeader, Panel, inputClass } from '../components/ui'

export function BatchPage() {
  const [models, setModels] = useState<string[]>([])
  const [modelName, setModelName] = useState('')
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void apiGet<{ trained_models: string[]; active_model?: string }>('/api/assessment/schema')
      .then((s) => {
        setModels(s.trained_models)
        setModelName(s.active_model || s.trained_models[0] || '')
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  async function onUpload(file: File | null) {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const res = await apiUpload<Record<string, unknown>>('/api/predict/batch', file, {
        model_name: modelName,
      })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const preview = (result?.preview || []) as Record<string, unknown>[]
  const columns = preview[0] ? Object.keys(preview[0]) : []

  return (
    <div className="space-y-6">
      <PageHeader
        title="Batch Prediction"
        subtitle="Upload an Excel/CSV with the same feature columns used in training."
      />
      <DisclaimerBanner
        synthetic={Boolean(result?.is_synthetic_context)}
        syntheticLabel={result?.synthetic_label as string | undefined}
        text={(result?.disclaimer as string) || undefined}
      />
      <ErrorText message={error} />

      <Panel title="Batch upload">
        <div className="grid gap-4 md:grid-cols-[1fr_auto] md:items-end">
          <Field label="Model">
            <select className={inputClass} value={modelName} onChange={(e) => setModelName(e.target.value)}>
              {!models.length ? <option value="">Train a model first</option> : null}
              {models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </Field>
          <label className="inline-flex cursor-pointer items-center justify-center rounded-xl bg-[var(--brand)] px-4 py-2.5 text-sm font-semibold text-white">
            {busy ? 'Scoring…' : 'Upload & predict'}
            <input
              type="file"
              accept=".xlsx,.xlsm,.csv"
              className="hidden"
              disabled={busy || !models.length}
              onChange={(e) => void onUpload(e.target.files?.[0] || null)}
            />
          </label>
        </div>
      </Panel>

      {result ? (
        <>
          <Panel title="Class summary">
            <pre className="m-0 rounded-xl bg-[#f7fbfb] p-3 text-sm">{JSON.stringify(result.class_summary, null, 2)}</pre>
            <p className="text-sm text-[var(--ink-soft)]">{String(result.n_rows)} rows scored with {String(result.model_name)}.</p>
          </Panel>
          <Panel title="Preview">
            <div className="overflow-auto">
              <table className="min-w-full text-left text-sm">
                <thead>
                  <tr>
                    {columns.map((c) => (
                      <th key={c} className="bg-[var(--brand)] px-3 py-2 text-white">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((row, idx) => (
                    <tr key={idx} className="odd:bg-white even:bg-[#f4f8f9]">
                      {columns.map((c) => (
                        <td key={c} className="whitespace-nowrap px-3 py-2">{String(row[c] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      ) : null}
    </div>
  )
}
