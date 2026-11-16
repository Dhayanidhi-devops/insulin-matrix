import { useEffect, useState } from 'react'
import { apiGet, apiSend } from '../api'
import { Button, DisclaimerBanner, ErrorText, Field, PageHeader, Panel, inputClass } from '../components/ui'

type Catalog = {
  classical: Array<{ name: string; key: string }>
  xgboost_available: boolean
}

export function TrainingPage() {
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [selected, setSelected] = useState<string[]>(['Logistic Regression', 'Random Forest', 'Gradient Boosting'])
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [epochs, setEpochs] = useState(20)

  useEffect(() => {
    void apiGet<Catalog>('/api/models/catalog')
      .then((c) => {
        setCatalog(c)
        if (c.xgboost_available) {
          setSelected((prev) => (prev.includes('XGBoost') ? prev : [...prev, 'XGBoost']))
        }
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  function toggle(name: string) {
    setSelected((prev) => (prev.includes(name) ? prev.filter((x) => x !== name) : [...prev, name]))
  }

  async function trainClassical() {
    setBusy(true)
    setError(null)
    try {
      // Ensure preprocessing exists
      await apiSend('/api/preprocess', 'POST', {})
      const res = await apiSend<Record<string, unknown>>('/api/train', 'POST', {
        model_names: selected,
      })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function trainAnn() {
    setBusy(true)
    setError(null)
    try {
      await apiSend('/api/preprocess', 'POST', {})
      const res = await apiSend<Record<string, unknown>>('/api/ann/train', 'POST', {
        epochs,
        hidden_layers: [64, 32],
        dropout: 0.2,
        batch_size: 64,
      })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Training"
        subtitle="Train real scikit-learn models and a Keras/TensorFlow ANN on the loaded dataset."
        actions={
          <>
            <Button variant="secondary" disabled={busy} onClick={() => void trainClassical()}>
              {busy ? 'Training…' : 'Train selected'}
            </Button>
            <Button disabled={busy} onClick={() => void trainAnn()}>
              {busy ? 'Training…' : 'Train ANN'}
            </Button>
          </>
        }
      />
      <DisclaimerBanner />
      <ErrorText message={error} />

      <Panel title="Classical models">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {(catalog?.classical || []).map((m) => (
            <label key={m.key} className="flex items-center gap-3 rounded-xl border border-[var(--line)] px-3 py-3">
              <input type="checkbox" checked={selected.includes(m.name)} onChange={() => toggle(m.name)} />
              <span className="font-medium">{m.name}</span>
            </label>
          ))}
        </div>
        <p className="mt-3 text-sm text-[var(--ink-soft)]">
          XGBoost {catalog?.xgboost_available ? 'is available' : 'is not installed in this environment'}.
        </p>
      </Panel>

      <Panel title="ANN options">
        <div className="max-w-xs">
          <Field label="Epochs">
            <input className={inputClass} type="number" min={1} max={100} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
          </Field>
        </div>
      </Panel>

      {result ? (
        <Panel title="Latest training result">
          <pre className="m-0 max-h-[28rem] overflow-auto rounded-xl bg-[#0b3d4a] p-4 text-xs text-[#d8f0ec]">
            {JSON.stringify(result, null, 2)}
          </pre>
        </Panel>
      ) : null}
    </div>
  )
}
