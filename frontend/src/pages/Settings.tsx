import { useEffect, useState } from 'react'
import { apiGet, apiSend } from '../api'
import { Button, DisclaimerBanner, ErrorText, Field, PageHeader, Panel, inputClass } from '../components/ui'

export function SettingsPage() {
  const [settings, setSettings] = useState<Record<string, unknown>>({})
  const [meta, setMeta] = useState<{ app?: string; version?: string; disclaimer?: string }>({})
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    void apiGet<{ settings: Record<string, unknown>; app: string; version: string; disclaimer: string }>('/api/settings')
      .then((data) => {
        setSettings(data.settings)
        setMeta({ app: data.app, version: data.version, disclaimer: data.disclaimer })
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  async function save() {
    setError(null)
    setSaved(false)
    try {
      const res = await apiSend<{ settings: Record<string, unknown> }>('/api/settings', 'PUT', settings)
      setSettings(res.settings)
      setSaved(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        subtitle={`${meta.app || 'Insulin Matrix AI'} v${meta.version || ''}`}
        actions={<Button onClick={() => void save()}>Save settings</Button>}
      />
      <DisclaimerBanner text={meta.disclaimer} />
      <ErrorText message={error} />
      {saved ? <div className="rounded-xl bg-[var(--accent-soft)] px-3 py-2 text-sm text-[var(--brand)]">Settings saved.</div> : null}

      <Panel title="Training defaults">
        <div className="grid gap-4 md:grid-cols-2">
          <Field label="Default test size">
            <input
              className={inputClass}
              type="number"
              step="0.05"
              min="0.05"
              max="0.5"
              value={Number(settings.default_test_size ?? 0.2)}
              onChange={(e) => setSettings((s) => ({ ...s, default_test_size: Number(e.target.value) }))}
            />
          </Field>
          <Field label="Default seed">
            <input
              className={inputClass}
              type="number"
              value={Number(settings.default_seed ?? 42)}
              onChange={(e) => setSettings((s) => ({ ...s, default_seed: Number(e.target.value) }))}
            />
          </Field>
          <Field label="Scale numeric features">
            <select
              className={inputClass}
              value={settings.scale_numeric ? 'yes' : 'no'}
              onChange={(e) => setSettings((s) => ({ ...s, scale_numeric: e.target.value === 'yes' }))}
            >
              <option value="yes">yes</option>
              <option value="no">no</option>
            </select>
          </Field>
          <Field label="Numeric imputation">
            <select
              className={inputClass}
              value={String(settings.numeric_impute ?? 'median')}
              onChange={(e) => setSettings((s) => ({ ...s, numeric_impute: e.target.value }))}
            >
              <option value="median">median</option>
              <option value="mean">mean</option>
            </select>
          </Field>
          <Field label="Auto-load synthetic sample on startup">
            <select
              className={inputClass}
              value={settings.auto_load_sample ? 'yes' : 'no'}
              onChange={(e) => setSettings((s) => ({ ...s, auto_load_sample: e.target.value === 'yes' }))}
            >
              <option value="yes">yes</option>
              <option value="no">no</option>
            </select>
          </Field>
        </div>
      </Panel>
    </div>
  )
}
