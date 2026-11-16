import { useEffect, useState } from 'react'
import { apiGet, apiSend, apiUpload } from '../api'
import { Button, DisclaimerBanner, ErrorText, Field, Loading, PageHeader, Panel, Stat, inputClass } from '../components/ui'

type Status = {
  loaded: boolean
  display_name?: string
  is_synthetic?: boolean
  synthetic_label?: string
  disclaimer?: string
  n_rows?: number
  n_columns?: number
  class_counts?: Record<string, number>
  rare_class_warnings?: string[]
  features?: { default: string[]; excluded_leakage: string[] }
  sample_available?: boolean
}

export function DatasetPage() {
  const [status, setStatus] = useState<Status | null>(null)
  const [preview, setPreview] = useState<{ columns: string[]; rows: Record<string, unknown>[] } | null>(null)
  const [stats, setStats] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function refresh() {
    const s = await apiGet<Status>('/api/dataset/status')
    setStatus(s)
    if (s.loaded) {
      setPreview(await apiGet('/api/dataset/preview?limit=20'))
      setStats(await apiGet('/api/dataset/stats'))
    } else {
      setPreview(null)
      setStats(null)
    }
  }

  useEffect(() => {
    void refresh().catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  async function loadDefault() {
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

  async function onUpload(file: File | null) {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      await apiUpload('/api/dataset/upload', file, { mark_synthetic: 'false' })
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dataset"
        subtitle="Load the bundled synthetic workbook or upload your own Excel/CSV."
        actions={
          <>
            <Button onClick={() => void loadDefault()} disabled={busy || !status?.sample_available}>
              Load synthetic sample
            </Button>
            <label className="inline-flex cursor-pointer items-center justify-center rounded-xl bg-[var(--accent-soft)] px-4 py-2.5 text-sm font-semibold text-[var(--brand)]">
              Upload file
              <input
                type="file"
                accept=".xlsx,.xlsm,.csv"
                className="hidden"
                onChange={(e) => void onUpload(e.target.files?.[0] || null)}
              />
            </label>
          </>
        }
      />
      <DisclaimerBanner
        synthetic={status?.is_synthetic}
        syntheticLabel={status?.synthetic_label}
        text={status?.disclaimer}
      />
      <ErrorText message={error} />

      {!status ? (
        <Loading />
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <Stat label="Status" value={status.loaded ? 'Loaded' : 'Empty'} hint={status.display_name || '—'} />
            <Stat label="Rows" value={status.n_rows || 0} />
            <Stat label="Columns" value={status.n_columns || 0} />
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <Panel title="Target classes">
              <ul className="m-0 space-y-2 p-0">
                {Object.entries(status.class_counts || {}).map(([k, v]) => (
                  <li key={k} className="flex justify-between border-b border-[var(--line)] py-2 text-sm">
                    <span>{k}</span>
                    <strong>{v}</strong>
                  </li>
                ))}
              </ul>
              {status.rare_class_warnings?.map((w) => (
                <p key={w} className="mb-0 text-sm text-[var(--warn)]">{w}</p>
              ))}
            </Panel>
            <Panel title="Feature policy">
              <p className="mt-0 text-sm text-[var(--ink-soft)]">
                Default inputs exclude identifiers and target-derived columns to avoid leakage.
              </p>
              <Field label="Default features">
                <div className="rounded-xl border border-[var(--line)] bg-[#f7fbfb] p-3 text-sm">
                  {(status.features?.default || []).join(', ') || '—'}
                </div>
              </Field>
              <div className="mt-3">
                <Field label="Excluded (leakage / IDs)">
                  <div className={`${inputClass} text-sm`}>
                    {(status.features?.excluded_leakage || []).join(', ') || '—'}
                  </div>
                </Field>
              </div>
            </Panel>
          </div>

          <Panel title="Preview">
            {!preview ? (
              <p className="text-[var(--ink-soft)]">No dataset loaded.</p>
            ) : (
              <div className="overflow-auto">
                <table className="min-w-full text-left text-sm">
                  <thead>
                    <tr>
                      {preview.columns.map((c) => (
                        <th key={c} className="sticky top-0 bg-[#0b3d4a] px-3 py-2 font-semibold text-white">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, idx) => (
                      <tr key={idx} className="odd:bg-white even:bg-[#f4f8f9]">
                        {preview.columns.map((c) => (
                          <td key={c} className="whitespace-nowrap px-3 py-2">
                            {String(row[c] ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>

          {stats ? (
            <Panel title="Profile payload">
              <pre className="m-0 max-h-80 overflow-auto rounded-xl bg-[#0b3d4a] p-4 text-xs text-[#d8f0ec]">
                {JSON.stringify((stats as { profile?: unknown }).profile || stats, null, 2)}
              </pre>
            </Panel>
          ) : null}
        </>
      )}
    </div>
  )
}
