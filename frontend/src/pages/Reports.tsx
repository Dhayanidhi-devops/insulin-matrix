import { useEffect, useState } from 'react'
import { apiGet, downloadBlob } from '../api'
import { Button, DisclaimerBanner, ErrorText, Field, PageHeader, Panel, inputClass } from '../components/ui'

export function ReportsPage() {
  const [history, setHistory] = useState<Array<{ prediction_id: string; predicted_class: string; model_name: string }>>([])
  const [predictionId, setPredictionId] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void apiGet<{ items: Array<{ prediction_id: string; predicted_class: string; model_name: string }> }>('/api/history?limit=50')
      .then((data) => {
        setHistory(data.items)
        if (data.items[0]) setPredictionId(data.items[0].prediction_id)
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  async function pdf() {
    setBusy(true)
    setError(null)
    try {
      await downloadBlob('/api/reports/pdf', 'Insulin_Matrix_Report.pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prediction_id: predictionId || null,
          include_comparison: true,
        }),
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function excel() {
    setBusy(true)
    setError(null)
    try {
      await downloadBlob('/api/reports/excel', 'Insulin_Matrix_Comparison.xlsx')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Reports"
        subtitle="Download ReportLab PDFs and OpenPyXL Excel workbooks generated from real results."
      />
      <DisclaimerBanner />
      <ErrorText message={error} />

      <Panel title="PDF report">
        <Field label="Optional prediction ID">
          <select className={inputClass} value={predictionId} onChange={(e) => setPredictionId(e.target.value)}>
            <option value="">Comparison only</option>
            {history.map((h) => (
              <option key={h.prediction_id} value={h.prediction_id}>
                {h.prediction_id} · {h.model_name} · {h.predicted_class}
              </option>
            ))}
          </select>
        </Field>
        <div className="mt-4 flex flex-wrap gap-2">
          <Button disabled={busy} onClick={() => void pdf()}>Download PDF</Button>
          <Button variant="secondary" disabled={busy} onClick={() => void excel()}>Download Excel comparison</Button>
        </div>
      </Panel>
    </div>
  )
}
