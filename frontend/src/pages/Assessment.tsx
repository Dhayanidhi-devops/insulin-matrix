import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { apiGet, apiSend, downloadBlob } from '../api'
import { Button, DisclaimerBanner, ErrorText, Field, PageHeader, Panel, inputClass } from '../components/ui'

type FieldDef = {
  key: string
  label: string
  type: 'number' | 'select'
  options?: string[]
  min?: number
  max?: number
  step?: number
}

const DEFAULTS: Record<string, string | number> = {
  Age_years: 52,
  Sex: 'Female',
  Height_cm: 165,
  Weight_kg: 78,
  BMI_kg_m2: 28.6,
  Waist_Circumference_cm: 92,
  Family_History_Diabetes: 'Yes',
  Physical_Activity_Level: 'Low',
  Fasting_Glucose_mg_dL: 118,
  HbA1c_percent: 6.1,
  C_Peptide_ng_mL: 1.8,
  Fasting_Insulin_uIU_mL: 14,
  Systolic_BP_mmHg: 132,
  Diastolic_BP_mmHg: 82,
  Triglycerides_mg_dL: 160,
  HDL_mg_dL: 42,
}

export function AssessmentPage() {
  const [fields, setFields] = useState<FieldDef[]>([])
  const [models, setModels] = useState<string[]>([])
  const [modelName, setModelName] = useState<string>('')
  const [form, setForm] = useState<Record<string, string | number>>(DEFAULTS)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void (async () => {
      const schema = await apiGet<{
        fields: FieldDef[]
        trained_models: string[]
        active_model?: string
      }>('/api/assessment/schema')
      setFields(schema.fields)
      setModels(schema.trained_models)
      setModelName(schema.active_model || schema.trained_models[0] || '')
    })().catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  const probData = useMemo(() => {
    const probs = (result?.probabilities || {}) as Record<string, number>
    return Object.entries(probs).map(([name, value]) => ({ name, value }))
  }, [result])

  async function predict() {
    setBusy(true)
    setError(null)
    try {
      const payload: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(form)) {
        payload[k] = typeof v === 'string' && fields.find((f) => f.key === k)?.type === 'number' ? Number(v) : v
      }
      const res = await apiSend<Record<string, unknown>>('/api/predict', 'POST', {
        model_name: modelName || null,
        features: payload,
      })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function downloadPdf() {
    if (!result?.prediction_id) return
    await downloadBlob('/api/reports/pdf', 'Insulin_Matrix_Prediction.pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prediction_id: result.prediction_id, include_comparison: true }),
    })
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Patient Assessment"
        subtitle="Score an individual profile with the active trained model."
        actions={
          <>
            <Button disabled={busy || !models.length} onClick={() => void predict()}>
              {busy ? 'Scoring…' : 'Run prediction'}
            </Button>
            <Button variant="secondary" disabled={!result?.prediction_id} onClick={() => void downloadPdf()}>
              Download PDF
            </Button>
          </>
        }
      />
      <DisclaimerBanner
        synthetic={Boolean(result?.is_synthetic_context)}
        syntheticLabel={result?.synthetic_label as string | undefined}
        text={(result?.disclaimer as string) || undefined}
      />
      <ErrorText message={error} />

      <Panel title="Model">
        <Field label="Active / selected model">
          <select className={inputClass} value={modelName} onChange={(e) => setModelName(e.target.value)}>
            {!models.length ? <option value="">Train a model first</option> : null}
            {models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </Field>
      </Panel>

      <Panel title="Clinical inputs">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {fields.map((field) => (
            <Field key={field.key} label={field.label}>
              {field.type === 'select' ? (
                <select
                  className={inputClass}
                  value={String(form[field.key] ?? '')}
                  onChange={(e) => setForm((prev) => ({ ...prev, [field.key]: e.target.value }))}
                >
                  {(field.options || []).map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              ) : (
                <input
                  className={inputClass}
                  type="number"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={form[field.key] ?? ''}
                  onChange={(e) => setForm((prev) => ({ ...prev, [field.key]: e.target.value }))}
                />
              )}
            </Field>
          ))}
        </div>
      </Panel>

      {result ? (
        <div className="grid gap-6 xl:grid-cols-2">
          <Panel title="Prediction">
            <div className="display text-4xl text-[var(--brand)]">{String(result.predicted_class)}</div>
            <p className="text-sm text-[var(--ink-soft)]">ID: {String(result.prediction_id)}</p>
            {probData.length ? (
              <div className="mt-4 h-64 w-full min-h-[16rem]">
                <ResponsiveContainer width="100%" height={256}>
                  <BarChart data={probData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#d5e2e6" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-15} textAnchor="end" height={70} />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#146b78" radius={[8, 8, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : null}
          </Panel>
          <Panel title="Explainability">
            <pre className="m-0 max-h-96 overflow-auto rounded-xl bg-[#f7fbfb] p-3 text-sm">
              {JSON.stringify(result.explanation, null, 2)}
            </pre>
          </Panel>
        </div>
      ) : null}
    </div>
  )
}
