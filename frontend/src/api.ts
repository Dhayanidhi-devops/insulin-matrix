export const DISCLAIMER =
  'Demonstration only — not a medical diagnosis. Predictions are generated from synthetic or research data for model-development education. Do not use outputs for clinical decision-making, treatment, or insulin dosing.'

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json()
    if (typeof data?.detail === 'string') return data.detail
    if (Array.isArray(data?.detail)) return data.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join('; ')
    return JSON.stringify(data)
  } catch {
    return res.statusText || 'Request failed'
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function apiSend<T>(
  path: string,
  method: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function apiUpload<T>(
  path: string,
  file: File,
  fields?: Record<string, string>,
): Promise<T> {
  const form = new FormData()
  form.append('file', file)
  if (fields) {
    Object.entries(fields).forEach(([k, v]) => form.append(k, v))
  }
  const res = await fetch(path, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function downloadBlob(path: string, filename: string, init?: RequestInit) {
  const res = await fetch(path, init)
  if (!res.ok) throw new Error(await parseError(res))
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export type Overview = {
  app: string
  version: string
  disclaimer: string
  synthetic_label?: string | null
  dataset_loaded: boolean
  dataset_name?: string | null
  is_synthetic: boolean
  n_rows: number
  class_counts: Record<string, number>
  rare_class_warnings: string[]
  storage_counts: { datasets: number; training_runs: number; predictions: number }
  trained_models: string[]
  active_model?: string | null
  best_model?: {
    model_name: string
    accuracy?: number
    f1_macro?: number
  } | null
  comparison: Array<Record<string, unknown>>
  preprocess_ready: boolean
}
