import type { ReactNode } from 'react'
import { DISCLAIMER } from '../api'

export function DisclaimerBanner({
  synthetic,
  syntheticLabel,
  text,
}: {
  synthetic?: boolean
  syntheticLabel?: string | null
  text?: string
}) {
  return (
    <div className="rounded-2xl border border-[#f0c2c2] bg-[var(--warn-bg)] px-4 py-3 text-sm text-[var(--warn)]">
      <p className="m-0 font-medium">{text || DISCLAIMER}</p>
      {synthetic ? (
        <p className="mt-1 mb-0 opacity-90">
          {syntheticLabel || 'Dataset is synthetic / simulated. It is not real patient data.'}
        </p>
      ) : null}
    </div>
  )
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string
  subtitle?: string
  actions?: ReactNode
}) {
  return (
    <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="m-0 text-3xl text-[var(--brand)] md:text-4xl">{title}</h1>
        {subtitle ? <p className="mt-2 mb-0 max-w-3xl text-[var(--ink-soft)]">{subtitle}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null}
    </div>
  )
}

export function Panel({
  title,
  children,
  className = '',
}: {
  title?: string
  children: ReactNode
  className?: string
}) {
  return (
    <section
      className={`rounded-[28px] border border-[var(--line)] bg-white/90 p-5 shadow-[var(--shadow)] backdrop-blur ${className}`}
    >
      {title ? <h2 className="mt-0 mb-4 text-xl text-[var(--brand)]">{title}</h2> : null}
      {children}
    </section>
  )
}

export function Button({
  children,
  onClick,
  disabled,
  variant = 'primary',
  type = 'button',
  className = '',
}: {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  type?: 'button' | 'submit'
  className?: string
}) {
  const styles = {
    primary: 'bg-[var(--brand)] text-white hover:bg-[var(--brand-2)]',
    secondary: 'bg-[var(--accent-soft)] text-[var(--brand)] hover:bg-[#c7eae4]',
    ghost: 'bg-transparent text-[var(--brand)] border border-[var(--line)] hover:bg-white',
    danger: 'bg-[var(--warn-bg)] text-[var(--warn)] hover:bg-[#f8d4d4]',
  }[variant]
  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${styles} ${className}`}
    >
      {children}
    </button>
  )
}

export function Stat({
  label,
  value,
  hint,
}: {
  label: string
  value: string | number
  hint?: string
}) {
  return (
    <div className="rounded-2xl border border-[var(--line)] bg-[linear-gradient(180deg,#fff,rgba(216,240,236,0.35))] p-4">
      <div className="text-xs font-semibold tracking-[0.08em] text-[var(--ink-soft)] uppercase">
        {label}
      </div>
      <div className="display mt-2 text-3xl text-[var(--brand)]">{value}</div>
      {hint ? <div className="mt-1 text-sm text-[var(--ink-soft)]">{hint}</div> : null}
    </div>
  )
}

export function ErrorText({ message }: { message?: string | null }) {
  if (!message) return null
  return (
    <div className="rounded-xl border border-[#f0c2c2] bg-[var(--warn-bg)] px-3 py-2 text-sm text-[var(--warn)]">
      {message}
    </div>
  )
}

export function Loading({ label = 'Working…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-[var(--ink-soft)]">
      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-[var(--accent)] border-t-transparent" />
      {label}
    </div>
  )
}

export function Field({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <label className="flex flex-col gap-1.5 text-sm">
      <span className="font-medium text-[var(--ink-soft)]">{label}</span>
      {children}
    </label>
  )
}

export const inputClass =
  'rounded-xl border border-[var(--line)] bg-white px-3 py-2.5 outline-none transition focus:border-[var(--accent)]'
