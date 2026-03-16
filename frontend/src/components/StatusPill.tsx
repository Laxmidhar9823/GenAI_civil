import type { OllamaCheckState } from '../lib/types'

export default function StatusPill({ status }: { status: OllamaCheckState }) {
  const dotClass =
    status.kind === 'ok'
      ? 'dot ok'
      : status.kind === 'needs_model'
        ? 'dot warn'
        : status.kind === 'error'
          ? 'dot err'
          : status.kind === 'checking'
            ? 'dot check'
            : 'dot'

  const label =
    status.kind === 'ok'
      ? 'Ready'
      : status.kind === 'needs_model'
        ? 'Model missing'
        : status.kind === 'error'
          ? 'Ollama unavailable'
          : status.kind === 'checking'
            ? 'Checking...'
            : 'Not checked'

  return (
    <span className="pill" role="status" aria-live="polite" aria-label={label}>
      <span className={dotClass} aria-hidden="true" />
      <span>{label}</span>
    </span>
  )
}
