import StatusPill from './StatusPill'
import type { OllamaCheckState } from '../lib/types'

export default function OllamaSettings(props: {
  ollamaUrl: string
  setOllamaUrl: (v: string) => void
  model: string
  setModel: (v: string) => void
  status: OllamaCheckState
  onCheckNow: () => void
  apiBaseUrl: string
}) {
  const { ollamaUrl, setOllamaUrl, model, setModel, status, onCheckNow, apiBaseUrl } = props

  return (
    <div className="card" aria-label="LLM settings">
      <div className="card-header">
        <h2>Ollama settings</h2>
        <StatusPill status={status} />
      </div>
      <div className="card-body">
        <div className="row">
          <div>
            <label className="label" htmlFor="ollamaUrl">Ollama URL</label>
            <input
              id="ollamaUrl"
              className="input"
              value={ollamaUrl}
              onChange={(e) => setOllamaUrl(e.target.value)}
              placeholder="http://localhost:11434"
              inputMode="url"
              autoComplete="off"
            />
            <div className="help">Example: <span className="kbd">http://localhost:11434</span></div>
          </div>

          <div>
            <label className="label" htmlFor="ollamaModel">Model</label>
            <input
              id="ollamaModel"
              className="input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="gemma3:12b"
              autoComplete="off"
            />
            <div className="help">Example: <span className="kbd">gemma3:12b</span></div>
          </div>

          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <button type="button" className="btn" onClick={onCheckNow}>
              Check status
            </button>
            <div className="help" style={{ marginTop: 8 }}>
              Backend API: <span className="kbd">{apiBaseUrl}</span>
            </div>
          </div>
          {'detail' in status && status.detail ? (
            <div className="help" role="status" aria-live="polite" style={{ lineHeight: 1.55 }}>
              {status.detail}
            </div>
          ) : null}
          {(status.kind === 'error' || status.kind === 'needs_model') && (
            <div className="help" role="alert" aria-live="polite">
              Commands:{' '}
              <span className="kbd">ollama serve</span> <span className="kbd">ollama list</span>{' '}
              <span className="kbd">ollama pull {model}</span>
            </div>
          )}
          {status.kind === 'needs_model' ? (
            <div className="help" role="alert" aria-live="polite">
              Model <span className="kbd">{status.model}</span> is not installed. Run{' '}
              <span className="kbd">ollama pull {status.model}</span>, then check status again.
              {status.availableModels.length ? ` Available now: ${status.availableModels.slice(0, 5).join(', ')}.` : ''}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
