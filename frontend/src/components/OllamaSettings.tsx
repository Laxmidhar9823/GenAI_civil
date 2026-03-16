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
  const { ollamaUrl, setOllamaUrl, model, setModel, status, onCheckNow } = props

  return (
    <div className="card">
      <div className="card-header">
        <h3>Backend Settings</h3>
        <StatusPill status={status} />
      </div>
      <div className="card-body">
        <div className="row">
          <div>
            <label className="label" htmlFor="ollamaUrl">
              Ollama Endpoint
            </label>
            <input
              id="ollamaUrl"
              className="input"
              value={ollamaUrl}
              onChange={(e) => setOllamaUrl(e.target.value)}
              placeholder="http://localhost:11434"
              inputMode="url"
              autoComplete="off"
            />
          </div>

          <div>
            <label className="label" htmlFor="ollamaModel">
              Model Tag
            </label>
            <input
              id="ollamaModel"
              className="input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="gemma3:12b"
              autoComplete="off"
            />
          </div>

          <div className="setting-actions">
            <button type="button" className="btn" style={{ fontSize: '12px', height: '32px' }} onClick={onCheckNow}>
              Check Connection
            </button>
          </div>

          {'detail' in status && status.detail ? (
            <div className="status-detail" role="status">
              {status.detail}
            </div>
          ) : null}

          {(status.kind === 'error' || status.kind === 'needs_model') && (
            <div className="status-detail warning" role="alert">
              Commands: <span className="kbd">ollama serve</span> <span className="kbd">ollama pull {model || 'modelname'}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
