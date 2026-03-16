import { useMemo } from 'react'

function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function toTxtReport(params: Record<string, unknown>) {
  const keys = Object.keys(params).sort((a, b) => a.localeCompare(b))
  const lines: string[] = []
  lines.push('Rigid Pavement Configuration')
  lines.push('')
  for (const k of keys) {
    lines.push(`${k}: ${String(params[k])}`)
  }
  lines.push('')
  lines.push(`Generated: ${new Date().toISOString()}`)
  return lines.join('\n')
}

export default function ParamsPanel(props: {
  progress: { done: number; total: number }
  paramInfo: Record<string, any> | null
  collected: Record<string, unknown> | null
  finalParams: Record<string, unknown> | null
}) {
  const { progress, paramInfo, collected, finalParams } = props

  const keys = useMemo(() => {
    if (paramInfo) return Object.keys(paramInfo)
    if (collected) return Object.keys(collected)
    return []
  }, [paramInfo, collected])

  const rows = useMemo(() => {
    if (!keys.length) return []
    const source = collected || {}
    return keys.map((k) => {
      const meta = paramInfo?.[k]
      const desc = meta?.description || meta?.desc || meta?.label || ''
      return {
        key: k,
        value: source[k],
        desc,
      }
    })
  }, [keys, collected, paramInfo])

  const percent = progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0

  return (
    <div className="card" aria-label="Progress and parameters">
      <div className="card-header">
        <h2>Configuration status</h2>
        <span className="pill pill-soft">{progress.total > 0 ? `${progress.done} / ${progress.total}` : '0 / 0'}</span>
      </div>

      <div className="progress-wrap">
        <div className="progress-container" aria-label="Progress bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
          <div className="progress-bar" style={{ width: `${percent}%` }} />
        </div>
      </div>

      <div className="card-body">
        {rows.length ? (
          <div className="table-wrap">
            <table className="table" aria-label="Collected parameters">
              <thead>
                <tr>
                  <th style={{ width: '20%' }}>Key</th>
                  <th style={{ width: '30%' }}>Value</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.key}>
                    <td>
                      <span className="kbd">{r.key}</span>
                    </td>
                    <td>{r.value === undefined || r.value === null || String(r.value) === '' ? <span className="sub">N/A</span> : String(r.value)}</td>
                    <td className="sub">{r.desc || ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="sub">Parameter table will appear after the conversation begins.</div>
        )}

        {finalParams ? (
          <div className="final-actions">
            <div className="pill" aria-label="Final configuration ready">
              <span className="dot ok" aria-hidden="true" />
              <span>Final configuration ready</span>
            </div>

            <div className="final-downloads">
              <button
                type="button"
                className="btn primary"
                onClick={() => {
                  const blob = new Blob([JSON.stringify(finalParams, null, 2)], { type: 'application/json' })
                  downloadBlob('pavement_config.json', blob)
                }}
              >
                Download JSON
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  const blob = new Blob([toTxtReport(finalParams)], { type: 'text/plain;charset=utf-8' })
                  downloadBlob('pavement_config.txt', blob)
                }}
              >
                Download TXT report
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
