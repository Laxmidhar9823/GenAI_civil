type FetchJsonOptions = {
  method?: 'GET' | 'POST'
  body?: unknown
  signal?: AbortSignal
}
import type {
  ChatResponse,
  ConversationState,
  ErrorResponse,
  OllamaStatusResponse,
  ResetResponse,
  SchemaResponse,
} from './types'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(
  /\/$/,
  '',
)

function stringifyUnknown(v: unknown): string {
  if (typeof v === 'string') return v
  if (v === undefined || v === null) return ''
  try {
    return JSON.stringify(v)
  } catch {
    return String(v)
  }
}

function parseErrorMessage(json: unknown): string {
  if (!json || typeof json !== 'object') return ''

  const payload = json as Partial<ErrorResponse> & { detail?: unknown; message?: unknown }

  if (payload.error && typeof payload.error === 'object') {
    const msg = typeof payload.error.message === 'string' ? payload.error.message : ''
    const details = stringifyUnknown(payload.error.details)
    if (msg && details) return `${msg} (${details})`
    if (msg) return msg
    if (details) return details
  }

  if (typeof payload.detail === 'string') return payload.detail
  if (typeof payload.message === 'string') return payload.message
  if (payload.detail !== undefined) return stringifyUnknown(payload.detail)

  return stringifyUnknown(json)
}

async function fetchJson<T>(path: string, options: FetchJsonOptions = {}): Promise<T> {
  const url = `${API_BASE_URL}${path}`

  const res = await fetch(url, {
    method: options.method ?? 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
    signal: options.signal,
  })

  if (!res.ok) {
    // Try to read a useful error message from JSON or text.
    const contentType = res.headers.get('content-type') || ''
    let detail = ''
    try {
      if (contentType.includes('application/json')) {
        const json = await res.json()
        detail = parseErrorMessage(json)
      } else {
        detail = await res.text()
      }
    } catch {
      // ignore
    }

    throw new Error(`Request failed (${res.status} ${res.statusText})${detail ? `: ${detail}` : ''}`)
  }

  const contentType = res.headers.get('content-type') || ''
  if (!contentType.includes('application/json')) {
    // Backend should return JSON; if not, surface it.
    const text = await res.text()
    throw new Error(`Expected JSON from ${path}, got: ${text.slice(0, 200)}`)
  }

  return (await res.json()) as T
}

export type LLMConfig = {
  ollama_url: string
  model: string
}

export async function apiReset(): Promise<ResetResponse> {
  return fetchJson('/reset', { method: 'POST', body: {} })
}

export async function apiChat(payload: {
  user_input: string
  state: ConversationState
  llm_config: LLMConfig
}): Promise<ChatResponse> {
  return fetchJson('/chat', { method: 'POST', body: payload })
}

export async function apiGetSchema(): Promise<SchemaResponse> {
  return fetchJson('/config/schema')
}

export async function apiOllamaStatus(params: { ollama_url: string; model: string }): Promise<OllamaStatusResponse> {
  const q = new URLSearchParams({
    ollama_url: params.ollama_url,
    model: params.model,
  }).toString()
  return fetchJson<OllamaStatusResponse>(`/ollama/status?${q}`)
}
