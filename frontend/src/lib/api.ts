type FetchJsonOptions = {
  method?: 'GET' | 'POST'
  body?: unknown
  signal?: AbortSignal
  headers?: Record<string, string>
}
import type {
  ChatResponse,
  ConversationState,
  ErrorResponse,
  OllamaStatusResponse,
  ResetResponse,
  SchemaResponse,
} from './types'

const configuredBaseUrl = (import.meta.env.VITE_API_BASE_URL || '').trim()

export const API_BASE_URL = (configuredBaseUrl || 'http://localhost:8000').replace(
  /\/$/,
  '',
)

function candidateApiBaseUrls(): string[] {
  if (configuredBaseUrl) return [API_BASE_URL]

  const out = new Set<string>([API_BASE_URL])
  const host = window.location.hostname

  if (host === '127.0.0.1') out.add('http://127.0.0.1:8000')
  if (host === 'localhost') out.add('http://localhost:8000')
  if (host === 'localhost') out.add('http://127.0.0.1:8000')
  if (host === '127.0.0.1') out.add('http://localhost:8000')

  return [...out]
}

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
  const baseUrls = candidateApiBaseUrls()
  let lastError: Error | null = null

  for (const baseUrl of baseUrls) {
    const url = `${baseUrl}${path}`

    try {
      const res = await fetch(url, {
        method: options.method ?? 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...(options.headers ?? {}),
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
    } catch (e) {
      lastError = e as Error
    }
  }

  throw lastError ?? new Error(`Request failed for ${path}`)
}

export type LLMConfig = {
  ollama_url: string
  model: string
  api_key?: string
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

export async function apiHealth(): Promise<{ status: string; service?: string; version?: string; [key: string]: unknown }> {
  return fetchJson('/health')
}

export async function apiOllamaStatus(params: { ollama_url: string; model: string }): Promise<OllamaStatusResponse> {
  const q = new URLSearchParams({
    ollama_url: params.ollama_url,
    model: params.model,
  }).toString()
  return fetchJson<OllamaStatusResponse>(`/ollama/status?${q}`)
}
