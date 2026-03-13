type FetchJsonOptions = {
  method?: 'GET' | 'POST'
  body?: unknown
  signal?: AbortSignal
}
import type { OllamaStatusResponse } from './types'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(
  /\/$/,
  '',
)

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
        detail = json?.detail ? String(json.detail) : JSON.stringify(json)
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

export type ResetResponse = {
  assistant_message: string
  state: unknown
}

export type ChatResponse = {
  assistant_message?: string
  state: unknown
  final_params?: Record<string, unknown>
}

export async function apiReset(): Promise<ResetResponse> {
  return fetchJson('/reset', { method: 'POST', body: {} })
}

export async function apiChat(payload: {
  user_input: string
  state: unknown
  llm_config: LLMConfig
}): Promise<ChatResponse> {
  return fetchJson('/chat', { method: 'POST', body: payload })
}

export async function apiGetSchema(): Promise<unknown> {
  return fetchJson('/config/schema')
}

export async function apiOllamaStatus(params: { ollama_url: string; model: string }): Promise<OllamaStatusResponse> {
  const q = new URLSearchParams({
    ollama_url: params.ollama_url,
    model: params.model,
  }).toString()
  return fetchJson<OllamaStatusResponse>(`/ollama/status?${q}`)
}
