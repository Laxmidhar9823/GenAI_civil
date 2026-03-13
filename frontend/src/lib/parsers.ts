import type { ChatMessage } from './types'

function safeString(v: unknown): string {
  if (typeof v === 'string') return v
  if (v === null || v === undefined) return ''
  try {
    return JSON.stringify(v)
  } catch {
    return String(v)
  }
}

export function extractMessagesFromState(state: unknown): ChatMessage[] {
  if (!state || typeof state !== 'object') return []

  // Common shapes:
  // - { messages: [{role:'assistant'|'user', content:'...'}] }
  // - { messages: [{type:'assistant', message:'...'}] }
  // - { history: [...] }
  const s = state as any
  const arr = Array.isArray(s.messages)
    ? s.messages
    : Array.isArray(s.history)
      ? s.history
      : Array.isArray(s.chat)
        ? s.chat
        : null

  if (!arr) return []

  const out: ChatMessage[] = []
  for (let i = 0; i < arr.length; i++) {
    const m = arr[i] as any
    const roleRaw = (m?.role ?? m?.sender ?? m?.type ?? m?.author) as unknown
    const contentRaw = (m?.content ?? m?.message ?? m?.text ?? m?.value) as unknown

    const role = roleRaw === 'user' ? 'user' : 'assistant'
    const content = safeString(contentRaw)
    if (!content) continue

    out.push({
      id: String(m?.id ?? `${i}`),
      role,
      content,
    })
  }

  return out
}

export function extractFinalParams(
  state: unknown,
  finalParamsFromResponse?: Record<string, unknown> | null,
): Record<string, unknown> | null {
  if (finalParamsFromResponse && typeof finalParamsFromResponse === 'object') return finalParamsFromResponse
  if (!state || typeof state !== 'object') return null

  const s = state as any
  const candidates = [s.final_params, s.finalParams, s.params, s.parameters, s.configuration, s.collected_params]

  for (const c of candidates) {
    if (c && typeof c === 'object' && !Array.isArray(c)) {
      return c as Record<string, unknown>
    }
  }

  return null
}

export function extractCollectedParams(state: unknown): Record<string, unknown> | null {
  if (!state || typeof state !== 'object') return null
  const s = state as any

  const candidates = [s.collected_params, s.collectedParams, s.params, s.parameters, s.configuration]
  for (const c of candidates) {
    if (c && typeof c === 'object' && !Array.isArray(c)) {
      return c as Record<string, unknown>
    }
  }
  return null
}

export function extractParamInfo(schema: unknown): Record<string, any> | null {
  if (!schema || typeof schema !== 'object') return null
  const s = schema as any

  // Expected (from existing app): { DEFAULT_VALUES: {...}, PARAM_INFO: {...} }
  if (s.PARAM_INFO && typeof s.PARAM_INFO === 'object') return s.PARAM_INFO

  // API shape (snake_case)
  if (s.param_info && typeof s.param_info === 'object') return s.param_info

  // Alternative shapes:
  if (s.parameters && typeof s.parameters === 'object') return s.parameters
  if (s.schema && typeof s.schema === 'object') return s.schema

  // If the schema itself is a mapping.
  if (!Array.isArray(s) && Object.keys(s).length) return s

  return null
}

export function computeProgress(args: {
  paramInfo: Record<string, any> | null
  collected: Record<string, unknown> | null
}): { done: number; total: number } {
  const { paramInfo, collected } = args
  const total = paramInfo ? Object.keys(paramInfo).length : collected ? Object.keys(collected).length : 0

  if (!collected) return { done: 0, total }

  if (!paramInfo) return { done: Object.keys(collected).length, total }

  const keys = new Set(Object.keys(paramInfo))
  let done = 0
  for (const k of Object.keys(collected)) {
    if (keys.has(k) && collected[k] !== undefined && collected[k] !== null && String(collected[k]) !== '') {
      done += 1
    }
  }

  return { done, total }
}
