import type { ChatMessage, ConversationState, ParamInfoMap, SchemaResponse } from './types'

type ObjectLike = Record<string, unknown>
type StateLike = ConversationState | ObjectLike | null | undefined

function safeString(v: unknown): string {
  if (typeof v === 'string') return v
  if (v === null || v === undefined) return ''
  try {
    return JSON.stringify(v)
  } catch {
    return String(v)
  }
}

export function extractMessagesFromState(state: StateLike): ChatMessage[] {
  if (!state || typeof state !== 'object') return []

  const s = state as ObjectLike
  const arrRaw = s.messages ?? s.history ?? s.chat
  const arr = Array.isArray(arrRaw) ? arrRaw : null

  if (!arr) return []

  const out: ChatMessage[] = []
  for (let i = 0; i < arr.length; i++) {
    if (!arr[i] || typeof arr[i] !== 'object') continue
    const m = arr[i] as ObjectLike
    const roleRaw = m.role ?? m.sender ?? m.type ?? m.author
    const contentRaw = m.content ?? m.message ?? m.text ?? m.value

    const role = roleRaw === 'user' ? 'user' : 'assistant'
    const content = safeString(contentRaw)
    if (!content) continue

    out.push({
      id: String(m.id ?? `${i}`),
      role,
      content,
    })
  }

  return out
}

export function extractFinalParams(
  state: StateLike,
  finalParamsFromResponse?: Record<string, unknown> | null,
): Record<string, unknown> | null {
  if (finalParamsFromResponse && typeof finalParamsFromResponse === 'object') {
    return finalParamsFromResponse
  }
  if (!state || typeof state !== 'object') return null

  const s = state as ObjectLike
  const candidates = [s.final_params, s.finalParams, s.configuration]

  for (const c of candidates) {
    if (!c || typeof c !== 'object' || Array.isArray(c)) continue
    const obj = c as Record<string, unknown>
    const hasFinalShape =
      typeof obj.nodes === 'object' &&
      typeof obj.slab === 'object' &&
      typeof obj.subgrade === 'object' &&
      typeof obj.loads === 'object'
    if (hasFinalShape) {
      return obj
    }
  }

  return null
}

export function extractCollectedParams(state: StateLike): Record<string, unknown> | null {
  if (!state || typeof state !== 'object') return null
  const s = state as ObjectLike

  const candidates = [s.collected_params, s.collectedParams, s.params, s.parameters, s.configuration]
  for (const c of candidates) {
    if (c && typeof c === 'object' && !Array.isArray(c)) {
      return c as Record<string, unknown>
    }
  }
  return null
}

export function extractParamInfo(schema: SchemaResponse | ObjectLike | null | undefined): ParamInfoMap | null {
  if (!schema || typeof schema !== 'object') return null
  const s = schema as SchemaResponse & ObjectLike

  // Expected (from existing app): { DEFAULT_VALUES: {...}, PARAM_INFO: {...} }
  if (s.PARAM_INFO && typeof s.PARAM_INFO === 'object' && !Array.isArray(s.PARAM_INFO)) return s.PARAM_INFO

  // API shape (snake_case)
  if (s.param_info && typeof s.param_info === 'object' && !Array.isArray(s.param_info)) return s.param_info

  // Alternative shapes:
  if (s.parameters && typeof s.parameters === 'object' && !Array.isArray(s.parameters)) return s.parameters as ParamInfoMap
  if (s.schema && typeof s.schema === 'object' && !Array.isArray(s.schema)) return s.schema as ParamInfoMap

  // If the schema itself is a mapping.
  if (!Array.isArray(s) && Object.keys(s).length) return s as ParamInfoMap

  return null
}

export function computeProgress(args: {
  paramInfo: ParamInfoMap | null
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
