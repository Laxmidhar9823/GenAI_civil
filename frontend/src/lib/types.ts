export type ChatRole = 'user' | 'assistant' | 'system'

export type Message = {
  role: ChatRole
  content: string
}

export type ConversationMode = 'welcome' | 'guided' | 'free' | 'complete'

export type ConversationState = {
  messages: Message[]
  params: Record<string, number>
  user_provided_keys: string[]
  current_asking: string | null
  mode: ConversationMode
  welcomed: boolean
}

export type ParamInfoEntry = {
  name?: string
  description?: string
  [key: string]: unknown
}

export type ParamInfoMap = Record<string, ParamInfoEntry>

export type SchemaResponse = {
  PARAM_INFO?: ParamInfoMap
  DEFAULT_VALUES?: Record<string, number>
  PARAM_ORDER?: string[]
  PARAM_CATEGORIES?: Record<string, unknown>
  param_info?: ParamInfoMap
  default_values?: Record<string, number>
  param_order?: string[]
  param_categories?: Record<string, unknown>
  [key: string]: unknown
}

export type ChatMessage = {
  id: string
  role: Exclude<ChatRole, 'system'>
  content: string
}

export type OllamaCheckState =
  | { kind: 'unknown' }
  | { kind: 'checking' }
  | { kind: 'ok'; detail?: string }
  | { kind: 'needs_model'; model: string; availableModels: string[]; detail?: string }
  | { kind: 'error'; detail?: string }

export type OllamaStatusResponse = {
  connected: boolean
  model_available: boolean
  available_models: string[]
}

export type ResetResponse = {
  assistant_message: string
  state: ConversationState
}

export type ChatResponse = {
  assistant_message?: string
  state: ConversationState
  final_params?: Record<string, number>
}

export type ErrorDetail = {
  code: string
  message: string
  details?: Record<string, unknown>
}

export type ErrorResponse = {
  error: ErrorDetail
}
