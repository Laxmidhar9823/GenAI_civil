export type ChatRole = 'user' | 'assistant' | 'system'

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
