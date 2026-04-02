import { useCallback, useEffect, useMemo, useState } from 'react'
import ChatPanel from '../components/ChatPanel'
import OllamaSettings from '../components/OllamaSettings'
import ParamsPanel from '../components/ParamsPanel'
import { apiChat, apiGetSchema, apiHealth, apiOllamaStatus, apiReset } from '../lib/api'
import type { ChatMessage, ConversationState, OllamaCheckState, SchemaResponse } from '../lib/types'
import { computeProgress, extractCollectedParams, extractFinalParams, extractMessagesFromState, extractParamInfo } from '../lib/parsers'

function newId() {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`
}

export default function AssistantPage() {
  const [error, setError] = useState<string | null>(null)
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [state, setState] = useState<ConversationState | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [finalParamsFromResponse, setFinalParamsFromResponse] = useState<Record<string, unknown> | null>(null)
  // When a configuration is complete, require the user to review/confirm before exporting.
  // This adds an edit/update step without changing backend business logic.
  const [finalConfirmed, setFinalConfirmed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [, setSchemaBusy] = useState(false)
  const [, setHealthBusy] = useState(false)
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null)
  const [ollamaUrl, setOllamaUrl] = useState(() => localStorage.getItem('ollamaUrl') || 'http://localhost:11434')
  const [model, setModel] = useState(() => localStorage.getItem('ollamaModel') || 'qwen3.5:cloud')
  const [ollamaStatus, setOllamaStatus] = useState<OllamaCheckState>({ kind: 'unknown' })

  const emptyConversationState: ConversationState = {
    messages: [],
    params: {},
    user_provided_keys: [],
    current_asking: null,
    mode: 'welcome',
    welcomed: false,
  }

  // Effects to persist settings
  useEffect(() => { localStorage.setItem('ollamaUrl', ollamaUrl) }, [ollamaUrl])
  useEffect(() => { localStorage.setItem('ollamaModel', model) }, [model])

  const paramInfo = useMemo(() => extractParamInfo(schema), [schema])
  const collected = useMemo(() => extractCollectedParams(state), [state])
  const finalParams = useMemo(() => extractFinalParams(state, finalParamsFromResponse), [state, finalParamsFromResponse])
  const progress = useMemo(() => computeProgress({ paramInfo, collected }), [paramInfo, collected])

  const applyStateToMessages = useCallback((nextState: ConversationState) => {
    const fromState = extractMessagesFromState(nextState)
    if (fromState.length) {
      setMessages(fromState)
      return true
    }
    return false
  }, [])

  const loadSchema = useCallback(async () => {
    setSchemaBusy(true)
    try {
      const s = await apiGetSchema()
      setSchema(s)
    } catch (e) {
      setError((e as Error).message)
      setSchema(null)
    } finally {
      setSchemaBusy(false)
    }
  }, [])

  const checkBackendHealth = useCallback(async () => {
    setHealthBusy(true)
    try {
      await apiHealth()
      setBackendHealthy(true)
      return true
    } catch (e) {
      setBackendHealthy(false)
      setError((e as Error).message)
      return false
    } finally {
      setHealthBusy(false)
    }
  }, [])

  const resetConversation = useCallback(async () => {
    setError(null)
    setBusy(true)
    setFinalParamsFromResponse(null)
    setFinalConfirmed(false)

    try {
      const resp = await apiReset()
      setState(resp.state)
      const used = applyStateToMessages(resp.state)
      if (!used) {
        setMessages([
          {
            id: newId(),
            role: 'assistant',
            content:
              resp.assistant_message ||
              'Welcome. The backend did not return message history in /reset.',
          },
        ])
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [applyStateToMessages])

  // Initial load
  useEffect(() => {
    checkBackendHealth()
    loadSchema()
  }, [checkBackendHealth, loadSchema])

  const checkOllama = useCallback(async () => {
    setOllamaStatus({ kind: 'checking' })
    try {
      const res = await apiOllamaStatus({ ollama_url: ollamaUrl, model })

      if (res.error) {
        setOllamaStatus({
          kind: 'error',
          detail: res.detail || res.error,
        })
        return
      }
      
      if (!res.connected) {
        setOllamaStatus({
          kind: 'error',
          detail: res.detail || `Cannot reach Ollama at ${ollamaUrl}.`,
        })
        return
      }

      if (!res.model_available) {
        setOllamaStatus({
          kind: 'needs_model',
          model: model,
          availableModels: res.available_models ?? [],
          suggestedModels: res.model_suggestions ?? [],
          detail: res.detail ?? undefined,
        })
        return
      }

      const matched = res.matched_model || model
      setOllamaStatus({ kind: 'ok', detail: `Ready using ${matched}.` })
    } catch (e) {
      setOllamaStatus({ kind: 'error', detail: (e as Error).message })
    }
  }, [ollamaUrl, model])

  // Initial check of Ollama if backend is healthy
  useEffect(() => {
    if (backendHealthy) {
       checkOllama()
    }
  }, [backendHealthy, checkOllama])

  // Ensure chat always has a valid non-null state payload.
  useEffect(() => {
    if (backendHealthy && !state && !busy) {
      resetConversation()
    }
  }, [backendHealthy, state, busy, resetConversation])


  const handleSend = async (text: string) => {
    setBusy(true)
    setError(null)
    setFinalParamsFromResponse(null)
    // Any new message (including edits) invalidates prior confirmation.
    setFinalConfirmed(false)

    // Optimistic user message
    const userMsg: ChatMessage = { id: newId(), role: 'user', content: text }
    setMessages((prev) => [...prev, userMsg])

    try {
      const nextState = state ?? emptyConversationState
      const resp = await apiChat({
        user_input: text,
        state: nextState,
        llm_config: {
          ollama_url: ollamaUrl,
          model,
        },
      })

      if (resp.state) {
        setState(resp.state)
        // If the backend returns all messages, use them to sync state
        const syncedMessages = extractMessagesFromState(resp.state)
        if (syncedMessages.length > 0) {
           setMessages(syncedMessages)
        } else {
           // Fallback if backend doesn't return full history in state
           if (resp.assistant_message) {
             setMessages(prev => [...prev, { id: newId(), role: 'assistant', content: resp.assistant_message! }])
           }
        }
      }

      const latestFinalParams = resp.final_params ?? extractFinalParams(resp.state, null)
      setFinalParamsFromResponse(latestFinalParams ?? null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="chat-layout">
       {/* Left Sidebar: Settings */}
       <div className="chat-sidebar">
          <OllamaSettings 
            ollamaUrl={ollamaUrl} 
            setOllamaUrl={setOllamaUrl}
            model={model}
            setModel={setModel}
            status={ollamaStatus}
            onCheckNow={checkOllama}
            apiBaseUrl="http://localhost:8000"
          />

          <div style={{ marginTop: 'auto' }}>
            <button className="btn" style={{ width: '100%' }} onClick={resetConversation}>
               Reset Session
            </button>
            {error && (
              <div className="status-detail warning" style={{ marginTop: '12px' }}>
                Error: {error}
              </div>
            )}
          </div>
       </div>

       {/* Center: Chat */}
       <ChatPanel messages={messages} busy={busy} onSend={handleSend} />

       {/* Right Sidebar: Params & Progress */}
       <ParamsPanel 
          progress={progress}
          paramInfo={paramInfo}
          collected={collected}
          finalParams={finalParams}
          busy={busy}
          finalConfirmed={finalConfirmed}
          onConfirmFinal={() => setFinalConfirmed(true)}
          onEditParam={async (key, value) => {
            // Route edits through the existing /chat logic (LLM extraction + validation)
            // so we don't duplicate rules in the frontend.
            await handleSend(`Update ${key} to ${value}`)
          }}
       />
    </div>
  )
}
