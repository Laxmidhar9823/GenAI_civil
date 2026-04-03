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

function normalizeMeshType(v: string): 'coarse' | 'medium' | 'fine' {
  const lowered = (v || '').trim().toLowerCase()
  if (lowered === 'coarse' || lowered === 'medium' || lowered === 'fine') return lowered
  return 'coarse'
}

function linspaceNodes(spanMm: number, elements: number): number[] {
  const span = Number(spanMm)
  const elems = Math.max(0, Math.floor(Number(elements)))
  if (!Number.isFinite(span) || span < 0) return []
  if (elems <= 0) return [0, span]

  const values: number[] = []
  for (let i = 0; i <= elems; i++) {
    const raw = (span * i) / elems
    values.push(Math.round(raw * 1000) / 1000)
  }
  values[0] = 0
  values[values.length - 1] = span
  return values
}

function inferNodesFromMesh(meshType: string, a: number, b: number) {
  const level = normalizeMeshType(meshType)
  const elementsByLevel: Record<'coarse' | 'medium' | 'fine', number> = {
    coarse: 10,
    medium: 15,
    fine: 30,
  }
  const elements = elementsByLevel[level]
  return {
    x: linspaceNodes(a, elements),
    y: linspaceNodes(b, elements),
  }
}

function inferSpanFromNodes(v: unknown): number | null {
  if (!Array.isArray(v) || v.length < 2) return null
  const last = v[v.length - 1]
  const n = typeof last === 'number' ? last : Number(last)
  return Number.isFinite(n) && n >= 0 ? n : null
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

  const applyParamEdit = useCallback(
    (key: string, rawValue: string) => {
      const k = key.trim()
      if (!k) return

      // Never allow manual editing of derived node arrays.
      if (k === 'x' || k === 'y') return

      // Any edit invalidates prior confirmation.
      setFinalConfirmed(false)

      // Build a consistent snapshot of params: (current state.params + this edit).
      const baseParams = (state?.params ?? {}) as Record<string, unknown>
      const nextParams = { ...baseParams } as Record<string, unknown>

      if (k === 'mesh_type' || k === 'load_location') {
        nextParams[k] = rawValue.trim().toLowerCase()
      } else {
        const n = Number(rawValue)
        if (!Number.isFinite(n)) return
        nextParams[k] = n
      }

      // Mesh dependency: x/y are derived from (a,b,mesh_type).
      let derivedNodes: { x: number[]; y: number[] } | null = null
      if (k === 'mesh_type' || k === 'a' || k === 'b') {
        const aRaw = Number(nextParams['a'])
        const bRaw = Number(nextParams['b'])
        const a = Number.isFinite(aRaw) && aRaw >= 0 ? aRaw : inferSpanFromNodes(nextParams['x'])
        const b = Number.isFinite(bRaw) && bRaw >= 0 ? bRaw : inferSpanFromNodes(nextParams['y'])
        const meshType = String(nextParams['mesh_type'] ?? '')
        if (a !== null && b !== null) {
          derivedNodes = inferNodesFromMesh(meshType, a, b)
          if (derivedNodes.x.length && derivedNodes.y.length) {
            nextParams['x'] = derivedNodes.x
            nextParams['y'] = derivedNodes.y
          }
        }
      }

      setState((prev) => {
        const base: ConversationState =
          prev ??
          ({
            messages: [],
            params: {},
            user_provided_keys: [],
            current_asking: null,
            mode: 'welcome',
            welcomed: false,
          } as ConversationState)

        const nextUserKeys = Array.isArray(base.user_provided_keys)
          ? Array.from(new Set([...base.user_provided_keys, k]))
          : [k]

        return {
          ...base,
          params: nextParams,
          user_provided_keys: nextUserKeys,
        }
      })

      // If export-ready final_params exists, patch its nodes too.
      if (derivedNodes) {
        setFinalParamsFromResponse((prevFinal) => {
          if (!prevFinal || typeof prevFinal !== 'object') return prevFinal
          const next = { ...prevFinal } as Record<string, unknown>
          const nodesObj =
            next.nodes && typeof next.nodes === 'object' && !Array.isArray(next.nodes)
              ? ({ ...(next.nodes as Record<string, unknown>) } as Record<string, unknown>)
              : ({} as Record<string, unknown>)
          nodesObj.x = derivedNodes.x
          nodesObj.y = derivedNodes.y
          next.nodes = nodesObj
          return next
        })
      }
    },
    [state],
  )

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
          onEditParam={(key, value) => {
            applyParamEdit(key, value)
          }}
       />
    </div>
  )
}
