import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ChatPanel from '../components/ChatPanel'
import OllamaSettings from '../components/OllamaSettings'
import OnboardingTour, { shouldShowTour } from '../components/OnboardingTour'
import ParamsPanel from '../components/ParamsPanel'
import { API_BASE_URL, apiChat, apiGenerateAnalysis, apiGetSchema, apiHealth, apiOllamaStatus, apiReset } from '../lib/api'
import type { ChatMessage, ConversationState, GenerateAnalysisResponse, OllamaCheckState, SchemaResponse } from '../lib/types'
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
  const [analysisBusy, setAnalysisBusy] = useState(false)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<GenerateAnalysisResponse | null>(null)
  // When a configuration is complete, require the user to review/confirm before exporting.
  // This adds an edit/update step without changing backend business logic.
  const [finalConfirmed, setFinalConfirmed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [, setSchemaBusy] = useState(false)
  const [, setHealthBusy] = useState(false)
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null)
  const [ollamaUrl, setOllamaUrl] = useState(() => localStorage.getItem('ollamaUrl') || 'http://localhost:11434')
  const [model, setModel] = useState(() => localStorage.getItem('ollamaModel') || 'qwen3.5:cloud')
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('ollamaApiKey') || '')
  const [ollamaStatus, setOllamaStatus] = useState<OllamaCheckState>({ kind: 'unknown' })

  // Onboarding tour
  const [showTour, setShowTour] = useState(false)

  // Param change highlighting
  const [highlightedParams, setHighlightedParams] = useState<Set<string>>(new Set())
  const highlightTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  const isFirstCollected = useRef(true)
  const prevCollectedRef = useRef<Record<string, unknown>>({})

  const emptyConversationState: ConversationState = {
    messages: [],
    memory: '',
    params: {},
    user_provided_keys: [],
    current_asking: null,
    mode: 'welcome',
    welcomed: false,
    analysis_generated: false,
    analysis_vtk_file: null,
    analysis_summary_file: null,
    analysis_plot_files: [],
  }

  // Effects to persist settings
  useEffect(() => { localStorage.setItem('ollamaUrl', ollamaUrl) }, [ollamaUrl])
  useEffect(() => { localStorage.setItem('ollamaModel', model) }, [model])
  useEffect(() => { localStorage.setItem('ollamaApiKey', apiKey) }, [apiKey])

  const paramInfo = useMemo(() => extractParamInfo(schema), [schema])
  const collected = useMemo(() => extractCollectedParams(state), [state])
  const finalParams = useMemo(() => extractFinalParams(state, finalParamsFromResponse), [state, finalParamsFromResponse])
  const progress = useMemo(() => computeProgress({ paramInfo, collected }), [paramInfo, collected])

  // Show onboarding tour once the page is fully settled
  useEffect(() => {
    if (!busy && state && backendHealthy && shouldShowTour()) {
      const t = setTimeout(() => setShowTour(true), 600)
      return () => clearTimeout(t)
    }
  }, [busy, state, backendHealthy])

  // Track param changes and highlight affected rows
  useEffect(() => {
    if (isFirstCollected.current) {
      isFirstCollected.current = false
      prevCollectedRef.current = collected ? { ...collected } : {}
      return
    }

    const prev = prevCollectedRef.current
    const curr = collected ?? {}

    const changed = new Set<string>()
    for (const key of Object.keys(curr)) {
      if (JSON.stringify(curr[key]) !== JSON.stringify(prev[key])) {
        changed.add(key)
      }
    }

    prevCollectedRef.current = { ...curr }

    if (changed.size === 0) return

    setHighlightedParams(p => new Set([...p, ...changed]))

    changed.forEach(key => {
      const existing = highlightTimers.current.get(key)
      if (existing) clearTimeout(existing)
      const t = setTimeout(() => {
        setHighlightedParams(p => {
          const next = new Set(p)
          next.delete(key)
          return next
        })
        highlightTimers.current.delete(key)
      }, 2500)
      highlightTimers.current.set(key, t)
    })
  }, [collected])

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
    setAnalysisError(null)
    setAnalysisResult(null)
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
      const res = await apiOllamaStatus({ ollama_url: ollamaUrl, model, api_key: apiKey || undefined })

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
  }, [ollamaUrl, model, apiKey])


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
    setAnalysisError(null)
    setAnalysisResult(null)
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
          api_key: apiKey || undefined,
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
      setAnalysisError(null)
      setAnalysisResult(null)

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
            memory: '',
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

      // Patch the cached export snapshot so it never goes stale after a panel edit.
      setFinalParamsFromResponse((prevFinal) => {
        if (!prevFinal || typeof prevFinal !== 'object') return prevFinal
        const next = { ...prevFinal } as Record<string, unknown>

        // Nodes (derived from mesh_type / a / b)
        if (derivedNodes) {
          const nodesObj =
            next.nodes && typeof next.nodes === 'object' && !Array.isArray(next.nodes)
              ? ({ ...(next.nodes as Record<string, unknown>) } as Record<string, unknown>)
              : ({} as Record<string, unknown>)
          nodesObj.x = derivedNodes.x
          nodesObj.y = derivedNodes.y
          next.nodes = nodesObj
        }

        // Slab properties
        if (k === 'Emod' || k === 'nu' || k === 't') {
          const slabObj =
            next.slab && typeof next.slab === 'object' && !Array.isArray(next.slab)
              ? ({ ...(next.slab as Record<string, unknown>) } as Record<string, unknown>)
              : ({} as Record<string, unknown>)
          slabObj[k] = nextParams[k]
          next.slab = slabObj
        }

        // Subgrade properties
        if (k === 'Kx' || k === 'Ky' || k === 'Kz') {
          const subgradeObj =
            next.subgrade && typeof next.subgrade === 'object' && !Array.isArray(next.subgrade)
              ? ({ ...(next.subgrade as Record<string, unknown>) } as Record<string, unknown>)
              : ({} as Record<string, unknown>)
          subgradeObj[k] = nextParams[k]
          next.subgrade = subgradeObj
        }

        // Load patch coords / pressure — only patch when there is exactly one load case
        // to avoid accidentally corrupting multi-case arrays that came from the backend.
        if (k === 'x1' || k === 'x2' || k === 'y1' || k === 'y2' || k === 'q') {
          const loadsObj =
            next.loads && typeof next.loads === 'object' && !Array.isArray(next.loads)
              ? ({ ...(next.loads as Record<string, unknown>) } as Record<string, unknown>)
              : ({} as Record<string, unknown>)
          const existingArr = loadsObj[k]
          if (!Array.isArray(existingArr) || existingArr.length <= 1) {
            loadsObj[k] = [nextParams[k]]
            next.loads = loadsObj
          }
        }

        return next
      })
    },
    [state],
  )

  const generateAnalysis = useCallback(async () => {
    if (!finalParams || busy || analysisBusy) return

    setAnalysisBusy(true)
    setAnalysisError(null)
    setAnalysisResult(null)

    try {
      const resp = await apiGenerateAnalysis({ final_params: finalParams })
      setAnalysisResult(resp)
      if (!resp.success) {
        setAnalysisError(resp.message || 'Analysis failed.')
        return
      }

      const resolvedPlotUrls = (resp.plot_files ?? []).map((plotPath) =>
        plotPath.startsWith('http') ? plotPath : `${API_BASE_URL}${plotPath.startsWith('/') ? plotPath : `/${plotPath}`}`,
      )

      const assistantMessage = 'Analysis complete. Ask me anything about your results.'
      const assistantStateMessage: ConversationState['messages'][number] = {
        role: 'assistant',
        content: assistantMessage,
      }

      setMessages((prev) => [...prev, { id: newId(), role: 'assistant', content: assistantMessage }])
      setState((prev) => {
        const base = prev ?? emptyConversationState
        const nextMessages = [...base.messages, assistantStateMessage]
        return {
          ...base,
          mode: base.mode === 'welcome' ? 'free' : base.mode,
          analysis_generated: true,
          analysis_vtk_file: resp.vtk_file ?? base.analysis_vtk_file ?? null,
          analysis_summary_file: resp.summary_file ?? base.analysis_summary_file ?? null,
          analysis_plot_files: resolvedPlotUrls,
          messages: nextMessages,
        }
      })
    } catch (e) {
      setAnalysisError((e as Error).message)
    } finally {
      setAnalysisBusy(false)
    }
  }, [analysisBusy, busy, finalParams])

  return (
    <div className="chat-layout">
       {/* Left Sidebar: Settings */}
       <div className="chat-sidebar">
          <OllamaSettings 
            ollamaUrl={ollamaUrl} 
            setOllamaUrl={setOllamaUrl}
            model={model}
            setModel={setModel}
            apiKey={apiKey}
            setApiKey={setApiKey}
            status={ollamaStatus}
            onCheckNow={checkOllama}
            apiBaseUrl="http://localhost:8000"
          />

          <div style={{ marginTop: 'auto' }}>
            <button
              className="btn"
              style={{ width: '100%' }}
              onClick={resetConversation}
              data-tour="reset-session"
            >
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
         analysisBusy={analysisBusy}
         analysisError={analysisError}
         analysisResult={analysisResult}
         onGenerateAnalysis={generateAnalysis}
          highlightedParams={highlightedParams}
          onEditParam={(key, value) => {
            applyParamEdit(key, value)
          }}
       />
       {showTour && <OnboardingTour onDone={() => setShowTour(false)} />}
    </div>
  )
}
