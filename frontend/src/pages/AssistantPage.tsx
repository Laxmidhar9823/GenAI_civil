import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import ChatPanel from '../components/ChatPanel'
import OllamaSettings from '../components/OllamaSettings'
import ParamsPanel from '../components/ParamsPanel'
import { API_BASE_URL, apiChat, apiGetSchema, apiHealth, apiOllamaStatus, apiReset } from '../lib/api'
import type { ChatMessage, ConversationState, OllamaCheckState, SchemaResponse } from '../lib/types'
import { computeProgress, extractCollectedParams, extractFinalParams, extractMessagesFromState, extractParamInfo } from '../lib/parsers'
import MotionSection from '../components/MotionSection'

function newId() {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`
}

export default function AssistantPage() {
  const [searchParams] = useSearchParams()
  const quickPrompt = searchParams.get('prompt')?.trim() || ''
  const quickStartSent = useRef(false)
  const [error, setError] = useState<string | null>(null)
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [state, setState] = useState<ConversationState | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [finalParamsFromResponse, setFinalParamsFromResponse] = useState<Record<string, number> | null>(null)
  const [busy, setBusy] = useState(false)
  const [schemaBusy, setSchemaBusy] = useState(false)
  const [healthBusy, setHealthBusy] = useState(false)
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null)
  const [ollamaUrl, setOllamaUrl] = useState(() => localStorage.getItem('ollamaUrl') || 'http://localhost:11434')
  const [model, setModel] = useState(() => localStorage.getItem('ollamaModel') || 'gemma3:12b')
  const [ollamaStatus, setOllamaStatus] = useState<OllamaCheckState>({ kind: 'unknown' })
  const statusTimer = useRef<number | null>(null)

  const paramInfo = useMemo(() => extractParamInfo(schema), [schema])
  const collected = useMemo(() => extractCollectedParams(state), [state])
  const finalParams = useMemo(() => extractFinalParams(state, finalParamsFromResponse), [state, finalParamsFromResponse])
  const progress = useMemo(() => computeProgress({ paramInfo, collected }), [paramInfo, collected])

  const schemaDot = schemaBusy ? 'check' : schema ? 'ok' : 'err'
  const schemaLabel = schemaBusy ? 'Loading schema...' : schema ? 'Schema loaded' : 'Schema unavailable'
  const backendDot = healthBusy ? 'check' : backendHealthy ? 'ok' : backendHealthy === false ? 'err' : 'dot'
  const backendLabel = healthBusy ? 'Checking backend...' : backendHealthy ? 'Backend connected' : backendHealthy === false ? 'Backend offline' : 'Backend unknown'

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
              'Welcome. The backend did not return message history in /reset. You can continue chatting, but returning messages in state is recommended.',
          },
        ])
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [applyStateToMessages])

  const reconnectBackend = useCallback(async () => {
    setError(null)
    const ok = await checkBackendHealth()
    if (!ok) return
    await Promise.all([loadSchema(), resetConversation()])
  }, [checkBackendHealth, loadSchema, resetConversation])

  const checkOllama = useCallback(async () => {
    setOllamaStatus({ kind: 'checking' })
    try {
      const res = await apiOllamaStatus({ ollama_url: ollamaUrl, model })
      const normalizedModel = model.trim()
      const isCloudTag = normalizedModel.toLowerCase().endsWith(':cloud')

      if (!res.connected) {
        setOllamaStatus({
          kind: 'error',
          detail: res.detail || `Cannot reach Ollama at ${ollamaUrl}. Start Ollama with "ollama serve" and try again.`,
        })
        return
      }

      if (!res.model_available) {
        const available = res.available_models?.length ? ` Available: ${res.available_models.slice(0, 5).join(', ')}.` : ''
        const suggestions = res.model_suggestions?.length ? ` Suggestions: ${res.model_suggestions.join(', ')}.` : ''
        const installHint = isCloudTag
          ? `Confirm the exact cloud tag and spelling, then run "ollama list".`
          : `Run "ollama pull ${normalizedModel}" and retry.`
        setOllamaStatus({
          kind: 'needs_model',
          model: normalizedModel,
          availableModels: res.available_models ?? [],
          suggestedModels: res.model_suggestions ?? [],
          detail:
            res.detail ||
            `Ollama is running, but model "${normalizedModel}" is not installed. ${installHint}${available}${suggestions}`,
        })
        return
      }

      const matched = res.matched_model || normalizedModel
      setOllamaStatus({ kind: 'ok', detail: `Ollama is running and model "${matched}" is available.` })
    } catch (e) {
      setOllamaStatus({ kind: 'error', detail: (e as Error).message })
    }
  }, [ollamaUrl, model])

  const send = useCallback(
    async (text: string) => {
      setError(null)
      setBusy(true)
      setMessages((prev) => [...prev, { id: newId(), role: 'user', content: text }])

      try {
        if (!state) throw new Error('No conversation state. Click Reset to start.')

        const resp = await apiChat({
          user_input: text,
          state,
          llm_config: {
            ollama_url: ollamaUrl,
            model,
          },
        })

        setState(resp.state)
        setFinalParamsFromResponse(resp.final_params ?? null)

        const used = applyStateToMessages(resp.state)
        if (!used) {
          const assistantText = resp.assistant_message || '(No assistant_message returned)'
          setMessages((prev) => [...prev, { id: newId(), role: 'assistant', content: assistantText }])
        }
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setBusy(false)
      }
    },
    [applyStateToMessages, model, ollamaUrl, state],
  )

  useEffect(() => {
    void reconnectBackend()
  }, [reconnectBackend])

  useEffect(() => {
    localStorage.setItem('ollamaUrl', ollamaUrl)
  }, [ollamaUrl])

  useEffect(() => {
    localStorage.setItem('ollamaModel', model)
  }, [model])

  useEffect(() => {
    if (statusTimer.current) window.clearTimeout(statusTimer.current)
    statusTimer.current = window.setTimeout(() => {
      void checkOllama()
    }, 600)

    return () => {
      if (statusTimer.current) window.clearTimeout(statusTimer.current)
    }
  }, [ollamaUrl, model, checkOllama])

  useEffect(() => {
    if (!quickPrompt || quickStartSent.current || !state || busy) return
    quickStartSent.current = true
    void send(quickPrompt)
  }, [quickPrompt, state, busy, send])

  return (
    <div className="page-stack">
      <MotionSection className="content-card app-header-card">
        <div>
          <p className="eyebrow">Assistant workspace</p>
          <h1>Run guided pavement configuration sessions</h1>
          <p className="subtle-copy">
            The backend controls question sequencing and parameter state while this UI keeps progress, model status, and
            exports in one focused layout.
          </p>
        </div>
        <div className="app-header-actions">
          <span className="pill pill-soft" aria-label="Backend status">
            <span className={`dot ${backendDot}`} aria-hidden="true" />
            <span>{backendLabel}</span>
          </span>
          <span className="pill pill-soft" aria-label="Schema status">
            <span className={`dot ${schemaDot}`} aria-hidden="true" />
            <span>{schemaLabel}</span>
          </span>
          <button className="btn" type="button" onClick={() => void reconnectBackend()} disabled={busy || schemaBusy || healthBusy}>
            Reconnect
          </button>
          <button className="btn danger" type="button" onClick={() => void resetConversation()} disabled={busy}>
            Reset session
          </button>
        </div>
      </MotionSection>

      {error ? (
        <div className="banner" role="alert" aria-live="assertive">
          <strong>Error:</strong> {error}
          <ul>
            <li>
              Start backend: <span className="kbd">uvicorn backend.main:app --reload --port 8000</span>
            </li>
            <li>
              Confirm frontend API base URL env: <span className="kbd">VITE_API_BASE_URL</span>
            </li>
            <li>
              Verify backend health: <span className="kbd">curl {API_BASE_URL}/health</span>
            </li>
            <li>
              Try <span className="kbd">Reconnect</span> after backend starts.
            </li>
            <li>
              Need setup help? Open <Link to="/docs">Docs</Link>.
            </li>
          </ul>
        </div>
      ) : null}

      <div className="assistant-grid">
        <MotionSection>
          <ChatPanel messages={messages} busy={busy} onSend={(t) => void send(t)} />
        </MotionSection>

        <MotionSection className="assistant-side">
          <OllamaSettings
            ollamaUrl={ollamaUrl}
            setOllamaUrl={setOllamaUrl}
            model={model}
            setModel={setModel}
            status={ollamaStatus}
            onCheckNow={() => void checkOllama()}
            apiBaseUrl={API_BASE_URL}
          />

          <ParamsPanel progress={progress} paramInfo={paramInfo} collected={collected} finalParams={finalParams} />

          <div className="card" aria-label="Workspace notes">
            <div className="card-header">
              <h2>Notes</h2>
              <span className="pill">
                <span className={`dot ${schemaDot}`} aria-hidden="true" />
                <span>{schemaLabel}</span>
              </span>
            </div>
            <div className="card-body">
              <p className="sub">
                This interface sends free-text responses only. The backend determines the next question and returns state
                updates after each message.
              </p>
              <ul className="notes-list">
                <li>
                  API base URL is configured via <span className="kbd">VITE_API_BASE_URL</span>.
                </li>
                <li>Assistant responses support Markdown rendering.</li>
                <li>
                  When backend returns <span className="kbd">final_params</span>, download actions appear in
                  Configuration Status.
                </li>
              </ul>
            </div>
          </div>
        </MotionSection>
      </div>
    </div>
  )
}
