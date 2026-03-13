import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import ChatPanel from '../components/ChatPanel'
import OllamaSettings from '../components/OllamaSettings'
import ParamsPanel from '../components/ParamsPanel'
import { API_BASE_URL, apiChat, apiGetSchema, apiOllamaStatus, apiReset } from '../lib/api'
import type { ChatMessage, OllamaCheckState } from '../lib/types'
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
  const [schema, setSchema] = useState<unknown>(null)
  const [state, setState] = useState<unknown>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [finalParamsFromResponse, setFinalParamsFromResponse] = useState<Record<string, unknown> | null>(null)
  const [busy, setBusy] = useState(false)
  const [schemaBusy, setSchemaBusy] = useState(false)
  const [ollamaUrl, setOllamaUrl] = useState(() => localStorage.getItem('ollamaUrl') || 'http://localhost:11434')
  const [model, setModel] = useState(() => localStorage.getItem('ollamaModel') || 'gemma3:12b')
  const [ollamaStatus, setOllamaStatus] = useState<OllamaCheckState>({ kind: 'unknown' })
  const statusTimer = useRef<number | null>(null)

  const paramInfo = useMemo(() => extractParamInfo(schema), [schema])
  const collected = useMemo(() => extractCollectedParams(state), [state])
  const finalParams = useMemo(() => extractFinalParams(state, finalParamsFromResponse), [state, finalParamsFromResponse])
  const progress = useMemo(() => computeProgress({ paramInfo, collected }), [paramInfo, collected])

  const schemaDot = schemaBusy ? 'check' : schema ? 'ok' : 'err'
  const schemaLabel = schemaBusy ? 'Loading schema…' : schema ? 'Schema loaded' : 'Schema unavailable'

  const applyStateToMessages = useCallback((nextState: unknown) => {
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
    } finally {
      setSchemaBusy(false)
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
              'Welcome! The backend did not return a message history in `/reset`. You can still chat, but consider returning `messages` in state.',
          },
        ])
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [applyStateToMessages])

  const checkOllama = useCallback(async () => {
    setOllamaStatus({ kind: 'checking' })
    try {
      const res = await apiOllamaStatus({ ollama_url: ollamaUrl, model })
      if (!res.connected) {
        setOllamaStatus({
          kind: 'error',
          detail: `Cannot reach Ollama at ${ollamaUrl}. Start Ollama with "ollama serve" and try again.`,
        })
        return
      }

      if (!res.model_available) {
        const available = res.available_models?.length ? ` Available: ${res.available_models.slice(0, 5).join(', ')}.` : ''
        setOllamaStatus({
          kind: 'needs_model',
          model,
          availableModels: res.available_models ?? [],
          detail: `Ollama is running, but model "${model}" is not installed. Run "ollama pull ${model}" and retry.${available}`,
        })
        return
      }

      setOllamaStatus({ kind: 'ok', detail: `Ollama is running and model "${model}" is available.` })
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
        setFinalParamsFromResponse((resp.final_params as Record<string, unknown> | undefined) ?? null)

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
    void loadSchema()
    void resetConversation()
  }, [loadSchema, resetConversation])

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
          <h1>Assistant Workspace</h1>
          <p className="subtle-copy">Backend-driven guided flow with local model settings and exportable outputs.</p>
        </div>
        <button className="btn danger" type="button" onClick={() => void resetConversation()} disabled={busy}>
          Reset Session
        </button>
      </MotionSection>

      {error ? (
        <div className="banner" role="alert" aria-live="assertive">
          <strong>Error:</strong> {error}
          <ul style={{ margin: '8px 0 0', paddingLeft: 18 }}>
            <li>
              Start backend: <span className="kbd">uvicorn backend.main:app --reload --port 8000</span>
            </li>
            <li>
              Confirm frontend API base URL env: <span className="kbd">VITE_API_BASE_URL</span>
            </li>
            <li>
              Verify backend health:{' '}
              <span className="kbd">curl {API_BASE_URL}/health</span>
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

          <div className="card" aria-label="Notes">
            <div className="card-header">
              <h2>Notes</h2>
              <span className="pill">
                <span className={`dot ${schemaDot}`} aria-hidden="true" />
                <span>{schemaLabel}</span>
              </span>
            </div>
            <div className="card-body">
              <div className="sub" style={{ lineHeight: 1.6 }}>
                <p style={{ marginTop: 0 }}>
                  This UI sends only your free-text responses. The backend controls the guided flow and returns the next
                  question.
                </p>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  <li>
                    API base is configured via <span className="kbd">VITE_API_BASE_URL</span>.
                  </li>
                  <li>Assistant messages support Markdown rendering.</li>
                  <li>
                    When the backend returns <span className="kbd">final_params</span>, downloads appear.
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </MotionSection>
      </div>
    </div>
  )
}
