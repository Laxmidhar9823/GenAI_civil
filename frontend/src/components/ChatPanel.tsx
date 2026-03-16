import { useEffect, useRef, useState } from 'react'
import type { ChatMessage } from '../lib/types'
import ChatMessageView from './ChatMessage'

export default function ChatPanel(props: {
  messages: ChatMessage[]
  busy: boolean
  onSend: (text: string) => void
}) {
  const { messages, busy, onSend } = props
  const headerDotClass = busy ? 'dot check' : messages.length > 0 ? 'dot ok' : 'dot'
  const [text, setText] = useState('')
  const scrollerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const el = scrollerRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages.length])

  return (
    <div className="card chat-card" aria-label="Chat">
      <div className="card-header">
        <div className="chat-title">
          <div className={headerDotClass} aria-hidden="true" />
          <h2>Guided assistant</h2>
        </div>
        <div className="sub sub-sm">
          Press <span className="kbd">Enter</span> to send
        </div>
      </div>

      <div className="chat" ref={scrollerRef} role="log" aria-live="polite" aria-relevant="additions">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-mark" aria-hidden="true" />
            <p>
              Welcome to the Pavement Configurator.
              <br />
              Share your first requirement to begin the guided flow.
            </p>
          </div>
        ) : (
          messages.map((m) => <ChatMessageView key={m.id} msg={m} />)
        )}
      </div>

      <div className="composer">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            const trimmed = text.trim()
            if (!trimmed || busy) return
            setText('')
            onSend(trimmed)
          }}
          className="composer-form"
        >
          <label className="sr-only" htmlFor="userInput">
            Your message
          </label>
          <input
            id="userInput"
            className="input composer-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={busy ? 'Thinking...' : 'Type your answer...'}
            disabled={busy}
            autoComplete="off"
          />
          <button className="btn primary" type="submit" disabled={busy || !text.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  )
}
