import { useEffect, useRef, useState } from 'react'
import type { ChatMessage } from '../lib/types'
import ChatMessageView from './ChatMessage'

export default function ChatPanel(props: {
  messages: ChatMessage[]
  busy: boolean
  onSend: (text: string) => void
}) {
  const { messages, busy, onSend } = props
  const [text, setText] = useState('')
  const scrollerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const el = scrollerRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages.length])

  return (
    <div className="chat-main" data-tour="chat-main">
      <div className="chat-messages" ref={scrollerRef} role="log" aria-live="polite" aria-relevant="additions">
        {messages.length === 0 ? (
          <div className="chat-empty-state">
            <div className="empty-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Z"/>
                <path d="m2 6 6 6"/>
                <path d="m22 6-6 6"/>
                <path d="M12 10a12.1 12.1 0 0 0-6 4v5a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-5a12.1 12.1 0 0 0-6-4Z"/>
                <path d="M8 20a4 4 0 0 1 8 0"/>
              </svg>
            </div>
            <h3>Pavement Assistant</h3>
            <p>Ready to configure your project.</p>
          </div>
        ) : (
          messages.map((m) => <ChatMessageView key={m.id} msg={m} />)
        )}
      </div>

      <div className="chat-input-area">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            const trimmed = text.trim()
            if (!trimmed || busy) return
            setText('')
            onSend(trimmed)
          }}
          className="input-group"
        >
          <label className="sr-only" htmlFor="userInput">
            Your message
          </label>
          <input
            id="userInput"
            className="chat-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={busy ? 'Processing...' : 'Ask about pavement configuration...'}
            disabled={busy}
            autoComplete="off"
          />
          <button className="send-btn" type="submit" disabled={busy || !text.trim()}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </form>
      </div>
    </div>
  )
}
