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
    <div className="chat-main">
      <div className="chat-messages" ref={scrollerRef} role="log" aria-live="polite" aria-relevant="additions">
        {messages.length === 0 ? (
          <div style={{ 
            height: '100%', 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center', 
            opacity: 0.5,
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>💬</div>
            <p>Ready to assist.</p>
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
