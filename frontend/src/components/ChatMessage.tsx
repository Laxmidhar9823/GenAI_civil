import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage as ChatMessageT } from '../lib/types'

export default function ChatMessage({ msg }: { msg: ChatMessageT }) {
  const isUser = msg.role === 'user'

  return (
    <article className={`message ${isUser ? 'user' : 'ai'}`}>
      <div className="message-avatar" aria-hidden="true">
        {isUser ? 'You' : 'AI'}
      </div>

      <div className="message-bubble">
        {isUser ? (
          <div>{msg.content}</div>
        ) : (
          <div className="markdown-content">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
          </div>
        )}
      </div>
    </article>
  )
}
