import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { API_BASE_URL } from '../lib/api'
import type { ChatMessage as ChatMessageT } from '../lib/types'
import { transformAssistantDisplayText } from '../lib/displayLabels'

const UserIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
    <circle cx="12" cy="7" r="4"></circle>
  </svg>
)

const BotIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Z"/>
    <path d="m2 6 6 6"/>
    <path d="m22 6-6 6"/>
    <path d="M12 10a12.1 12.1 0 0 0-6 4v5a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-5a12.1 12.1 0 0 0-6-4Z"/>
    <path d="M8 20a4 4 0 0 1 8 0"/>
  </svg>
)

function resolveMarkdownImageSrc(src?: string): string | undefined {
  if (!src) return src
  if (/^https?:\/\//i.test(src) || src.startsWith('data:')) return src
  if (src.startsWith('/')) return `${API_BASE_URL}${src}`
  return `${API_BASE_URL}/${src}`
}

export default function ChatMessage({ msg }: { msg: ChatMessageT }) {
  const isUser = msg.role === 'user'
  const displayContent = isUser ? msg.content : transformAssistantDisplayText(msg.content)

  return (
    <article className={`message ${isUser ? 'user' : 'ai'}`}>
      <div className="message-avatar" aria-hidden="true">
        {isUser ? <UserIcon /> : <BotIcon />}
      </div>

      <div className="message-bubble">
        {isUser ? (
          <div className="user-content">{msg.content}</div>
        ) : (
          <div className="markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                img: ({ src, alt }) => (
                  <img src={resolveMarkdownImageSrc(src)} alt={alt || 'plot'} loading="lazy" />
                ),
              }}
            >
              {displayContent}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </article>
  )
}
