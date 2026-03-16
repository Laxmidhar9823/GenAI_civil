import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage as ChatMessageT } from '../lib/types'

export default function ChatMessage({ msg }: { msg: ChatMessageT }) {
  const isUser = msg.role === 'user'

  return (
    <article className={`msg ${isUser ? 'user' : 'assistant'}`} aria-label={isUser ? 'User message' : 'Assistant message'}>
      <div className={`avatar ${isUser ? 'user' : 'assistant'}`} aria-hidden="true">
        {isUser ? 'U' : 'AI'}
      </div>

      <div className={`bubble ${isUser ? 'user' : 'assistant'}`}>
        {isUser ? <div className="user-text">{msg.content}</div> : <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>}
      </div>
    </article>
  )
}
