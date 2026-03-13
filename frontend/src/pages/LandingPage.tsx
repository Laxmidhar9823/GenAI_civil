import { Link } from 'react-router-dom'
import MotionSection from '../components/MotionSection'

const highlights = [
  {
    title: 'Structured intelligence',
    description: 'Conversational parameter capture with backend-driven logic and transparent progress tracking.',
  },
  {
    title: 'Production-minded flow',
    description: 'Built for local engineering workflows with deterministic reset/chat behavior and export-ready output.',
  },
  {
    title: 'Refined interaction model',
    description: 'Subtle motion, crisp hierarchy, and responsive layout tuned for focus-heavy technical sessions.',
  },
]

export default function LandingPage() {
  return (
    <div className="page-stack">
      <MotionSection className="hero-card">
        <p className="eyebrow">Pavement AI Assistant</p>
        <h1>Premium assistant experience for rigid pavement configuration.</h1>
        <p className="hero-copy">
          A light-first workspace for local guided setup. This app runs on your machine and needs both the backend API and
          Ollama running before you start chatting.
        </p>
        <div className="hero-actions">
          <Link to="/app" className="btn primary">
            Launch Assistant
          </Link>
          <Link to="/examples" className="btn">
            Explore Prompts
          </Link>
        </div>
      </MotionSection>

      <MotionSection className="content-card">
        <h2>Getting started (3 steps)</h2>
        <ol>
          <li>
            Read the <Link to="/docs">Docs</Link> for install and local setup commands.
          </li>
          <li>
            Start the backend and Ollama (<span className="kbd">uvicorn backend.main:app --reload --port 8000</span> +{' '}
            <span className="kbd">ollama serve</span>).
          </li>
          <li>
            Open the <Link to="/app">Assistant</Link>, check Ollama status, and begin your prompt flow.
          </li>
        </ol>
      </MotionSection>

      <MotionSection className="feature-grid">
        {highlights.map((item) => (
          <article key={item.title} className="feature-card interactive-lift">
            <h3>{item.title}</h3>
            <p>{item.description}</p>
          </article>
        ))}
      </MotionSection>
    </div>
  )
}
