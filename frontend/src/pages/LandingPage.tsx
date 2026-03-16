import { Link } from 'react-router-dom'
import MotionSection from '../components/MotionSection'

const highlights = [
  {
    title: 'Structured chat workflow',
    description: 'The backend manages state and next-step prompts, so configuration stays deterministic and easy to audit.',
  },
  {
    title: 'Local-first architecture',
    description: 'Run backend and Ollama on your machine for private development workflows and repeatable results.',
  },
  {
    title: 'Production-ready output',
    description: 'Track completion progress live and export final parameters when the session reaches a complete state.',
  },
]

const quickFacts = [
  'Designed for engineering-focused workflows',
  'Responsive workspace optimized for long sessions',
  'Markdown support for rich assistant responses',
]

export default function LandingPage() {
  return (
    <div className="page-stack">
      <MotionSection className="hero-card">
        <p className="eyebrow">Pavement AI Assistant</p>
        <h1>Confident rigid pavement configuration, guided one prompt at a time.</h1>
        <p className="hero-copy">
          Build configurations with a clean local workspace that keeps technical context readable. Start with docs,
          verify services, then run your conversation flow in the assistant.
        </p>
        <div className="hero-actions">
          <Link to="/app" className="btn primary">
            Open Assistant
          </Link>
          <Link to="/docs" className="btn">
            Setup Guide
          </Link>
        </div>
      </MotionSection>

      <MotionSection className="hero-meta-grid">
        {quickFacts.map((fact) => (
          <article key={fact} className="feature-card interactive-lift">
            <h3>Workspace detail</h3>
            <p>{fact}</p>
          </article>
        ))}
      </MotionSection>

      <MotionSection className="content-card">
        <h2>Get started in 3 quick steps</h2>
        <ol>
          <li>
            Read <Link to="/docs">Docs</Link> and run install commands for backend and frontend.
          </li>
          <li>
            Start services: <span className="kbd">uvicorn backend.main:app --reload --port 8000</span> and{' '}
            <span className="kbd">ollama serve</span>.
          </li>
          <li>
            Open <Link to="/app">Assistant</Link>, confirm Ollama status, and begin your configuration chat.
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
