import { Link } from 'react-router-dom'
import MotionSection from '../components/MotionSection'

const highlights = [
  {
    title: 'Deterministic Workflow',
    description: 'The backend manages state and next-step prompts ensuring every configuration is auditable and precise.',
  },
  {
    title: 'Private & Local',
    description: 'Run the entire stack on your machine. No data leaves your network.',
  },
  {
    title: 'Live Tracking',
    description: 'Watch your configuration evolve in real-time and export production-ready parameters instantly.',
  },
]

export default function LandingPage() {
  return (
    <div className="landing-page">
      <div className="page-stack">
        <MotionSection className="hero-section">
          <p className="eyebrow">Pavement AI Assistant</p>
          <h1 className="hero-title">Rigid Pavement Configuration.<br/>Simplified.</h1>
          <p className="hero-desc">
            Experience a new standard in civil engineering workflows. 
            Build complex configurations with a clean, intelligent assistant that guides you every step of the way.
          </p>
          <div className="hero-actions">
            <Link to="/app" className="btn primary">
              Launch Assistant
            </Link>
            <Link to="/docs" className="btn">
              View Documentation
            </Link>
          </div>
        </MotionSection>

        <MotionSection className="feature-section">
          {highlights.map((item) => (
            <article key={item.title} className="feature-card">
              <h3>{item.title}</h3>
              <p>{item.description}</p>
            </article>
          ))}
        </MotionSection>

        <MotionSection className="feature-card content-card">
          <div style={{ maxWidth: '100%', textAlign: 'center' }}>
            <h3>Get started in seconds</h3>
            <p style={{ maxWidth: '600px', margin: '0 auto 24px' }}>
              Setup is minimal. Just run the local servers and you're ready to go.
            </p>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
              <span className="kbd">uvicorn backend.main:app</span>
              <span className="kbd">ollama serve</span>
            </div>
          </div>
        </MotionSection>
      </div>
    </div>
  )
}
