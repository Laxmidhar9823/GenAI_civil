import { Link } from 'react-router-dom'
import MotionSection from '../components/MotionSection'

const examples = [
  'Start a rigid pavement project for a heavy-duty industrial corridor with high axle loads.',
  'Configure a two-lane arterial with moderate truck traffic and warm climate assumptions.',
  'I need a quick baseline configuration for a municipal roadway with standard design constraints.',
  'Use conservative defaults and ask me one question at a time for missing inputs.',
]

export default function ExamplesPage() {
  return (
    <div className="page-stack">
      <MotionSection className="content-card">
        <h1>Example prompts</h1>
        <p className="subtle-copy">Jump directly into the assistant with a starter message.</p>
      </MotionSection>

      <MotionSection className="example-grid">
        {examples.map((prompt) => (
          <article key={prompt} className="example-card interactive-lift">
            <p>{prompt}</p>
            <Link className="btn" to={`/app?prompt=${encodeURIComponent(prompt)}`}>
              Open in App
            </Link>
          </article>
        ))}
      </MotionSection>
    </div>
  )
}
