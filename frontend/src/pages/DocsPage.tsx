import MotionSection from '../components/MotionSection'

export default function DocsPage() {
  return (
    <div className="page-stack">
      <MotionSection className="content-card">
        <h1>Setup and operating guide</h1>
        <p className="subtle-copy">
          Follow the steps below to launch backend/frontend and validate Ollama model connectivity from the assistant
          workspace.
        </p>
      </MotionSection>

      <MotionSection className="docs-grid">
        <article className="doc-card interactive-lift">
          <h3>1) Install dependencies</h3>
          <ol>
            <li>
              Python dependencies: <span className="kbd">pip install -r requirements.txt</span>
            </li>
            <li>
              Frontend dependencies: <span className="kbd">cd frontend && npm install</span>
            </li>
          </ol>
        </article>

        <article className="doc-card interactive-lift">
          <h3>2) Start backend and frontend</h3>
          <ol>
            <li>
              Start backend: <span className="kbd">uvicorn backend.main:app --reload --port 8000</span>
            </li>
            <li>
              Health check: <span className="kbd">curl http://localhost:8000/health</span>
            </li>
            <li>
              Start frontend: <span className="kbd">cd frontend && npm run dev</span>
            </li>
          </ol>
        </article>

        <article className="doc-card interactive-lift">
          <h3>3) Configure Ollama</h3>
          <ol>
            <li>
              Run daemon: <span className="kbd">ollama serve</span>
            </li>
            <li>
              Use a model tag, for example <span className="kbd">kimi-k2.5:cloud</span> or{' '}
              <span className="kbd">qwen3.5:cloud</span>.
            </li>
            <li>
              Set URL <span className="kbd">http://localhost:11434</span> and model name in Assistant.
            </li>
          </ol>
        </article>

        <article className="doc-card interactive-lift">
          <h3>4) Troubleshooting</h3>
          <ul>
            <li>
              Backend unreachable: confirm <span className="kbd">VITE_API_BASE_URL</span> and backend process status.
            </li>
            <li>
              Ollama unavailable: verify URL, daemon, and model spelling.
            </li>
            <li>No assistant response: reset session and send the prompt again.</li>
          </ul>
        </article>
      </MotionSection>

      <MotionSection className="content-card">
        <h2>Execution notes</h2>
        <ul>
          <li>Chat requests go to your configured local backend endpoint.</li>
          <li>Model inference runs through your configured Ollama host.</li>
          <li>Cloud-tag model names (for example, <span className="kbd">kimi-k2.5:cloud</span>) are supported.</li>
        </ul>
      </MotionSection>
    </div>
  )
}
