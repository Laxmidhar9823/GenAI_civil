import MotionSection from '../components/MotionSection'

export default function DocsPage() {
  return (
    <div className="page-stack">
      <MotionSection className="content-card">
        <h1>Docs</h1>
        <p className="subtle-copy">Local-only setup, Ollama configuration, and practical troubleshooting.</p>
      </MotionSection>

      <MotionSection className="content-card">
        <h2>1) Install dependencies</h2>
        <ol>
          <li>
            Python backend deps: <code>pip install -r requirements.txt</code>
          </li>
          <li>
            Frontend deps: <code>cd frontend && npm install</code>
          </li>
        </ol>
      </MotionSection>

      <MotionSection className="content-card">
        <h2>2) Run local-only stack</h2>
        <ol>
          <li>Start backend: <code>uvicorn backend.main:app --reload --port 8000</code></li>
          <li>
            Verify backend is up: <code>curl http://localhost:8000/health</code>
          </li>
          <li>
            (Optional) set frontend API URL: <code>VITE_API_BASE_URL=http://localhost:8000</code> before running dev/build.
          </li>
          <li>Start frontend: <code>cd frontend && npm run dev</code></li>
          <li>Open <code>http://localhost:5173</code> and navigate to App.</li>
        </ol>
      </MotionSection>

      <MotionSection className="content-card">
        <h2>3) Configure Ollama</h2>
        <ol>
          <li>Install and run Ollama locally.</li>
          <li>
            Start daemon: <code>ollama serve</code>
          </li>
          <li>Pull your model, for example: <code>ollama pull gemma3:12b</code></li>
          <li>In App, set URL to <code>http://localhost:11434</code> and your model name.</li>
          <li>Use <strong>Check status</strong> to verify reachability through backend.</li>
        </ol>
      </MotionSection>

      <MotionSection className="content-card">
        <h2>4) Local-only + privacy notes</h2>
        <ul>
          <li>This project is designed for local development on your machine.</li>
          <li>Chat requests go to your configured local backend (<code>VITE_API_BASE_URL</code>).</li>
          <li>Model inference uses your local Ollama endpoint; no cloud LLM dependency is required.</li>
        </ul>
      </MotionSection>

      <MotionSection className="content-card">
        <h2>5) Troubleshooting</h2>
        <ul>
          <li>
            <strong>Backend unreachable:</strong> confirm API base URL and backend process on <code>:8000</code>.
          </li>
          <li>
            <strong>Ollama unavailable:</strong> check Ollama daemon, URL, model spelling, and local firewall.
          </li>
          <li>
            <strong>No response after send:</strong> reset session, then retry prompt to rebuild state.
          </li>
        </ul>
      </MotionSection>
    </div>
  )
}
