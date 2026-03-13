# Pavement Configuration Assistant — Frontend

This is a React + Vite + TypeScript frontend that talks to the FastAPI backend.

## Prerequisites

- Node.js 18+ recommended
- Backend running (default expected at `http://localhost:8000`)

## Configure API base URL

Vite environment variables must be prefixed with `VITE_`.

Create `frontend/.env` (optional):

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Run (dev)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Routes

- `/` — Landing page
- `/app` — Assistant workspace (chat/reset integration)
- `/docs` — Local setup, Ollama setup, troubleshooting
- `/examples` — Example prompts with quick-start links into `/app`

## Build (production)

```bash
cd frontend
npm run build
npm run preview
```
