<div align="center">

# PaveAI

Natural-language assistant for rigid pavement / concrete slab analysis.

Generate solver-ready JSON, run the FEM solver, and interrogate results (VTK) via scripts or chat.

[Quickstart](#quickstart) · [Architecture](#architecture) · [API](#api) · [Analysis--vtk](#analysis--vtk) · [Troubleshooting](#troubleshooting)

</div>

---

## What this repo contains

- **FastAPI backend** (stateless): `backend/main.py`
- **React + Vite frontend**: `frontend/`
- **Streamlit app** (legacy/standalone UI): `app.py`
- **Solver scripts** (run locally): `DOF5_07.09.2025.py` + `fem_local_Plate20_DOF5_07_09_2025.py`
- **VTK tooling** (no external VTK deps): `backend/scripts/` and `backend/vtk_stats.py`

## Key capabilities

- **Conversational parameter capture**: slab size, thickness, concrete properties, subgrade support, loads.
- **Implicit engineering shortcuts**:
  - `mesh_type` (`coarse`/`medium`/`fine`) → infers solver node grids (`nodes.x`, `nodes.y`).
  - `load_location` (`corner`/`edge`/`interior`) → infers a reasonable load patch (`x1/x2/y1/y2`).
- **Unit handling + sanity checks**: supports common metric/imperial inputs; flags physically implausible values.
- **End-to-end run**: `POST /analysis/generate` executes the solver and produces artifacts in `output_python_square/`.
- **VTK Q&A**: ask summary/statistics questions after analysis, or run the scripts directly.

---

## Quickstart

### Option A — Docker Compose (frontend + API)

```bash
docker compose up --build
```

Then open:

- Frontend: `http://localhost:5173`
- API: `http://localhost:8000` (OpenAPI docs: `http://localhost:8000/docs`)

**Important (Ollama + Docker):** the backend container cannot reach your host’s `http://localhost:11434`.

- On Docker Desktop (macOS/Windows): set `ollama_url` in the UI to `http://host.docker.internal:11434`.
- On Linux: use your host IP (often `http://172.17.0.1:11434` on the default bridge) or run the backend locally (Option B).

Stop:

```bash
docker compose down
```

### Option B — Local dev (recommended for analysis runs)

Backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### Option C — Streamlit UI (standalone)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Ollama setup

1) Install Ollama: https://ollama.com/

2) Start the daemon:

```bash
ollama serve
```

3) Pull a model (examples):

```bash
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
```

The backend/frontend default model string is `qwen3.5:cloud`, but the app will fall back to an available local model when needed.
If you use a cloud-tag model, supply an API key in the UI (sent as `llm_config.api_key`).

---

## Architecture

```mermaid
flowchart LR
  U[User] -->|browser| W[React Frontend\n/frontend]
  W -->|HTTP JSON| API[FastAPI Backend\n/backend/main.py]
  API -->|chat + extraction| OLL[(Ollama\nhttp://localhost:11434)]
  API -->|runs| SOLVER[Python Solver\nDOF5_07.09.2025.py]
  SOLVER --> OUT[output_python_square/\nresults.vtk + summaries]
  API -->|serves| ART[/artifacts/*]
  W -->|shows links| ART
```

---

## API

The backend is intentionally **stateless**: the client stores conversation state and sends it each turn.

### Endpoints

- `GET /health`
- `GET /config/schema` — parameters, defaults, ordering, and final output defaults
- `GET /ollama/status` — connectivity + available model tags
- `POST /reset` — start a new session (returns a fresh `ConversationState`)
- `POST /chat` — one assistant turn
- `POST /analysis/generate` — run solver for a completed config

### Stateless contract (high level)

- `POST /reset` → `{ assistant_message, state }`
- `POST /chat` → request `{ user_input, state, llm_config }`, response `{ assistant_message, state, final_params? }`
  - `final_params` is returned when the assistant reaches **complete** mode.
  - `final_params` format:
    - `nodes: { x: number[], y: number[] }`
    - `slab: { Emod: number, nu: number, t: number }`
    - `subgrade: { Kx: number, Ky: number, Kz: number }`
    - `loads: { x1: number[], x2: number[], y1: number[], y2: number[], q: number[] }`

Tip: treat `GET /config/schema` as the source of truth (it returns both legacy `UPPERCASE` keys and `snake_case` keys).

---

## Analysis + VTK

### Run analysis from the app

Once you have `final_params`, call:

```bash
curl -sS -X POST http://localhost:8000/analysis/generate \
  -H 'Content-Type: application/json' \
  -d '{"final_params": {"nodes": {"x": [0,1500,3500], "y": [0,1500,3500]}, "slab": {"Emod": 24000, "nu": 0.18, "t": 200}, "subgrade": {"Kx": 50, "Ky": 50, "Kz": 50}, "loads": {"x1": [100], "x2": [1000], "y1": [200], "y2": [2000], "q": [0.1]} } }'
```

Artifacts are written to `output_python_square/` and served from `GET /artifacts/...`.

Note: if you run the API via the current Docker image, the solver scripts are not bundled into the container, so `POST /analysis/generate` will fail unless you mount/copy them in.

### Run solver directly (CLI)

```bash
python3 DOF5_07.09.2025.py path/to/pavement-config.json
```

Expected outputs in `output_python_square/`:

- `pavement-config.generated.json`
- `results.vtk`
- `results.txt`
- `summary_results.txt`

### Query VTK results (scripts)

Detailed report:

```bash
python3 backend/scripts/vtk_detailed_report.py --file output_python_square/results.vtk --pretty
```

Aggregation examples:

```bash
python3 backend/scripts/vtk_aggregate_query.py --file output_python_square/results.vtk --field w --agg mean
python3 backend/scripts/vtk_aggregate_query.py --file output_python_square/results.vtk --field sxx_top --agg max
python3 backend/scripts/vtk_aggregate_query.py --file output_python_square/results.vtk --field displacement --agg p95 --component magnitude
```

Supported aggregations: `mean`, `max`, `min`, `sum`, `std`, `var`, `median`, `p95`, `p05`, `count`.

### Ask about VTK in chat

After an analysis run (or by pointing to a file), try prompts like:

- `analyze vtk output_python_square/results.vtk`
- `max sxx_top from vtk`
- `p95 displacement magnitude in output_python_square/results.vtk`

---

## Example prompts

- `5m x 5m slab, 250mm thick, Emod 30000 MPa, nu 0.18, K=60, mesh medium, load at edge`
- `Use corner loading, fine mesh, 0.35m x 0.30m contact patch, 80 kN wheel load`
- `Make it a 3.5m by 4.5m slab and use default subgrade values`

---

## Configuration notes (how inference works)

- If you set `mesh_type` and do not provide explicit `nodes.x`/`nodes.y`, the backend infers node coordinates that span `0..a` and `0..b`.
- If you set `load_location` and do not provide `x1/x2/y1/y2`, the backend infers a reasonable load patch near the corner/edge/interior.
- The backend enforces **hard** validation (for example Poisson’s ratio is constrained to typical pavement values) and adds **soft** warnings when values look physically implausible.

---

## Environment variables

Backend:

- `BACKEND_CORS_ORIGINS` — comma-separated list of allowed origins (defaults include common localhost dev ports)
- `BACKEND_ENV` — `dev`/`development`/`local` for more verbose error details
- `API_BASE_URL` — base URL used by some VTK lookup flows
- `VTK_AGENT_MODEL` — model tag used for the VTK router (default: `kimi-k2.6:cloud`)
- `OLLAMA_API_KEY` — optional Bearer token used when calling Ollama (can also be provided per-request)

Frontend:

- `VITE_API_BASE_URL` — backend base URL (default: `http://localhost:8000`)

---

## Tests

```bash
pip install -r requirements.txt
pip install pytest
pytest -q
```

---

## Troubleshooting

- **Frontend can’t reach backend**: verify `uvicorn` is running and `VITE_API_BASE_URL` points at it.
- **Backend can’t reach Ollama**: confirm `ollama serve`, the model name is correct, and the URL is reachable from where the backend runs (host vs container).
- **Model not found**: open `/ollama/status` in the UI; the backend can suggest close matches.
- **Analysis fails**: ensure the solver scripts exist in the same filesystem where the backend is running, and that the process has write access to `output_python_square/`.

---

## License

No license file is included in this repository. If you intend to publish or redistribute, add an explicit license.
