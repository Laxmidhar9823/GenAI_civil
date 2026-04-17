# Rigid Pavement Parameter Configuration Agent

A Streamlit-based intelligent agent that uses Ollama with gemma3:12b model to parse natural language input and generate JSON configuration for rigid pavement analysis.

## Features

- 🤖 **Natural Language Processing**: Understands layman terms and technical jargon
- ✅ **Input Validation**: Validates constraints and provides meaningful error messages
- 📋 **Default Values**: Uses sensible defaults for missing parameters
- 🔄 **Interactive Flow**: Asks for confirmation before using defaults
- 📥 **JSON Export**: Download the generated configuration as a JSON file
- 🏠 **Local AI**: Runs completely locally using Ollama - no API keys needed!

## Prerequisites

1. **Install Ollama**: Download from [ollama.com](https://ollama.com/)

2. **Pull the gemma3:12b model**:
```bash
ollama pull gemma3:12b
```

3. **Ensure Ollama is running**:
```bash
ollama serve
```

## Installation

1. Clone or navigate to the project directory:
```bash
cd /home/surriya_gokul/GenAI_civil
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Running the FastAPI Backend

The repository now also includes a stateless backend in `backend/`.

```bash
uvicorn backend.main:app --reload
```

API will be available at `http://localhost:8000` with docs at `http://localhost:8000/docs`.

## Running the Web Frontend

The `frontend/` directory contains a React app.

```bash
cd frontend
npm install
npm run dev
```

By default it calls the backend at `http://localhost:8000`. To override:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

## Run with Docker Compose (optional)

From the repo root:

```bash
docker compose up --build
```

This starts:
- `api` on `http://localhost:8000`
- `web` on `http://localhost:5173`

The web image reads `VITE_API_BASE_URL` at build time. Default is `http://localhost:8000`.
To override:

```bash
VITE_API_BASE_URL=http://localhost:8000 docker compose up --build
```

Stop services:

```bash
docker compose down
```

> Ollama remains an external dependency (run `ollama serve` on your host and pull the model).

Backend endpoints:
- `GET /health`
- `GET /ollama/status?ollama_url=...&model=...&timeout_seconds=...`
- `GET /config/schema`
- `POST /reset`
- `POST /chat`

### VTK Result Workflow

1. Complete the assistant conversation and export the final JSON (nested `nodes/slab/subgrade/loads`).
2. Run the solver with that JSON as direct input:

```bash
python3 DOF5_07.09.2025.py path/to/pavement-config.json
```

3. Output files are generated in `output_python_square/`:
- `results.txt`
- `summary_results.txt`
- `results.vtk`

### Querying VTK Results from Python Scripts

Detailed report (fields, tuple counts, stats per component):

```bash
python3 backend/scripts/vtk_detailed_report.py --file output_python_square/results.vtk --pretty
```

Aggregation query examples:

```bash
python3 backend/scripts/vtk_aggregate_query.py --file output_python_square/results.vtk --field w --agg mean
python3 backend/scripts/vtk_aggregate_query.py --file output_python_square/results.vtk --field sxx_top --agg max
python3 backend/scripts/vtk_aggregate_query.py --file output_python_square/results.vtk --field displacement --agg p95 --component magnitude
```

Supported aggregations:
- `mean`, `max`, `min`, `sum`, `std`, `var`, `median`, `p95`, `p05`, `count`

### VTK Lookup Agent in Chat

The backend chat now includes a VTK lookup router. In the same conversation, users can ask things like:

- `analyze vtk output_python_square/results.vtk`
- `what is the mean deflection in vtk?`
- `max sxx_top from vtk`
- `p95 displacement magnitude in output_python_square/results.vtk`

The backend will pick the right VTK script, execute it, and return a clean aggregated result in chat.

Backend environment variables:
- `BACKEND_CORS_ORIGINS` (comma-separated) to override allowed CORS origins.
  - Defaults remain local dev origins (`localhost:3000`, `5173`, `5174`, `5175`, `8080`, `8501`) when not set.
- `BACKEND_ENV` (`development`/`dev`/`local`) to include extra exception details in error responses.

> Ollama must be running and the target model must be pulled (same as Streamlit mode).

## API Integration (stateless contract)

This repository currently ships a **Streamlit UI** (`app.py`). If you are building a separate frontend (React/Next/etc.), the backend exposes **stateless** endpoints that accept/return a `ConversationState` object each turn (no server-side sessions).

Endpoints:
- `POST /reset` → returns `{ assistant_message, state }` where `state` is a fresh `ConversationState`
- `POST /chat` → request `{ user_input, state, llm_config }` and response `{ assistant_message, state, final_params? }`
  - `llm_config` is `{ ollama_url, model }`
  - `final_params` is included when the conversation is complete in nested format:
    - `nodes` with default coordinates (`x` and `y` arrays)
    - `slab` (`Emod`, `nu`, `t`)
    - `subgrade` (`Kx`, `Ky`, `Kz`)
    - `loads` (`x1`, `x2`, `y1`, `y2`, `q`) as single-item arrays

`GET /config/schema` returns parameter definitions and defaults (both legacy UPPERCASE keys like `PARAM_INFO` and snake_case keys like `param_info`) so different frontends can integrate easily.

For parity with the existing UI, mirror the Streamlit flow in `app.py` (modes, `current_asking`, defaults-for-rest detection, and per-parameter validation).

## Usage

1. Ensure Ollama is running with gemma3:12b model (sidebar will show status)
2. Type your requirements in natural language in the chat input
3. The agent will extract parameters and ask about missing values
4. Confirm to use default values or provide additional details
5. Download the generated JSON configuration

## Example Inputs

- "I need a 4m square slab with elasticity 30000 MPa"
- "Set the tyre pressure to 0.5 MPa and load area from 100 to 800mm in both directions"
- "Use a 5 meter slab with Poisson ratio 0.25 and stiffness constants all equal to 100"
- "Modulus of elasticity 28000, slab size 4.5m, thickness 250mm"

## Parameter Reference

| Parameter | Description | Unit | Default | Constraints |
|-----------|-------------|------|---------|-------------|
| Emod | Modulus of Elasticity | MPa | 24000 | > 0 |
| nu | Poisson's Ratio | - | 0.3 | 0 to 0.5 |
| a | Slab Dimension (X) | mm | 3500 | > 0 |
| b | Slab Dimension (Y) | mm | 3500 | > 0 |
| t | Thickness | mm | 200 | > 0 |
| Kx | Stiffness Constant (X) | - | 50 | > 0 |
| Ky | Stiffness Constant (Y) | - | 50 | > 0 |
| Kz | Stiffness Constant (Z) | - | 50 | > 0 |
| x1 | Load Start (X) | mm | 100 | ≤ a |
| x2 | Load End (X) | mm | 1000 | ≤ a, > x1 |
| y1 | Load Start (Y) | mm | 200 | ≤ b |
| y2 | Load End (Y) | mm | 2000 | ≤ b, > y1 |
| q | Tyre Pressure | MPa | 0.1 | ≤ 0.7 |

## Constraints

- `x1`, `x2`, `y1`, `y2` cannot exceed slab dimensions `a` and `b`
- Tyre pressure `q` cannot exceed 0.7 MPa
- `x1` must be less than `x2`
- `y1` must be less than `y2`
- For square slabs, `a` = `b`

## License

MIT License
