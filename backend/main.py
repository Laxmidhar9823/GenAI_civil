import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.agent_logic import (
    DEFAULT_VALUES,
    NODE_DEFAULT_VALUES,
    NODE_PARAM_INFO,
    PARAM_CATEGORIES,
    PARAM_INFO,
    PARAM_ORDER,
    build_final_configuration,
    check_use_all_defaults_intent,
    process_user_input_with_llm,
    convert_to_standard_unit,
    format_value_with_unit,
    generate_completion_message,
    generate_autonomous_assistant_message,
    generate_interactive_followup,
    generate_welcome_message,
    validate_single_param,
    build_conversation_context,
    find_first_inconsistent_param,
    get_default_value,
    apply_implicit_inferences,
    normalize_load_location,
    normalize_mesh_type,
    with_progress_tail,
)
from backend.ollama import (
    check_ollama_connection,
    get_ollama_llm,
    normalize_model_name,
    resolve_model_match,
    suggest_available_models,
)
from backend.vtk_lookup_agent import handle_vtk_lookup_command
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationState,
    ErrorDetail,
    ErrorResponse,
    GenerateAnalysisRequest,
    GenerateAnalysisResponse,
    Message,
    OllamaStatusResponse,
    ResetResponse,
)

APP_NAME = "Pavement Assistant Backend"
APP_VERSION = "1.0.0"
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
APP_ENV = os.getenv("BACKEND_ENV", os.getenv("ENV", "production")).lower()
IS_DEV = APP_ENV in {"dev", "development", "local"}
REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_SCRIPT = REPO_ROOT / "DOF5_07.09.2025.py"
FEM_KERNEL_SCRIPT = REPO_ROOT / "fem_local_Plate20_DOF5_07_09_2025.py"
OUTPUT_DIR = REPO_ROOT / "output_python_square"
GENERATED_INPUT_FILE = OUTPUT_DIR / "pavement-config.generated.json"
VTK_DETAIL_SCRIPT = REPO_ROOT / "backend" / "scripts" / "vtk_detailed_report.py"


def _allowed_origins() -> list[str]:
    env_value = os.getenv("BACKEND_CORS_ORIGINS", "").strip()
    if not env_value:
        return DEFAULT_CORS_ORIGINS
    origins = [origin.strip() for origin in env_value.split(",") if origin.strip()]
    return origins or DEFAULT_CORS_ORIGINS


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _tail_text(text: str, max_chars: int = 2000) -> str:
    body = (text or "").strip()
    if len(body) <= max_chars:
        return body
    return body[-max_chars:]


app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.mount("/artifacts", StaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="artifacts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = {"request_id": _request_id(request), "path": str(request.url.path), "errors": exc.errors()}
    err = ErrorResponse(
        error=ErrorDetail(
            code="validation_error",
            message="Invalid request payload. Please check the request body and try again.",
            details=details,
        )
    )
    return JSONResponse(status_code=422, content=err.model_dump())


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    details = {"request_id": _request_id(request)}
    if IS_DEV:
        details.update(
            {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "path": str(request.url.path),
                "method": request.method,
            }
        )
    err = ErrorResponse(
        error=ErrorDetail(
            code="internal_error",
            message="An unexpected error occurred. Share request_id if you need support.",
            details=details,
        )
    )
    return JSONResponse(status_code=500, content=err.model_dump())


def _trim_messages(state: ConversationState, max_messages: int = 50, memory_max_chars: int = 20000) -> None:
    if len(state.messages) <= max_messages:
        return

    overflow = state.messages[:-max_messages]
    keep = state.messages[-max_messages:]

    # Compact the overflow into a rolling memory string.
    lines: list[str] = []
    for msg in overflow:
        role = msg.role
        if role == "user":
            label = "User"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = "System"

        content = (msg.content or "").strip().replace("\r\n", "\n")
        content = " ".join(content.split())
        if len(content) > 400:
            content = content[:400] + "..."
        if content:
            lines.append(f"{label}: {content}")

    snippet = "\n".join(lines).strip()
    if snippet:
        prior = (state.memory or "").strip()
        state.memory = (prior + "\n" + snippet).strip() if prior else snippet
        if len(state.memory) > memory_max_chars:
            state.memory = state.memory[-memory_max_chars:]

    state.messages = keep


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "service": "pavement-assistant-backend",
        "version": app.version,
        "environment": APP_ENV,
        "stateless": True,
    }


@app.get("/ollama/status", response_model=OllamaStatusResponse)
def ollama_status(
    ollama_url: str = Query("http://localhost:11434"),
    model: str = Query("qwen3.5:cloud"),
    timeout_seconds: float = Query(5.0, ge=1.0, le=30.0),
    x_ollama_api_key: str | None = Header(default=None, alias="x-ollama-api-key"),
):
    connected, model_available, available_models, error = check_ollama_connection(
        ollama_url, model, timeout_seconds, api_key=x_ollama_api_key
    )
    matched_model = resolve_model_match(model, available_models) if connected else None
    suggestions = (
        suggest_available_models(model, available_models)
        if connected and not model_available
        else []
    )
    detail = None
    if error:
        detail = error
    elif connected and not model_available:
        detail = f'Model "{model}" is not available on this Ollama server.'
        if suggestions:
            detail += f" Did you mean: {', '.join(suggestions)}?"
    elif connected and matched_model:
        detail = f'Model "{matched_model}" is available.'

    return OllamaStatusResponse(
        connected=connected,
        model_available=model_available,
        available_models=available_models,
        timeout_seconds=timeout_seconds,
        error=error,
        detail=detail,
        normalized_model=normalize_model_name(model),
        matched_model=matched_model,
        model_suggestions=suggestions,
    )


@app.get("/config/schema")
def config_schema() -> Dict[str, object]:
    final_output_defaults = build_final_configuration({})

    # Return both legacy (UPPERCASE) and API (snake_case) keys for frontend compatibility.
    return {
        "PARAM_INFO": PARAM_INFO,
        "DEFAULT_VALUES": DEFAULT_VALUES,
        "NODE_DEFAULT_VALUES": NODE_DEFAULT_VALUES,
        "NODE_PARAM_INFO": NODE_PARAM_INFO,
        "PARAM_ORDER": PARAM_ORDER,
        "PARAM_CATEGORIES": PARAM_CATEGORIES,
        "FINAL_OUTPUT_DEFAULTS": final_output_defaults,
        "param_info": PARAM_INFO,
        "default_values": DEFAULT_VALUES,
        "node_default_values": NODE_DEFAULT_VALUES,
        "node_param_info": NODE_PARAM_INFO,
        "param_order": PARAM_ORDER,
        "param_categories": PARAM_CATEGORIES,
        "final_output_defaults": final_output_defaults,
    }


@app.post("/reset", response_model=ResetResponse)
def reset() -> ResetResponse:
    welcome = generate_welcome_message()
    state = ConversationState(
        messages=[Message(role="assistant", content=welcome)],
        params={},
        user_provided_keys=[],
        current_asking=None,
        mode="welcome",
        welcomed=True,
        analysis_generated=False,
        analysis_vtk_file=None,
        analysis_summary_file=None,
        analysis_plot_files=[],
    )
    return ResetResponse(assistant_message=welcome, state=state)


@app.post("/analysis/generate", response_model=GenerateAnalysisResponse)
def generate_analysis(req: GenerateAnalysisRequest) -> GenerateAnalysisResponse:
    if not isinstance(req.final_params, dict) or not req.final_params:
        return GenerateAnalysisResponse(
            success=False,
            message="Missing final_params payload. Please complete and confirm configuration first.",
        )

    required_sections = {"nodes", "slab", "subgrade", "loads"}
    missing_sections = sorted(required_sections.difference(req.final_params.keys()))
    if missing_sections:
        return GenerateAnalysisResponse(
            success=False,
            message=(
                "final_params is not in unified solver format. "
                f"Missing section(s): {', '.join(missing_sections)}."
            ),
        )

    try:
        if not SOLVER_SCRIPT.exists():
            return GenerateAnalysisResponse(
                success=False,
                message=f"Solver script is missing: {SOLVER_SCRIPT}",
            )

        if not FEM_KERNEL_SCRIPT.exists():
            return GenerateAnalysisResponse(
                success=False,
                message=(
                    "Cannot run analysis: required FEM kernel file is missing "
                    f"({FEM_KERNEL_SCRIPT.name})."
                ),
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        GENERATED_INPUT_FILE.write_text(
            json.dumps(req.final_params, indent=2),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [sys.executable, str(SOLVER_SCRIPT), str(GENERATED_INPUT_FILE)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )

        combined_output = "\n".join(
            chunk for chunk in [completed.stdout.strip(), completed.stderr.strip()] if chunk
        ).strip()

        if completed.returncode != 0:
            return GenerateAnalysisResponse(
                success=False,
                message="Analysis run failed while executing the solver script.",
                input_json_path=str(GENERATED_INPUT_FILE),
                output_dir=str(OUTPUT_DIR),
                solver_stdout=_tail_text(combined_output),
            )

        vtk_file = OUTPUT_DIR / "results.vtk"
        results_file = OUTPUT_DIR / "results.txt"
        summary_file = OUTPUT_DIR / "summary_results.txt"

        if not vtk_file.exists():
            return GenerateAnalysisResponse(
                success=False,
                message="Solver completed but results.vtk was not generated.",
                input_json_path=str(GENERATED_INPUT_FILE),
                output_dir=str(OUTPUT_DIR),
                results_file=str(results_file) if results_file.exists() else None,
                summary_file=str(summary_file) if summary_file.exists() else None,
                solver_stdout=_tail_text(combined_output),
            )

        detailed_report: Dict[str, Any] | None = None
        if VTK_DETAIL_SCRIPT.exists():
            detail_cmd = subprocess.run(
                [sys.executable, str(VTK_DETAIL_SCRIPT), "--file", str(vtk_file)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if detail_cmd.returncode == 0 and detail_cmd.stdout.strip():
                try:
                    detailed_report = json.loads(detail_cmd.stdout)
                except json.JSONDecodeError:
                    detailed_report = None

        plot_files = [f"/artifacts/{p.name}" for p in sorted(OUTPUT_DIR.glob("*.png"))]

        return GenerateAnalysisResponse(
            success=True,
            message=(
                "Analysis generated successfully. VTK and summary artifacts were updated in "
                "output_python_square/."
            ),
            input_json_path=str(GENERATED_INPUT_FILE),
            output_dir=str(OUTPUT_DIR),
            vtk_file=str(vtk_file),
            results_file=str(results_file) if results_file.exists() else None,
            summary_file=str(summary_file) if summary_file.exists() else None,
            detailed_report=detailed_report,
            solver_stdout=_tail_text(combined_output),
            plot_files=plot_files,
        )
    except subprocess.TimeoutExpired:
        return GenerateAnalysisResponse(
            success=False,
            message="Analysis timed out after 10 minutes.",
            input_json_path=str(GENERATED_INPUT_FILE),
            output_dir=str(OUTPUT_DIR),
        )
    except Exception as exc:
        return GenerateAnalysisResponse(
            success=False,
            message=f"Analysis failed: {exc}",
            input_json_path=str(GENERATED_INPUT_FILE),
            output_dir=str(OUTPUT_DIR),
        )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    state = req.state.model_copy(deep=True)

    state.messages.append(Message(role="user", content=req.user_input))

    analysis_vtk = (state.analysis_vtk_file or "").strip() if state.analysis_generated else ""
    lookup_response = handle_vtk_lookup_command(
        req.user_input,
        default_vtk_file=analysis_vtk or None,
        allow_implicit=bool(analysis_vtk),
    )
    if lookup_response is not None:
        if state.mode == "welcome":
            state.mode = "free"

        response = with_progress_tail(
            lookup_response,
            state.params,
            state.current_asking,
            include_progress=(state.mode != "complete"),
        )
        state.messages.append(Message(role="assistant", content=response))
        _trim_messages(state)
        final_params = build_final_configuration(state.params) if state.mode == "complete" else None
        return ChatResponse(assistant_message=response, state=state, final_params=final_params)

    # Build extraction context from recent conversation + collected params.
    # This enables multi-turn, stateful assistance (values, corrections, and missing-info followups).
    context = build_conversation_context(
        [m.model_dump() for m in state.messages],
        state.params,
        state.memory,
    )
    lower_input = req.user_input.lower().strip()

    response = ""
    model_notice = ""

    # Prepare LLM for both guided and free-form turns.
    requested_model = req.llm_config.model
    api_key = (req.llm_config.api_key or "").strip() or None
    effective_model = requested_model
    connected, model_available, available_models, _ = check_ollama_connection(
        req.llm_config.ollama_url,
        requested_model,
        timeout_seconds=3.0,
        api_key=api_key,
    )

    # If no API key is provided, prefer a likely-local model to avoid 401s on cloud models.
    if connected and not api_key and available_models:
        requested_norm = normalize_model_name(requested_model)
        local_candidates = [m for m in available_models if "cloud" not in normalize_model_name(m)]
        if "cloud" in requested_norm and local_candidates:
            effective_model = local_candidates[0]
            model_notice = (
                f"No API key provided for cloud model '{requested_model}'. "
                f"Using local model '{effective_model}' instead."
            )

    if connected and not model_available and available_models and effective_model == requested_model:
        effective_model = available_models[0]
        model_notice = (
            f"Model '{requested_model}' is not installed on your Ollama server. "
            f"Using available model '{effective_model}' instead."
        )

    llm = get_ollama_llm(req.llm_config.ollama_url, effective_model, api_key=api_key)

    if lower_input in ["let's begin", "lets begin", "start", "begin", "guide me", "help me"]:
        state.mode = "guided"
        missing = [k for k in PARAM_ORDER if k not in state.params]
        if not missing:
            state.mode = "complete"
            response = generate_completion_message(state.params, state.user_provided_keys)
        else:
            state.current_asking = missing[0]
            response = (
                "Great, let's do this together. You can share as many details as you already know in one message, "
                "and I'll capture all of them at once.\n\n"
            )
            response += generate_interactive_followup(state.params)

        # Narrate guided start via the LLM (fallback: deterministic text above).
        response = generate_autonomous_assistant_message(
            llm=llm,
            memory=state.memory,
            user_input=req.user_input,
            mode=state.mode,
            params=state.params,
            applied_keys=[],
            inference_notes=[],
            conversion_notes=[],
            problems=[],
            current_asking=state.current_asking,
            conversation_only=True,
            fallback_message=response,
        )
    else:
        result = process_user_input_with_llm(
            req.user_input,
            llm,
            context,
            state.params,
            state.current_asking,
        )

        conversation_only = bool(result.get("conversation_only"))

        extracted_multiple = result.get("extracted_multiple") or {}
        extracted_units = result.get("extracted_units") or {}
        single_value = result.get("understood_value")
        single_key = result.get("parameter_key") or state.current_asking

        use_all_defaults = bool(result.get("use_all_defaults") or check_use_all_defaults_intent(req.user_input))
        if use_all_defaults:
            prefill_updates: Dict[str, object] = {}
            prefill_explicit_keys = set()

            if isinstance(extracted_multiple, dict):
                for key, val in extracted_multiple.items():
                    if key in PARAM_ORDER and isinstance(val, (int, float)):
                        prefill_updates[key] = float(val)
                        prefill_explicit_keys.add(key)
                    elif key in {"x1", "x2", "y1", "y2"} and isinstance(val, (int, float)):
                        prefill_updates[key] = float(val)
                        prefill_explicit_keys.add(key)
                    elif key in {"mesh_type", "load_location"} and isinstance(val, str):
                        prefill_updates[key] = val.strip().lower()
                        prefill_explicit_keys.add(key)
                    elif key in {"x", "y"} and isinstance(val, list) and all(
                        isinstance(item, (int, float)) for item in val
                    ):
                        prefill_updates[key] = [float(item) for item in val]
                        prefill_explicit_keys.add(key)
                    elif key == "load_cases" and isinstance(val, list):
                        normalized_cases = []
                        for case in val:
                            if not isinstance(case, dict):
                                continue
                            required = ["x1", "x2", "y1", "y2", "q"]
                            if not all(k in case and isinstance(case[k], (int, float)) for k in required):
                                continue
                            normalized_cases.append({k: float(case[k]) for k in required})
                        if normalized_cases:
                            prefill_updates[key] = normalized_cases
                            prefill_explicit_keys.add(key)

            if isinstance(single_value, (int, float)) and single_key in PARAM_ORDER:
                prefill_updates[single_key] = float(single_value)
                prefill_explicit_keys.add(single_key)
            elif isinstance(single_value, (int, float)) and single_key in {"x1", "x2", "y1", "y2"}:
                prefill_updates[single_key] = float(single_value)
                prefill_explicit_keys.add(single_key)
            elif isinstance(single_value, str) and single_key in {"mesh_type", "load_location"}:
                prefill_updates[single_key] = single_value.strip().lower()
                prefill_explicit_keys.add(single_key)
            elif isinstance(single_value, list) and single_key in {"x", "y"} and all(
                isinstance(item, (int, float)) for item in single_value
            ):
                prefill_updates[single_key] = [float(item) for item in single_value]
                prefill_explicit_keys.add(single_key)

            tentative = dict(state.params)
            explicit_applied: List[str] = []
            for key, value in prefill_updates.items():
                if key == "load_cases" and isinstance(value, list):
                    tentative["load_cases"] = value
                    explicit_applied.append(key)
                    continue

                working_value = value
                unit_for_key = extracted_units.get(key)
                if key == single_key and result.get("original_unit"):
                    unit_for_key = result.get("original_unit")
                if unit_for_key and isinstance(working_value, (int, float)):
                    converted_value, _ = convert_to_standard_unit(float(working_value), unit_for_key, key)
                    working_value = converted_value

                if key == "mesh_type" and isinstance(working_value, str):
                    working_value = normalize_mesh_type(working_value) or working_value
                if key == "load_location" and isinstance(working_value, str):
                    working_value = normalize_load_location(working_value) or working_value

                is_valid, _ = validate_single_param(key, working_value, tentative)
                if is_valid:
                    tentative[key] = working_value
                    explicit_applied.append(key)

            if explicit_applied:
                state.params = tentative
                for key in explicit_applied:
                    if key in prefill_explicit_keys and key != "load_cases" and key not in state.user_provided_keys:
                        state.user_provided_keys.append(key)

            missing = [k for k in PARAM_ORDER if k not in state.params]
            for key in missing:
                state.params[key] = get_default_value(key)

            _, inference_notes = apply_implicit_inferences(state.params, state.user_provided_keys)

            state.mode = "complete"
            if missing:
                response = f"Perfect. I filled the remaining {len(missing)} values with safe defaults.\n\n"
            else:
                response = "Everything is already filled in.\n\n"
            if inference_notes:
                response += "\n".join(f"- {note}" for note in inference_notes) + "\n\n"
            response += generate_completion_message(state.params, state.user_provided_keys)

            # Let the LLM narrate the completed state (fallback: deterministic completion message above).
            response = generate_autonomous_assistant_message(
                llm=llm,
                memory=state.memory,
                user_input=req.user_input,
                mode=state.mode,
                params=state.params,
                applied_keys=list(prefill_explicit_keys),
                inference_notes=inference_notes,
                conversion_notes=[],
                problems=[],
                current_asking=state.current_asking,
                conversation_only=conversation_only,
                fallback_message=response,
            )
        else:
            updates: Dict[str, object] = {}
            explicit_keys = set()
            default_applied_keys = set()

            if isinstance(extracted_multiple, dict):
                for key, val in extracted_multiple.items():
                    if key in PARAM_ORDER and isinstance(val, (int, float)):
                        updates[key] = float(val)
                        explicit_keys.add(key)
                    elif key in {"x1", "x2", "y1", "y2"} and isinstance(val, (int, float)):
                        updates[key] = float(val)
                        explicit_keys.add(key)
                    elif key in {"mesh_type", "load_location"} and isinstance(val, str):
                        updates[key] = val.strip().lower()
                        explicit_keys.add(key)
                    elif key in {"x", "y"} and isinstance(val, list) and all(
                        isinstance(item, (int, float)) for item in val
                    ):
                        updates[key] = [float(item) for item in val]
                        explicit_keys.add(key)
                    elif key == "load_cases" and isinstance(val, list):
                        normalized_cases = []
                        for case in val:
                            if not isinstance(case, dict):
                                continue
                            required = ["x1", "x2", "y1", "y2", "q"]
                            if not all(k in case and isinstance(case[k], (int, float)) for k in required):
                                continue
                            normalized_cases.append({k: float(case[k]) for k in required})
                        if normalized_cases:
                            updates[key] = normalized_cases
                            explicit_keys.add(key)

            if isinstance(single_value, (int, float)) and single_key in PARAM_ORDER:
                updates[single_key] = float(single_value)
                explicit_keys.add(single_key)
            elif isinstance(single_value, (int, float)) and single_key in {"x1", "x2", "y1", "y2"}:
                updates[single_key] = float(single_value)
                explicit_keys.add(single_key)
            elif isinstance(single_value, str) and single_key in {"mesh_type", "load_location"}:
                updates[single_key] = single_value.strip().lower()
                explicit_keys.add(single_key)
            elif isinstance(single_value, list) and single_key in {"x", "y"} and all(
                isinstance(item, (int, float)) for item in single_value
            ):
                updates[single_key] = [float(item) for item in single_value]
                explicit_keys.add(single_key)

            if result.get("use_default") and state.current_asking in PARAM_ORDER:
                key = state.current_asking
                if key not in updates:
                    updates[key] = get_default_value(key)
                    default_applied_keys.add(key)

            tentative = dict(state.params)
            applied: List[str] = []
            applied_load_cases = False
            problems: List[str] = []
            conversion_notes: List[str] = []

            for key, value in updates.items():
                if key == "load_cases" and isinstance(value, list):
                    tentative["load_cases"] = value
                    applied_load_cases = True
                    continue

                working_value = value
                unit_for_key = extracted_units.get(key)
                if key == single_key and result.get("original_unit"):
                    unit_for_key = result.get("original_unit")
                if unit_for_key and isinstance(working_value, (int, float)):
                    converted_value, conversion_msg = convert_to_standard_unit(
                        float(working_value), unit_for_key, key
                    )
                    working_value = converted_value
                    if conversion_msg:
                        conversion_notes.append(conversion_msg)

                if key == "mesh_type" and isinstance(working_value, str):
                    working_value = normalize_mesh_type(working_value) or working_value
                if key == "load_location" and isinstance(working_value, str):
                    working_value = normalize_load_location(working_value) or working_value

                is_valid, error_msg = validate_single_param(key, working_value, tentative)
                if is_valid:
                    tentative[key] = working_value
                    applied.append(key)
                else:
                    info = PARAM_INFO.get(key, {"name": key})
                    problems.append(f"{info['name']}: {error_msg}")

            if applied or applied_load_cases:
                state.params = tentative
                for key in applied:
                    if key in explicit_keys and key not in default_applied_keys and key not in state.user_provided_keys:
                        state.user_provided_keys.append(key)

                inferred_keys, inference_notes = apply_implicit_inferences(state.params, state.user_provided_keys)
                for inferred_key in inferred_keys:
                    if inferred_key in tentative:
                        tentative[inferred_key] = state.params[inferred_key]
                applied.extend([k for k in inferred_keys if k not in applied])
            else:
                inference_notes = []

            if not applied and not problems:
                response = result.get("friendly_response") or (
                    "I want to make sure I capture this correctly. You can share multiple details together, "
                    "for example: length, width, thickness, and pressure in one message."
                )
            else:
                parts: List[str] = []
                if result.get("friendly_response"):
                    parts.append(str(result["friendly_response"]).strip())

                if conversion_notes:
                    parts.append("\n".join(f"📐 {note}" for note in conversion_notes))

                if inference_notes:
                    parts.append("\n".join(f"🧠 {note}" for note in inference_notes))

                if applied:
                    captured = "\n".join(
                        f"- **{PARAM_INFO[k]['name']}**: **{format_value_with_unit(state.params[k], k)}**"
                        for k in applied
                    )
                    parts.append(f"Captured now:\n{captured}")

                if problems:
                    issues = "\n".join(f"- {p}" for p in problems)
                    parts.append(f"I need one quick correction:\n{issues}")

                response = "\n\n".join(p for p in parts if p)

            # If this turn was clearly conversational (e.g., "hi", "what is Kz?"),
            # do NOT force the user into the guided data-entry flow.
            if conversation_only and not applied and not applied_load_cases and not problems and not default_applied_keys:
                if state.mode == "welcome":
                    state.mode = "free"
                # Keep current_asking as-is (if we were already in guided mode).
            else:
                bad = find_first_inconsistent_param(state.params)
                if bad:
                    bad_key, bad_msg = bad
                    response = f"⚠️ One thing to fix before we continue: {bad_msg}\n\n"
                    response += (
                        f"Please update **{PARAM_INFO[bad_key]['name']}**, or say "
                        f"'default' to use **{format_value_with_unit(get_default_value(bad_key), bad_key)}**."
                    )
                    state.current_asking = bad_key
                    state.mode = "guided"
                else:
                    missing = [k for k in PARAM_ORDER if k not in state.params]
                    if missing:
                        state.current_asking = missing[0]
                        state.mode = "guided"
                        response = f"{response}\n\n{generate_interactive_followup(state.params)}".strip()
                    else:
                        state.current_asking = None
                        state.mode = "complete"
                        response = f"{response}\n\n{generate_completion_message(state.params, state.user_provided_keys)}".strip()

            if not response:
                response = (
                    "I want to make sure I understood you correctly. "
                    "Please share any values you know, and I can capture several at once."
                )

            # Replace the mostly hard-coded assembly above with an autonomous LLM narration.
            # If the LLM is unavailable, we keep the deterministic response as fallback.
            response = generate_autonomous_assistant_message(
                llm=llm,
                memory=state.memory,
                user_input=req.user_input,
                mode=state.mode,
                params=state.params,
                applied_keys=applied,
                inference_notes=inference_notes,
                conversion_notes=conversion_notes,
                problems=problems,
                current_asking=state.current_asking,
                conversation_only=conversation_only,
                fallback_message=response,
            )

    if model_notice:
        response = f"{model_notice}\n\n{response}".strip()

    response = with_progress_tail(
        response,
        state.params,
        state.current_asking,
        include_progress=(state.mode != "complete"),
    )

    state.messages.append(Message(role="assistant", content=response))
    _trim_messages(state)

    final_params = build_final_configuration(state.params) if state.mode == "complete" else None
    return ChatResponse(assistant_message=response, state=state, final_params=final_params)
