import os
import uuid
from typing import Dict

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.agent_logic import (
    DEFAULT_VALUES,
    PARAM_CATEGORIES,
    PARAM_INFO,
    PARAM_ORDER,
    check_use_all_defaults_intent,
    process_user_input_with_llm,
    convert_to_standard_unit,
    format_value_with_unit,
    generate_completion_message,
    generate_welcome_message,
    get_friendly_param_question,
    validate_single_param,
    build_conversation_context,
    find_first_inconsistent_param,
)
from backend.ollama import check_ollama_connection, get_ollama_llm
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationState,
    ErrorDetail,
    ErrorResponse,
    Message,
    OllamaStatusResponse,
    ResetResponse,
)

APP_NAME = "Pavement Assistant Backend"
APP_VERSION = "1.0.0"
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://localhost:8501",
]
APP_ENV = os.getenv("BACKEND_ENV", os.getenv("ENV", "production")).lower()
IS_DEV = APP_ENV in {"dev", "development", "local"}


def _allowed_origins() -> list[str]:
    env_value = os.getenv("BACKEND_CORS_ORIGINS", "").strip()
    if not env_value:
        return DEFAULT_CORS_ORIGINS
    origins = [origin.strip() for origin in env_value.split(",") if origin.strip()]
    return origins or DEFAULT_CORS_ORIGINS


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
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


def _trim_messages(state: ConversationState, max_messages: int = 50) -> None:
    if len(state.messages) > max_messages:
        state.messages = state.messages[-max_messages:]


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
    model: str = Query("gemma3:12b"),
    timeout_seconds: float = Query(5.0, ge=1.0, le=30.0),
):
    connected, model_available, available_models, error = check_ollama_connection(
        ollama_url, model, timeout_seconds
    )
    return OllamaStatusResponse(
        connected=connected,
        model_available=model_available,
        available_models=available_models,
        timeout_seconds=timeout_seconds,
        error=error,
    )


@app.get("/config/schema")
def config_schema() -> Dict[str, object]:
    # Return both legacy (UPPERCASE) and API (snake_case) keys for frontend compatibility.
    return {
        "PARAM_INFO": PARAM_INFO,
        "DEFAULT_VALUES": DEFAULT_VALUES,
        "PARAM_ORDER": PARAM_ORDER,
        "PARAM_CATEGORIES": PARAM_CATEGORIES,
        "param_info": PARAM_INFO,
        "default_values": DEFAULT_VALUES,
        "param_order": PARAM_ORDER,
        "param_categories": PARAM_CATEGORIES,
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
    )
    return ResetResponse(assistant_message=welcome, state=state)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    state = req.state.model_copy(deep=True)

    state.messages.append(Message(role="user", content=req.user_input))
    context = build_conversation_context([m.model_dump() for m in state.messages], state.params)
    lower_input = req.user_input.lower().strip()

    response = ""

    if lower_input in ["let's begin", "lets begin", "start", "begin", "guide me", "help me"]:
        state.mode = "guided"
        missing = [k for k in PARAM_ORDER if k not in state.params]
        if missing:
            state.current_asking = missing[0]
            response = "Great! Let's go through this step by step. 😊\n\n"
            response += get_friendly_param_question(missing[0], state.params)
        else:
            state.mode = "complete"
            response = generate_completion_message(state.params, state.user_provided_keys)

    elif check_use_all_defaults_intent(req.user_input):
        missing = [k for k in PARAM_ORDER if k not in state.params]
        for key in missing:
            state.params[key] = DEFAULT_VALUES[key]
        state.mode = "complete"
        if missing:
            response = f"Perfect! I've filled in the remaining {len(missing)} values with defaults. 👍\n\n"
        else:
            response = "Great! All parameters were already set. 👍\n\n"

        bad = find_first_inconsistent_param(state.params)
        if bad:
            bad_key, bad_msg = bad
            state.mode = "guided"
            state.current_asking = bad_key
            response += f"⚠️ One quick check: {bad_msg}\n\n"
            response += get_friendly_param_question(bad_key, state.params)
        else:
            response += generate_completion_message(state.params, state.user_provided_keys)

    else:
        llm = get_ollama_llm(req.llm_config.ollama_url, req.llm_config.model)
        result = process_user_input_with_llm(req.user_input, llm, context, state.current_asking)

        response = result.get("friendly_response", "")

        # If the LLM extracted multiple parameters, apply them first so the "next question" logic
        # doesn't re-ask for values that were already provided.
        if result.get("extracted_multiple"):
            for key, val in result["extracted_multiple"].items():
                if key in DEFAULT_VALUES and key not in state.params:
                    is_valid, _ = validate_single_param(key, val, state.params)
                    if is_valid:
                        state.params[key] = val
                        if key not in state.user_provided_keys:
                            state.user_provided_keys.append(key)

        # Match the Streamlit reducer semantics in app.py
        if result.get("understood_value") is not None:
            value = result["understood_value"]
            param_key = result.get("parameter_key") or state.current_asking

            if param_key:
                original_unit = result.get("original_unit")
                if original_unit:
                    converted_value, conversion_msg = convert_to_standard_unit(value, original_unit, param_key)
                    if conversion_msg:
                        response = f"📐 {conversion_msg}\n\n" + response
                    value = converted_value

                is_valid, error_msg = validate_single_param(param_key, value, state.params)

                if is_valid:
                    state.params[param_key] = value
                    if param_key not in state.user_provided_keys:
                        state.user_provided_keys.append(param_key)

                    info = PARAM_INFO[param_key]
                    response = (
                        f"✅ Got it! **{info['name']}** is set to **{format_value_with_unit(value, param_key)}**.\n\n"
                    )

                    bad = find_first_inconsistent_param(state.params)
                    if bad:
                        bad_key, bad_msg = bad
                        state.mode = "guided"
                        state.current_asking = bad_key
                        response += f"⚠️ One quick check: {bad_msg}\n\n"
                        response += get_friendly_param_question(bad_key, state.params)
                    else:
                        missing = [k for k in PARAM_ORDER if k not in state.params]
                        if missing:
                            state.current_asking = missing[0]
                            response += "Moving on to the next setting...\n\n"
                            response += get_friendly_param_question(missing[0], state.params)
                        else:
                            state.mode = "complete"
                            response += generate_completion_message(state.params, state.user_provided_keys)
                else:
                    response = f"⚠️ Hmm, that value has an issue: {error_msg}\n\n"
                    response += (
                        "Could you try a different value? Or say 'default' to use "
                        f"**{format_value_with_unit(DEFAULT_VALUES[param_key], param_key)}**."
                    )

        elif result.get("use_all_defaults"):
            missing = [k for k in PARAM_ORDER if k not in state.params]
            for key in missing:
                state.params[key] = DEFAULT_VALUES[key]
            state.mode = "complete"

            if missing:
                response = f"Perfect! I've filled in the remaining {len(missing)} values with defaults. 👍\n\n"
            else:
                response = "Great! All parameters were already set. 👍\n\n"

            bad = find_first_inconsistent_param(state.params)
            if bad:
                bad_key, bad_msg = bad
                state.mode = "guided"
                state.current_asking = bad_key
                response += f"⚠️ One quick check: {bad_msg}\n\n"
                response += get_friendly_param_question(bad_key, state.params)
            else:
                response += generate_completion_message(state.params, state.user_provided_keys)

        elif result.get("use_default") and state.current_asking:
            param_key = state.current_asking
            default_val = DEFAULT_VALUES[param_key]
            state.params[param_key] = default_val

            info = PARAM_INFO[param_key]
            response = (
                f"👍 No problem! Using the default value of **{format_value_with_unit(default_val, param_key)}** "
                f"for {info['name']}.\n\n"
            )

            bad = find_first_inconsistent_param(state.params)
            if bad:
                bad_key, bad_msg = bad
                state.mode = "guided"
                state.current_asking = bad_key
                response += f"⚠️ One quick check: {bad_msg}\n\n"
                response += get_friendly_param_question(bad_key, state.params)
            else:
                missing = [k for k in PARAM_ORDER if k not in state.params]
                if missing:
                    state.current_asking = missing[0]
                    response += get_friendly_param_question(missing[0], state.params)
                else:
                    state.mode = "complete"
                    response += generate_completion_message(state.params, state.user_provided_keys)


        if not response:
            response = (
                "I'm not quite sure I understood that. Could you try again? "
                "You can enter a number, or say 'default' to use the suggested value. 😊"
            )

    state.messages.append(Message(role="assistant", content=response))
    _trim_messages(state)

    final_params = {**DEFAULT_VALUES, **state.params} if state.mode == "complete" else None
    return ChatResponse(assistant_message=response, state=state, final_params=final_params)
