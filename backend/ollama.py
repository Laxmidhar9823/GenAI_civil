import os
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple

import requests
from langchain_ollama import ChatOllama

DEFAULT_OLLAMA_TIMEOUT = 5.0


def _build_auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    raw = (api_key or os.getenv("OLLAMA_API_KEY", "")).strip()
    if not raw:
        return {}
    if raw.lower().startswith("bearer "):
        token = raw.split(" ", 1)[1].strip()
    else:
        token = raw
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def get_ollama_llm(
    base_url: str = "http://localhost:11434",
    model: str = "qwen3.5:cloud",
    api_key: Optional[str] = None,
):
    headers = _build_auth_headers(api_key)
    kwargs = {"client_kwargs": {"headers": headers}} if headers else {}
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.3,
        num_predict=1000,
        **kwargs,
    )


def normalize_model_name(model_name: str) -> str:
    normalized = (model_name or "").strip().lower().replace("：", ":")
    for source, target in ((" ", ""), ("_", "-"), ("/", "-")):
        normalized = normalized.replace(source, target)
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized


def resolve_model_match(requested_model: str, available_models: List[str]) -> Optional[str]:
    requested_normalized = normalize_model_name(requested_model)
    if not requested_normalized:
        return None

    available_pairs = [
        (model_name, normalize_model_name(model_name)) for model_name in available_models if model_name
    ]

    for original, normalized in available_pairs:
        if normalized == requested_normalized:
            return original

    # Backward compatible: if no tag requested, allow matching by model family.
    if ":" not in requested_normalized:
        requested_family = requested_normalized.split(":", 1)[0]
        for original, normalized in available_pairs:
            if normalized.split(":", 1)[0] == requested_family:
                return original
    return None


def suggest_available_models(requested_model: str, available_models: List[str], limit: int = 3) -> List[str]:
    requested_normalized = normalize_model_name(requested_model)
    if not requested_normalized or not available_models:
        return []

    normalized_to_original = {
        normalize_model_name(model_name): model_name for model_name in available_models if model_name
    }
    normalized_names = list(normalized_to_original.keys())

    suggestions: List[str] = []
    for match in get_close_matches(requested_normalized, normalized_names, n=limit, cutoff=0.45):
        original = normalized_to_original.get(match)
        if original and original not in suggestions:
            suggestions.append(original)

    if ":" in requested_normalized:
        requested_family = requested_normalized.split(":", 1)[0]
        for model_name in available_models:
            normalized_model = normalize_model_name(model_name)
            if normalized_model.startswith(f"{requested_family}:") and model_name not in suggestions:
                suggestions.append(model_name)
            if len(suggestions) >= limit:
                break

    return suggestions[:limit]


def check_ollama_connection(
    base_url: str = "http://localhost:11434",
    model_name: str = "qwen3.5:cloud",
    timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT,
    api_key: Optional[str] = None,
) -> Tuple[bool, bool, List[str], Optional[str]]:
    normalized_url = (base_url or "").strip().rstrip("/")
    endpoint = f"{normalized_url}/api/tags"
    headers = _build_auth_headers(api_key)

    try:
        response = requests.get(endpoint, timeout=timeout_seconds, headers=headers or None)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return False, False, [], "Ollama returned an unexpected response shape."

        models = payload.get("models", [])
        available_models = [
            model.get("name", "")
            for model in models
            if isinstance(model, dict) and isinstance(model.get("name", ""), str)
        ]
        model_available = resolve_model_match(model_name, available_models) is not None
        return True, model_available, available_models, None
    except requests.exceptions.InvalidURL:
        return False, False, [], f'Invalid Ollama URL "{base_url}". Include protocol and host, e.g. http://localhost:11434.'
    except requests.exceptions.MissingSchema:
        return False, False, [], f'Ollama URL "{base_url}" is missing a protocol (http:// or https://).'
    except requests.exceptions.Timeout:
        return False, False, [], f"Ollama server timeout after {timeout_seconds:.1f}s at {normalized_url}."
    except requests.exceptions.ConnectionError:
        return False, False, [], f"Cannot reach Ollama at {normalized_url}. Verify URL and that the server is running."
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        if status_code == 401:
            return False, False, [], "Ollama Cloud authorization failed (401). Provide a valid API key."
        return False, False, [], f"Ollama API request to {endpoint} failed with HTTP {status_code}."
    except ValueError:
        return False, False, [], "Ollama returned an invalid JSON response."
    except requests.exceptions.RequestException as exc:
        return False, False, [], f"Ollama request failed: {exc}"
