from typing import List, Optional, Tuple

import requests
from langchain_ollama import ChatOllama

DEFAULT_OLLAMA_TIMEOUT = 5.0


def get_ollama_llm(base_url: str = "http://localhost:11434", model: str = "gemma3:12b"):
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.3,
        num_predict=1000,
    )


def check_ollama_connection(
    base_url: str = "http://localhost:11434",
    model_name: str = "gemma3:12b",
    timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT,
) -> Tuple[bool, bool, List[str], Optional[str]]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout_seconds)
        response.raise_for_status()
        models = response.json().get("models", [])
        available_models = [model.get("name", "") for model in models]
        model_available = any(model_name in m for m in available_models)
        return True, model_available, available_models, None
    except requests.exceptions.Timeout:
        return False, False, [], f"Connection timed out after {timeout_seconds:.1f}s."
    except requests.exceptions.ConnectionError:
        return False, False, [], "Unable to connect to Ollama. Is the server running?"
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        return False, False, [], f"Ollama returned HTTP {status_code}."
    except ValueError:
        return False, False, [], "Ollama returned an invalid JSON response."
    except requests.exceptions.RequestException as exc:
        return False, False, [], f"Ollama request failed: {exc}"
