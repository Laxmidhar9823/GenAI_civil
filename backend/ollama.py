from typing import Tuple

import requests
from langchain_ollama import ChatOllama


def get_ollama_llm(base_url: str = "http://localhost:11434", model: str = "gemma3:12b"):
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.3,
        num_predict=1000,
    )


def check_ollama_connection(
    base_url: str = "http://localhost:11434", model_name: str = "gemma3:12b"
) -> Tuple[bool, bool, list]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model.get("name", "") for model in models]
            model_available = any(model_name in m for m in available_models)
            return True, model_available, available_models
        return False, False, []
    except requests.exceptions.RequestException:
        return False, False, []
