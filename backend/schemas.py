from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ConversationState(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    user_provided_keys: List[str] = Field(default_factory=list)
    current_asking: Optional[str] = None
    mode: Literal["welcome", "guided", "free", "complete"] = "welcome"
    welcomed: bool = False


class LLMConfig(BaseModel):
    ollama_url: str = "http://localhost:11434"
    model: str = "gemma3:12b"


class ChatRequest(BaseModel):
    user_input: str
    state: ConversationState
    llm_config: LLMConfig = Field(default_factory=LLMConfig)

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_alias(cls, data):
        if isinstance(data, dict) and "llm_config" not in data and "llm" in data:
            data = dict(data)
            data["llm_config"] = data["llm"]
        return data


class ChatResponse(BaseModel):
    assistant_message: str
    state: ConversationState
    final_params: Optional[Dict[str, Any]] = None


class ResetResponse(BaseModel):
    assistant_message: str
    state: ConversationState


class OllamaStatusResponse(BaseModel):
    connected: bool
    model_available: bool
    available_models: List[str]
    timeout_seconds: float
    error: Optional[str] = None
    detail: Optional[str] = None
    normalized_model: Optional[str] = None
    matched_model: Optional[str] = None
    model_suggestions: List[str] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
