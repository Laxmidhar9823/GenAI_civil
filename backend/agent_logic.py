import json
import math
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.ollama import invoke_ollama_multimodal_chat

DEFAULT_VALUES = {
    "Emod": 24000.0,
    "nu": 0.18,
    "a": 3500,
    "b": 3500,
    "t": 200.0,
    "Kx": 50,
    "Ky": 50,
    "Kz": 50,
    "x1": 100.0,
    "x2": 1000.0,
    "y1": 200.0,
    "y2": 2000.0,
    "q": 0.1,
    "mesh_type": "coarse",
    "load_location": "interior",
}

NODE_DEFAULT_VALUES = {
    "x": [0.0, 350.0, 700.0, 1050.0, 1400.0, 1750.0, 2100.0, 2450.0, 2800.0, 3150.0, 3500.0],
    "y": [0.0, 350.0, 700.0, 1050.0, 1400.0, 1750.0, 2100.0, 2450.0, 2800.0, 3150.0, 3500.0],
}

NODE_PARAM_INFO = {
    "x": {
        "name": "Node Coordinates (X)",
        "technical_name": "Mesh Node Locations in X-direction",
        "unit": "mm",
        "category": "Mesh / Nodes",
        "simple_explanation": "Reference mesh coordinates along slab length used by the solver grid.",
        "typical_range": "0 to slab length",
        "example": "[0, 350, 700, ..., 3500]",
        "icon": "🧭",
        "default": [0.0, 350.0, 700.0, 1050.0, 1400.0, 1750.0, 2100.0, 2450.0, 2800.0, 3150.0, 3500.0],
    },
    "y": {
        "name": "Node Coordinates (Y)",
        "technical_name": "Mesh Node Locations in Y-direction",
        "unit": "mm",
        "category": "Mesh / Nodes",
        "simple_explanation": "Reference mesh coordinates across slab width used by the solver grid.",
        "typical_range": "0 to slab width",
        "example": "[0, 350, 700, ..., 3500]",
        "icon": "🧭",
        "default": [0.0, 350.0, 700.0, 1050.0, 1400.0, 1750.0, 2100.0, 2450.0, 2800.0, 3150.0, 3500.0],
    },
}

PARAM_INFO = {
    "Emod": {
        "name": "Concrete Stiffness",
        "technical_name": "Modulus of Elasticity",
        "unit": "MPa",
        "alt_units": {"gpa": 1000.0},
        "category": "Material Properties",
        "simple_explanation": "How stiff or rigid the concrete is. Higher values mean stiffer concrete that bends less under load.",
        "typical_range": "20,000 - 35,000 MPa for concrete pavements",
        "example": "For normal concrete roads, use around 24,000-30,000 MPa",
        "icon": "🧱",
    },
    "nu": {
        "name": "Concrete Flexibility Ratio",
        "technical_name": "Poisson's Ratio",
        "unit": "(no unit)",
        "category": "Material Properties",
        "simple_explanation": "How much the concrete squeezes sideways when pressed down. It's always between 0 and 0.5.",
        "typical_range": "0.15 – 0.25 for concrete (IS 456); typical pavement: 0.15 – 0.20",
        "example": "For most concrete, use 0.18 as a standard value",
        "icon": "📐",
    },
    "a": {
        "name": "Slab Length",
        "technical_name": "Slab Dimension (X-direction)",
        "unit": "mm",
        "alt_units": {
            "m": 1000,
            "meter": 1000,
            "meters": 1000,
            "cm": 10,
            "ft": 304.8,
            "feet": 304.8,
            "inch": 25.4,
            "inches": 25.4,
        },
        "category": "Slab Size",
        "simple_explanation": "The length of your concrete slab (one side of the rectangle).",
        "typical_range": "3,000 - 6,000 mm (3 to 6 meters)",
        "example": "A typical road slab is about 4.5 meters (4500 mm) long",
        "icon": "📏",
    },
    "b": {
        "name": "Slab Width",
        "technical_name": "Slab Dimension (Y-direction)",
        "unit": "mm",
        "alt_units": {
            "m": 1000,
            "meter": 1000,
            "meters": 1000,
            "cm": 10,
            "ft": 304.8,
            "feet": 304.8,
            "inch": 25.4,
            "inches": 25.4,
        },
        "category": "Slab Size",
        "simple_explanation": "The width of your concrete slab (the other side of the rectangle).",
        "typical_range": "3,000 - 4,500 mm (3 to 4.5 meters)",
        "example": "A typical road lane is about 3.5 meters (3500 mm) wide",
        "icon": "📐",
    },
    "t": {
        "name": "Slab Thickness",
        "technical_name": "Thickness of Slab",
        "unit": "mm",
        "alt_units": {"m": 1000, "meter": 1000, "meters": 1000, "cm": 10, "inch": 25.4, "inches": 25.4},
        "category": "Slab Size",
        "simple_explanation": "How thick the concrete slab is (from top to bottom).",
        "typical_range": "150 - 300 mm",
        "example": "Normal roads use about 200-250 mm thickness",
        "icon": "📊",
    },
    "Kx": {
        "name": "Ground Support (Horizontal)",
        "technical_name": "Foundation Stiffness (X)",
        "unit": "(stiffness factor)",
        "category": "Ground Support",
        "simple_explanation": "How well the ground underneath supports the slab horizontally. Higher means stronger support.",
        "typical_range": "30 - 100",
        "example": "Soft soil: 30-50, Medium soil: 50-70, Hard soil: 70-100",
        "icon": "🌍",
    },
    "Ky": {
        "name": "Ground Support (Sideways)",
        "technical_name": "Foundation Stiffness (Y)",
        "unit": "(stiffness factor)",
        "category": "Ground Support",
        "simple_explanation": "How well the ground underneath supports the slab sideways.",
        "typical_range": "30 - 100",
        "example": "Usually same as horizontal support (Kx)",
        "icon": "🌍",
    },
    "Kz": {
        "name": "Ground Support (Vertical)",
        "technical_name": "Foundation Stiffness (Z)",
        "unit": "(stiffness factor)",
        "category": "Ground Support",
        "simple_explanation": "How well the ground underneath pushes back when the slab is pressed down.",
        "typical_range": "30 - 100",
        "example": "Usually same as other directions",
        "icon": "🌍",
    },
    "x1": {
        "name": "Load Start (Length-wise)",
        "technical_name": "Load Start Position (X)",
        "unit": "mm",
        "alt_units": {"m": 1000, "meter": 1000, "meters": 1000, "cm": 10},
        "category": "Load Position",
        "simple_explanation": "Where the tire load starts along the length of the slab.",
        "typical_range": "Depends on slab size",
        "example": "If analyzing a tire in the middle, start at about 1/3 of slab length",
        "icon": "🚗",
    },
    "x2": {
        "name": "Load End (Length-wise)",
        "technical_name": "Load End Position (X)",
        "unit": "mm",
        "alt_units": {"m": 1000, "meter": 1000, "meters": 1000, "cm": 10},
        "category": "Load Position",
        "simple_explanation": "Where the tire load ends along the length of the slab.",
        "typical_range": "Must be greater than x1",
        "example": "Typically x1 + tire contact length (about 200-400 mm)",
        "icon": "🚗",
    },
    "y1": {
        "name": "Load Start (Width-wise)",
        "technical_name": "Load Start Position (Y)",
        "unit": "mm",
        "alt_units": {"m": 1000, "meter": 1000, "meters": 1000, "cm": 10},
        "category": "Load Position",
        "simple_explanation": "Where the tire load starts across the width of the slab.",
        "typical_range": "Depends on slab size",
        "example": "For edge loading, start near the edge",
        "icon": "🚗",
    },
    "y2": {
        "name": "Load End (Width-wise)",
        "technical_name": "Load End Position (Y)",
        "unit": "mm",
        "alt_units": {"m": 1000, "meter": 1000, "meters": 1000, "cm": 10},
        "category": "Load Position",
        "simple_explanation": "Where the tire load ends across the width of the slab.",
        "typical_range": "Must be greater than y1",
        "example": "Typically y1 + tire width (about 200-300 mm)",
        "icon": "🚗",
    },
    "mesh_type": {
        "name": "Mesh Density",
        "technical_name": "Finite Element Mesh Classification",
        "unit": "(no unit)",
        "category": "Mesh / Nodes",
        "simple_explanation": "Choose how dense the solver grid should be: coarse, medium, or fine.",
        "typical_range": "coarse / medium / fine",
        "example": "medium",
        "icon": "🕸️",
    },
    "load_location": {
        "name": "Load Location Type",
        "technical_name": "Load Placement Class",
        "unit": "(no unit)",
        "category": "Load Position",
        "simple_explanation": "Choose where the load acts on the slab: at corner, at edge, or interior.",
        "typical_range": "corner / edge / interior",
        "example": "corner",
        "icon": "📍",
    },
    "q": {
        "name": "Tire Pressure on Ground",
        "technical_name": "Tyre Contact Pressure",
        "unit": "MPa",
        "alt_units": {"kpa": 0.001, "psi": 0.00689476},
        "category": "Load",
        "simple_explanation": "How hard the tire presses on the concrete. This depends on vehicle weight and tire size.",
        "typical_range": "Varies by aircraft/vehicle load and tire contact area",
        "example": "For 200 kN on 400 mm x 400 mm, q = 1.25 MPa",
        "icon": "🛞",
    },
}

PARAM_INFO.update(NODE_PARAM_INFO)

NODE_PARAM_KEYS = {"x", "y"}
MESH_TYPE_KEY = "mesh_type"
LOAD_LOCATION_KEY = "load_location"

MESH_LEVEL_ELEMENTS = {
    "coarse": 10,
    "medium": 15,
    "fine": 30,
}

STANDARD_POISSON_RATIO = 0.18


_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hii",
    "hiii",
    "good morning",
    "good afternoon",
    "good evening",
}

_THANKS = {
    "thanks",
    "thank you",
    "thx",
    "ty",
    "appreciate it",
}

_GOODBYES = {
    "bye",
    "goodbye",
    "see you",
    "see ya",
}

_HELP_PATTERNS = {
    "help",
    "how do i use",
    "how to use",
    "what can you do",
    "what do you do",
    "commands",
    "examples",
    "sample",
}

_EXPLAIN_PATTERNS = {
    "explain",
    "meaning",
    "what is",
    "what's",
    "define",
    "units",
    "unit",
    "why",
    "how",
}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _contains_default_keyword(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return bool(re.search(r"\bdefaults?\b", normalized))


def _looks_like_greeting(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if normalized in _GREETINGS:
        return True
    # Handle quick greeting followed by name etc.
    first = normalized.split(" ", 1)[0]
    return first in {"hi", "hello", "hey"} and len(normalized) <= 20


def _looks_like_thanks(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(phrase in normalized for phrase in _THANKS)


def _looks_like_goodbye(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(phrase in normalized for phrase in _GOODBYES)


def _looks_like_help_request(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(p in normalized for p in _HELP_PATTERNS)


def _looks_like_explanation_request(text: str) -> bool:
    normalized = _normalize_text(text)
    if "?" in (text or ""):
        return True
    return any(p in normalized for p in _EXPLAIN_PATTERNS)


def _extract_param_key_from_text(text: str) -> Optional[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return None

    # Direct key mentions.
    for key in PARAM_INFO.keys():
        if f" {key.lower()} " in f" {normalized} ":
            return key

    # Common phrases.
    if "poisson" in normalized or "poissons" in normalized:
        return "nu"
    if "mesh" in normalized:
        return "mesh_type"
    if "load location" in normalized or "loading location" in normalized or "where is the load" in normalized:
        return "load_location"
    if "thickness" in normalized or "thick" in normalized:
        return "t"
    if "length" in normalized and "slab" in normalized:
        return "a"
    if "width" in normalized and "slab" in normalized:
        return "b"
    if "pressure" in normalized or "contact pressure" in normalized:
        return "q"
    if "subgrade" in normalized or "foundation" in normalized:
        # If user is generic, explain Kz first.
        if "kx" in normalized:
            return "Kx"
        if "ky" in normalized:
            return "Ky"
        if "kz" in normalized:
            return "Kz"
        return "Kz"

    # Alias matching.
    for key, aliases in PARAM_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                return key

    # Name / technical name token matching.
    for key, info in PARAM_INFO.items():
        name = _normalize_text(str(info.get("name", "")))
        tech = _normalize_text(str(info.get("technical_name", "")))
        if name and name in normalized:
            return key
        if tech and tech in normalized:
            return key
    return None


def _parameter_explanation(key: str) -> str:
    info = PARAM_INFO.get(key)
    if not info:
        return ""

    title = f"{info.get('icon', 'ℹ️')} **{info.get('name', key)}**"
    tech = info.get("technical_name")
    unit = info.get("unit")
    pieces = [title]
    if tech:
        pieces.append(f"Technical name: {tech}")
    if unit and unit != "(no unit)":
        pieces.append(f"Unit: {unit}")
    simple = info.get("simple_explanation")
    if simple:
        pieces.append(f"In simple words: {simple}")
    typical = info.get("typical_range")
    if typical:
        pieces.append(f"Typical range: {typical}")
    example = info.get("example")
    if example:
        pieces.append(f"Example: {example}")
    pieces.append(f"Default we can use: {format_value_with_unit(get_default_value(key), key)}")
    return "\n".join(pieces)


def handle_conversational_intent(
    user_input: str,
    params: Dict[str, Any],
    current_asking: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Deterministic conversation handler.

    Returns a result dict matching process_user_input_with_llm output when the input
    is clearly conversational (greeting/help/explain), so we don't depend on the LLM.
    """

    normalized = _normalize_text(user_input)
    if not normalized:
        return {
            "understood_value": None,
            "use_default": False,
            "use_all_defaults": False,
            "needs_clarification": False,
            "friendly_response": "I’m here whenever you’re ready. Tell me about the slab and loading, or say ‘let’s begin’ for guided setup.",
            "original_unit": None,
            "parameter_key": current_asking,
            "extracted_multiple": {},
            "conversation_only": True,
        }

    if _looks_like_greeting(normalized):
        msg = (
            "Hi! I can help you set up a concrete slab / rigid pavement analysis. "
            "If you describe your slab and loading in one message, I’ll extract the values.\n\n"
            "Examples you can paste:\n"
            "- ‘5m x 5m slab, 250mm thick, M30 concrete, K=60 MPa/m, 40 kN wheel load at edge, 200x300mm contact, mesh medium’\n"
            "- Or say ‘let’s begin’ and I’ll guide you step-by-step."
        )
        return {
            "understood_value": None,
            "use_default": False,
            "use_all_defaults": False,
            "needs_clarification": False,
            "friendly_response": msg,
            "original_unit": None,
            "parameter_key": current_asking,
            "extracted_multiple": {},
            "conversation_only": True,
        }

    if _looks_like_thanks(normalized):
        return {
            "understood_value": None,
            "use_default": False,
            "use_all_defaults": False,
            "needs_clarification": False,
            "friendly_response": "You’re welcome. If you want, share the slab dimensions + loading and I’ll take it from there.",
            "original_unit": None,
            "parameter_key": current_asking,
            "extracted_multiple": {},
            "conversation_only": True,
        }

    if _looks_like_goodbye(normalized):
        return {
            "understood_value": None,
            "use_default": False,
            "use_all_defaults": False,
            "needs_clarification": False,
            "friendly_response": "Bye! Come back anytime—just describe your slab and loading when you’re ready.",
            "original_unit": None,
            "parameter_key": current_asking,
            "extracted_multiple": {},
            "conversation_only": True,
        }

    if _looks_like_help_request(normalized):
        return {
            "understood_value": None,
            "use_default": False,
            "use_all_defaults": False,
            "needs_clarification": False,
            "friendly_response": (
                "I can help with slab/pavement analysis inputs (dimensions, thickness, concrete grade, subgrade K, mesh, load location, wheel load/contact area).\n\n"
                "You can either:\n"
                "- Describe your case in plain text (best), or\n"
                "- Say ‘let’s begin’ for a guided step-by-step flow, or\n"
                "- Ask ‘explain Emod’ / ‘what is Kz?’ / ‘what does mesh mean?’ and I’ll explain." 
            ),
            "original_unit": None,
            "parameter_key": current_asking,
            "extracted_multiple": {},
            "conversation_only": True,
        }

    # Parameter explanation / "what does X mean" type requests.
    if _looks_like_explanation_request(user_input) and not re.search(r"\d", user_input or ""):
        # If user asks "what is this" while we're prompting for a specific field, explain that field.
        key = _extract_param_key_from_text(user_input) or current_asking
        if key and key in PARAM_INFO:
            follow = ""
            if current_asking and key == current_asking:
                follow = "\n\nWhen you’re ready, you can reply with a value (or say ‘default’)."
            else:
                follow = "\n\nIf you share your slab + loading details, I’ll capture the values for you."
            return {
                "understood_value": None,
                "use_default": False,
                "use_all_defaults": False,
                "needs_clarification": False,
                "friendly_response": f"{_parameter_explanation(key)}{follow}",
                "original_unit": None,
                "parameter_key": current_asking,
                "extracted_multiple": {},
                "conversation_only": True,
            }

        if any(phrase in normalized for phrase in {"parameters", "inputs", "what are the parameters", "what inputs"}):
            categories = []
            for category, keys in PARAM_CATEGORIES.items():
                pretty = ", ".join(PARAM_INFO[k]["name"] for k in keys if k in PARAM_INFO)
                categories.append(f"- {category}: {pretty}")
            return {
                "understood_value": None,
                "use_default": False,
                "use_all_defaults": False,
                "needs_clarification": False,
                "friendly_response": "Here are the main inputs I work with:\n" + "\n".join(categories),
                "original_unit": None,
                "parameter_key": current_asking,
                "extracted_multiple": {},
                "conversation_only": True,
            }

    # Polite refusal for clearly off-topic smalltalk.
    if _looks_like_explanation_request(user_input) and any(
        phrase in normalized
        for phrase in {
            "weather",
            "time in",
            "cricket score",
            "stock",
            "bitcoin",
            "movie",
            "song",
        }
    ):
        return {
            "understood_value": None,
            "use_default": False,
            "use_all_defaults": False,
            "needs_clarification": False,
            "friendly_response": "I can’t help with that topic, but I can help with slab/pavement analysis inputs and explanations. Tell me your slab size + loading, or ask what a parameter means.",
            "original_unit": None,
            "parameter_key": current_asking,
            "extracted_multiple": {},
            "conversation_only": True,
        }

    return None


def get_default_value(key: str) -> Any:
    if key in NODE_PARAM_KEYS:
        return list(NODE_DEFAULT_VALUES[key])
    return DEFAULT_VALUES[key]

PARAM_CATEGORIES = {
    "Mesh / Nodes": ["mesh_type", "x", "y"],
    "Slab Size": ["a", "b", "t"],
    "Material Properties": ["Emod", "nu"],
    "Ground Support": ["Kx", "Ky", "Kz"],
    "Load Position": ["load_location", "x1", "x2", "y1", "y2"],
    "Load": ["q"],
}

PARAM_ORDER = ["a", "b", "mesh_type", "t", "Emod", "nu", "Kx", "Ky", "Kz", "load_location", "q"]

LLM_INVOKE_TIMEOUT_SECONDS = float(os.getenv("LLM_INVOKE_TIMEOUT_SECONDS", "45.0"))
LLM_NARRATION_TIMEOUT_SECONDS = float(os.getenv("LLM_NARRATION_TIMEOUT_SECONDS", "25.0"))

SINGLE_DEFAULT_PATTERNS = {
    "skip",
    "ok",
    "okay",
    "yes",
    "sure",
}

PARAM_ALIASES = {
    "a": ["a", "length", "slab length", "x length", "x dimension", "slab dimensions", "slab size"],
    "b": ["b", "width", "slab width", "y length", "y dimension", "slab dimensions", "slab size"],
    "t": ["t", "thickness", "slab thickness", "depth"],
    "Emod": [
        "emod",
        "e",
        "youngs modulus",
        "young's modulus",
        "modulus",
        "modulus of elasticity",
        "elasticity",
        "concrete grade",
        "compressive strength",
    ],
    "nu": ["nu", "poisson", "poissons ratio", "poisson ratio"],
    "Kx": ["kx", "stiffness x", "foundation x", "subgrade modulus", "foundation modulus"],
    "Ky": ["ky", "stiffness y", "foundation y", "subgrade modulus", "foundation modulus"],
    "Kz": ["kz", "stiffness z", "foundation z", "subgrade modulus", "foundation modulus"],
    "x1": ["x1", "load start x", "start x"],
    "x2": ["x2", "load end x", "end x"],
    "y1": ["y1", "load start y", "start y"],
    "y2": ["y2", "load end y", "end y"],
    "q": ["q", "pressure", "tyre pressure", "tire pressure", "contact pressure"],
    "x": ["node x", "x nodes", "x coordinates", "mesh x", "grid x"],
    "y": ["node y", "y nodes", "y coordinates", "mesh y", "grid y"],
}


def normalize_mesh_type(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in {"1", "coarse", "coarser", "low", "low mesh", "low density"}:
        return "coarse"
    if text in {"2", "medium", "med", "mid", "moderate"}:
        return "medium"
    if text in {"3", "fine", "finer", "high", "high mesh", "high density"}:
        return "fine"
    return None


def normalize_load_location(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in {"1", "corner", "at corner", "on corner", "corner load"}:
        return "corner"
    if text in {"2", "edge", "at edge", "on edge", "edge load", "near edge"}:
        return "edge"
    if text in {"3", "interior", "inside", "middle", "center", "centre", "at interior"}:
        return "interior"
    if "corner" in text:
        return "corner"
    if "edge" in text:
        return "edge"
    if any(word in text for word in ["interior", "inside", "middle", "center", "centre"]):
        return "interior"
    return None


def _clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        return (lower + upper) / 2.0
    return min(max(float(value), float(lower)), float(upper))


def _resolve_reference_location_token(load_location: Optional[str], context_text: Optional[str] = None) -> str:
    raw = f"{load_location or ''} {context_text or ''}".strip().lower()

    is_corner = "corner" in raw
    is_edge = "edge" in raw
    is_interior = any(word in raw for word in ["interior", "inside", "middle", "center", "centre"])

    # Corner variants by explicit symbolic coordinates.
    if is_corner:
        has_x0 = bool(re.search(r"x\s*=\s*0", raw))
        has_xl = bool(re.search(r"x\s*=\s*l\b", raw))
        has_y0 = bool(re.search(r"y\s*=\s*0", raw))
        has_yb = bool(re.search(r"y\s*=\s*b\b", raw))

        if has_xl and has_yb:
            return "corner_xL_yB"
        if has_x0 and has_yb:
            return "corner_x0_yB"
        if has_xl and has_y0:
            return "corner_xL_y0"
        if has_x0 and has_y0:
            return "corner_x0_y0"

        # Textual corner qualifiers.
        right = "right" in raw
        left = "left" in raw
        top = "top" in raw or "upper" in raw
        bottom = "bottom" in raw or "lower" in raw

        if right and top:
            return "corner_xL_yB"
        if left and top:
            return "corner_x0_yB"
        if right and bottom:
            return "corner_xL_y0"
        return "corner_x0_y0"

    # Edge variants by explicit symbolic coordinates.
    if is_edge:
        if re.search(r"edge\s*x\s*=\s*0|x\s*=\s*0\s*edge", raw) or "left edge" in raw:
            return "edge_x0"
        if re.search(r"edge\s*x\s*=\s*l\b|x\s*=\s*l\b\s*edge", raw) or "right edge" in raw:
            return "edge_xL"
        if re.search(r"edge\s*y\s*=\s*0|y\s*=\s*0\s*edge", raw) or "bottom edge" in raw:
            return "edge_y0"
        if re.search(r"edge\s*y\s*=\s*b\b|y\s*=\s*b\b\s*edge", raw) or "top edge" in raw:
            return "edge_yB"
        # Default edge interpretation when side is not specified.
        return "edge_y0"

    if is_interior:
        return "interior"

    normalized = normalize_load_location(load_location or "") or "interior"
    if normalized == "corner":
        return "corner_x0_y0"
    if normalized == "edge":
        return "edge_y0"
    return "interior"


def _reference_point_from_token(token: str, slab_a: float, slab_b: float) -> Tuple[float, float]:
    if token == "corner_x0_y0":
        return 0.0, 0.0
    if token == "corner_xL_y0":
        return float(slab_a), 0.0
    if token == "corner_x0_yB":
        return 0.0, float(slab_b)
    if token == "corner_xL_yB":
        return float(slab_a), float(slab_b)
    if token == "edge_x0":
        return 0.0, float(slab_b) / 2.0
    if token == "edge_xL":
        return float(slab_a), float(slab_b) / 2.0
    if token == "edge_y0":
        return float(slab_a) / 2.0, 0.0
    if token == "edge_yB":
        return float(slab_a) / 2.0, float(slab_b)
    return float(slab_a) / 2.0, float(slab_b) / 2.0


def _first_wheel_center_from_token(
    token: str,
    slab_a: float,
    slab_b: float,
    contact_lx_mm: float,
    contact_ly_mm: float,
) -> Tuple[float, float]:
    half_x = float(contact_lx_mm) / 2.0
    half_y = float(contact_ly_mm) / 2.0

    if token == "corner_x0_y0":
        return half_x, half_y
    if token == "corner_xL_y0":
        return float(slab_a) - half_x, half_y
    if token == "corner_x0_yB":
        return half_x, float(slab_b) - half_y
    if token == "corner_xL_yB":
        return float(slab_a) - half_x, float(slab_b) - half_y
    if token == "edge_x0":
        return half_x, float(slab_b) / 2.0
    if token == "edge_xL":
        return float(slab_a) - half_x, float(slab_b) / 2.0
    if token == "edge_y0":
        return float(slab_a) / 2.0, half_y
    if token == "edge_yB":
        return float(slab_a) / 2.0, float(slab_b) - half_y
    return float(slab_a) / 2.0, float(slab_b) / 2.0


def _dual_spacing_direction(token: str, rx: float, ry: float, slab_a: float, slab_b: float) -> Tuple[float, float]:
    if token.startswith("corner"):
        vx = (float(slab_a) / 2.0) - float(rx)
        vy = (float(slab_b) / 2.0) - float(ry)
        norm = math.hypot(vx, vy)
        if norm <= 1e-12:
            return 1.0, 0.0
        return vx / norm, vy / norm
    if token in {"edge_x0", "edge_xL"}:
        # Parallel to vertical edge -> spacing along y.
        return 0.0, 1.0
    if token in {"edge_y0", "edge_yB"}:
        # Parallel to horizontal edge -> spacing along x.
        return 1.0, 0.0
    # Interior/center default: longitudinal along slab length (x).
    return 1.0, 0.0


def _clamp_center_to_slab(
    center_x: float,
    center_y: float,
    slab_a: float,
    slab_b: float,
    contact_lx_mm: float,
    contact_ly_mm: float,
) -> Tuple[float, float]:
    half_x = float(contact_lx_mm) / 2.0
    half_y = float(contact_ly_mm) / 2.0
    min_cx = half_x
    max_cx = float(slab_a) - half_x
    min_cy = half_y
    max_cy = float(slab_b) - half_y

    if min_cx > max_cx:
        clamped_x = float(slab_a) / 2.0
    else:
        clamped_x = _clamp(center_x, min_cx, max_cx)

    if min_cy > max_cy:
        clamped_y = float(slab_b) / 2.0
    else:
        clamped_y = _clamp(center_y, min_cy, max_cy)

    return clamped_x, clamped_y


def _center_to_bounded_patch(
    center_x: float,
    center_y: float,
    contact_lx_mm: float,
    contact_ly_mm: float,
    slab_a: float,
    slab_b: float,
) -> Dict[str, float]:
    half_x = float(contact_lx_mm) / 2.0
    half_y = float(contact_ly_mm) / 2.0
    return {
        "x1": round(max(0.0, center_x - half_x), 3),
        "x2": round(min(float(slab_a), center_x + half_x), 3),
        "y1": round(max(0.0, center_y - half_y), 3),
        "y2": round(min(float(slab_b), center_y + half_y), 3),
    }


def infer_semantic_choices(user_input: str, current_asking: Optional[str] = None) -> Dict[str, str]:
    lowered = user_input.lower()
    extracted: Dict[str, str] = {}

    mesh = normalize_mesh_type(user_input)
    if mesh:
        extracted[MESH_TYPE_KEY] = mesh
    elif any(word in lowered for word in ["coarse", "medium", "fine"]):
        if "coarse" in lowered:
            extracted[MESH_TYPE_KEY] = "coarse"
        elif "medium" in lowered:
            extracted[MESH_TYPE_KEY] = "medium"
        elif "fine" in lowered:
            extracted[MESH_TYPE_KEY] = "fine"

    load_loc = normalize_load_location(user_input)
    if load_loc:
        extracted[LOAD_LOCATION_KEY] = load_loc
    elif re.search(r"\bboth\s+at\s+interior\b|\bat\s+interior\b", lowered):
        extracted[LOAD_LOCATION_KEY] = "interior"
    elif re.search(r"\bboth\s+at\s+edge\b|\bat\s+edge\b|\bedge\s+loading\b", lowered):
        extracted[LOAD_LOCATION_KEY] = "edge"
    elif re.search(r"\bboth\s+at\s+corner\b|\bat\s+corner\b|\bcorner\s+loading\b", lowered):
        extracted[LOAD_LOCATION_KEY] = "corner"
    elif "corner" in lowered:
        extracted[LOAD_LOCATION_KEY] = "corner"
    elif any(word in lowered for word in ["interior", "inside", "middle", "center", "centre"]):
        extracted[LOAD_LOCATION_KEY] = "interior"
    elif "edge" in lowered:
        extracted[LOAD_LOCATION_KEY] = "edge"

    if current_asking == MESH_TYPE_KEY and MESH_TYPE_KEY not in extracted:
        keyed = normalize_mesh_type(user_input)
        if keyed:
            extracted[MESH_TYPE_KEY] = keyed
    if current_asking == LOAD_LOCATION_KEY and LOAD_LOCATION_KEY not in extracted:
        keyed = normalize_load_location(user_input)
        if keyed:
            extracted[LOAD_LOCATION_KEY] = keyed

    return extracted


def _linspace_nodes(span_mm: float, elements: int) -> List[float]:
    if elements <= 0:
        return [0.0, float(span_mm)]
    values = [round((float(span_mm) * i) / elements, 3) for i in range(elements + 1)]
    values[0] = 0.0
    values[-1] = float(span_mm)
    return values


def infer_nodes_from_mesh(mesh_type: str, a: float, b: float) -> Tuple[List[float], List[float]]:
    level = normalize_mesh_type(mesh_type) or "coarse"
    elements = MESH_LEVEL_ELEMENTS[level]
    return _linspace_nodes(a, elements), _linspace_nodes(b, elements)


def infer_load_patch_from_location(load_location: str, a: float, b: float) -> Dict[str, float]:
    location = normalize_load_location(load_location) or "interior"
    slab_a = float(a)
    slab_b = float(b)

    patch_x = max(250.0, min(1000.0, round(0.2 * slab_a, 3)))
    patch_y = max(250.0, min(1000.0, round(0.2 * slab_b, 3)))

    if location == "corner":
        return {
            "x1": 0.0,
            "x2": min(patch_x, slab_a),
            "y1": 0.0,
            "y2": min(patch_y, slab_b),
        }

    if location == "edge":
        x1 = max(0.0, round((slab_a - patch_x) / 2.0, 3))
        x2 = min(slab_a, round(x1 + patch_x, 3))
        return {
            "x1": x1,
            "x2": x2,
            "y1": 0.0,
            "y2": min(patch_y, slab_b),
        }

    start_x = min(10.0, max(1.0, round(0.02 * slab_a, 3)))
    start_y = min(10.0, max(1.0, round(0.02 * slab_b, 3)))
    end_margin_x = min(100.0, max(start_x + 1.0, round(0.1 * slab_a, 3)))
    end_margin_y = min(100.0, max(start_y + 1.0, round(0.1 * slab_b, 3)))
    x2 = max(start_x + 1.0, round(slab_a - end_margin_x, 3))
    y2 = max(start_y + 1.0, round(slab_b - end_margin_y, 3))

    return {
        "x1": round(start_x, 3),
        "x2": min(slab_a, x2),
        "y1": round(start_y, 3),
        "y2": min(slab_b, y2),
    }


def infer_load_patch_from_contact_area(
    load_location: str,
    a: float,
    b: float,
    contact_lx_mm: float,
    contact_ly_mm: float,
    context_text: Optional[str] = None,
) -> Dict[str, float]:
    slab_a = float(a)
    slab_b = float(b)
    lx = max(1.0, float(contact_lx_mm))
    ly = max(1.0, float(contact_ly_mm))
    location_token = _resolve_reference_location_token(load_location, context_text)
    cx, cy = _first_wheel_center_from_token(location_token, slab_a, slab_b, lx, ly)
    cx, cy = _clamp_center_to_slab(cx, cy, slab_a, slab_b, lx, ly)
    return _center_to_bounded_patch(cx, cy, lx, ly, slab_a, slab_b)


def _extract_spacing_mm(text: str) -> Optional[float]:
    patterns = [
        r"(?:spacing|wheel\s*spacing|axle\s*spacing|c\s*/\s*c|center\s*to\s*center|centre\s*to\s*centre)\s*(?:of|=|:|is)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)",
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)\s*(?:spacing|c\s*/\s*c|center\s*to\s*center|centre\s*to\s*centre)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        return _length_to_mm(float(match.group(1)), match.group(2))
    return None


def _extract_wheel_spacing_mm(text: str) -> Optional[float]:
    patterns = [
        r"(?:wheel\s*spacing|dual\s*wheel\s*spacing|center\s*to\s*center|centre\s*to\s*centre|c\s*/\s*c)\s*(?:of|=|:|is)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)",
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)\s*(?:wheel\s*spacing|dual\s*wheel\s*spacing|center\s*to\s*center|centre\s*to\s*centre|c\s*/\s*c)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _length_to_mm(float(match.group(1)), match.group(2))
    return None


def _extract_axle_spacing_mm(text: str) -> Optional[float]:
    patterns = [
        r"(?:axle\s*spacing|spacing\s*between\s*axles|tandem\s*spacing)\s*(?:of|=|:|is)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)",
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)\s*(?:axle\s*spacing|spacing\s*between\s*axles|tandem\s*spacing)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _length_to_mm(float(match.group(1)), match.group(2))
    return None


def _offset_load_patch(
    patch: Dict[str, float], spacing_mm: float, slab_a: float, slab_b: float
) -> Dict[str, float]:
    x1 = float(patch["x1"])
    x2 = float(patch["x2"])
    y1 = float(patch["y1"])
    y2 = float(patch["y2"])
    spacing = float(spacing_mm)

    # Prefer offset along slab length (x-direction), then fallback to width.
    if x2 + spacing <= slab_a:
        return {"x1": round(x1 + spacing, 3), "x2": round(x2 + spacing, 3), "y1": y1, "y2": y2}
    if x1 - spacing >= 0.0:
        return {"x1": round(x1 - spacing, 3), "x2": round(x2 - spacing, 3), "y1": y1, "y2": y2}
    if y2 + spacing <= slab_b:
        return {"x1": x1, "x2": x2, "y1": round(y1 + spacing, 3), "y2": round(y2 + spacing, 3)}
    if y1 - spacing >= 0.0:
        return {"x1": x1, "x2": x2, "y1": round(y1 - spacing, 3), "y2": round(y2 - spacing, 3)}

    return {"x1": x1, "x2": x2, "y1": y1, "y2": y2}


def _detect_number_of_wheels(text: str) -> int:
    lowered = text.lower()
    if re.search(r"\btridem\b|\bthree\s*wheels\b|\b3\s*wheels\b", lowered):
        return 3
    if re.search(r"\btandem\s*axle\b|\btandem\b|\bdual\s*wheel\b|\btwo\s*wheels\b|\b2\s*wheels\b", lowered):
        return 2
    return 1


def _extract_edge_distances_mm(text: str) -> Dict[str, float]:
    distances: Dict[str, float] = {}
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)\s*from\s*(left|right|top|bottom)\s*edge"
    )
    for match in pattern.finditer(text):
        value_mm = _length_to_mm(float(match.group(1)), match.group(2))
        side = match.group(3).lower()
        distances[side] = value_mm
    return distances


def _center_to_patch(center_x: float, center_y: float, contact_lx_mm: float, contact_ly_mm: float) -> Dict[str, float]:
    half_x = float(contact_lx_mm) / 2.0
    half_y = float(contact_ly_mm) / 2.0
    return {
        "x1": round(center_x - half_x, 3),
        "x2": round(center_x + half_x, 3),
        "y1": round(center_y - half_y, 3),
        "y2": round(center_y + half_y, 3),
    }


def _nearest_mesh_point(value: float, nodes: List[float]) -> float:
    if not nodes:
        return float(value)
    return float(min(nodes, key=lambda node: abs(float(node) - float(value))))


def _compute_patch_from_description(
    text: str,
    slab_a: float,
    slab_b: float,
    contact_lx_mm: float,
    contact_ly_mm: float,
    load_location: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Derive exact patch coordinates from a natural-language description.

    Priority order:
    1. Explicit edge distances ("1m from left edge, 2m from bottom")
    2. Named location + actual contact dims via infer_load_patch_from_contact_area()
    3. "center of slab" / "middle of slab" shorthand

    Returns None when there is insufficient information (callers fall back to
    infer_load_patch_from_location() which uses the 20% heuristic).
    """
    lowered = text.lower()

    # Priority 1: explicit edge distances
    distances = _extract_edge_distances_mm(text)
    if distances:
        cx = slab_a / 2.0
        cy = slab_b / 2.0
        if "left" in distances:
            cx = float(distances["left"])
        elif "right" in distances:
            cx = slab_a - float(distances["right"])
        if "bottom" in distances:
            cy = float(distances["bottom"])
        elif "top" in distances:
            cy = slab_b - float(distances["top"])
        cx, cy = _clamp_center_to_slab(cx, cy, slab_a, slab_b, contact_lx_mm, contact_ly_mm)
        return _center_to_bounded_patch(cx, cy, contact_lx_mm, contact_ly_mm, slab_a, slab_b)

    # Priority 2: named location
    resolved = normalize_load_location(load_location or "")
    if not resolved:
        resolved = infer_semantic_choices(text).get(LOAD_LOCATION_KEY)
    if resolved:
        return infer_load_patch_from_contact_area(
            resolved, slab_a, slab_b, contact_lx_mm, contact_ly_mm, text
        )

    # Priority 3: "center of slab" / "middle" shorthand
    if re.search(r"\bcenter\s*of\s*slab\b|\bmiddle\s*of\s*slab\b|\bslab\s*center\b", lowered):
        cx, cy = _clamp_center_to_slab(slab_a / 2.0, slab_b / 2.0, slab_a, slab_b, contact_lx_mm, contact_ly_mm)
        return _center_to_bounded_patch(cx, cy, contact_lx_mm, contact_ly_mm, slab_a, slab_b)

    return None


def _build_group_wheel_load_cases(
    number_of_wheels: int,
    spacing_mm: float,
    slab_a: float,
    slab_b: float,
    contact_lx_mm: float,
    contact_ly_mm: float,
    load_location: str,
    text: str,
    q_value: float,
    is_tandem: bool = False,
    wheels_per_axle: int = 1,
    axle_spacing_mm: Optional[float] = None,
) -> List[Dict[str, float]]:
    contact_x = max(1.0, float(contact_lx_mm))
    contact_y = max(1.0, float(contact_ly_mm))
    spacing = max(0.0, float(spacing_mm))

    location_token = _resolve_reference_location_token(load_location, text)
    rx, ry = _reference_point_from_token(location_token, slab_a, slab_b)

    cx1, cy1 = _first_wheel_center_from_token(location_token, slab_a, slab_b, contact_x, contact_y)
    cx1, cy1 = _clamp_center_to_slab(cx1, cy1, slab_a, slab_b, contact_x, contact_y)

    dual_dx, dual_dy = _dual_spacing_direction(location_token, rx, ry, slab_a, slab_b)
    centers: List[Tuple[float, float]] = []

    if is_tandem:
        per_axle = max(1, int(wheels_per_axle))
        axle_spacing = max(0.0, float(axle_spacing_mm if axle_spacing_mm is not None else spacing))

        for axle_idx in range(2):
            axle_cx = cx1 + (axle_idx * axle_spacing)
            axle_cy = cy1
            if per_axle == 1:
                cxi, cyi = _clamp_center_to_slab(axle_cx, axle_cy, slab_a, slab_b, contact_x, contact_y)
                centers.append((cxi, cyi))
                continue

            for wheel_idx in range(per_axle):
                wheel_cx = axle_cx + (wheel_idx * spacing * dual_dx)
                wheel_cy = axle_cy + (wheel_idx * spacing * dual_dy)
                cxi, cyi = _clamp_center_to_slab(wheel_cx, wheel_cy, slab_a, slab_b, contact_x, contact_y)
                centers.append((cxi, cyi))
    else:
        wheels = max(1, int(number_of_wheels))
        for wheel_idx in range(wheels):
            wheel_cx = cx1 + (wheel_idx * spacing * dual_dx)
            wheel_cy = cy1 + (wheel_idx * spacing * dual_dy)
            cxi, cyi = _clamp_center_to_slab(wheel_cx, wheel_cy, slab_a, slab_b, contact_x, contact_y)
            centers.append((cxi, cyi))

    if not centers:
        centers = [(cx1, cy1)]

    expected_count = max(1, int(number_of_wheels))
    if len(centers) > expected_count:
        centers = centers[:expected_count]

    # Geometry validation: ensure center spacing is preserved when feasible.
    # Corner/edge constraints can force clamping; this check is tolerant and non-fatal.
    spacing_tolerance_mm = 1.0
    if is_tandem:
        per_axle = max(1, int(wheels_per_axle))
        if per_axle >= 2 and spacing > 0.0:
            for axle_idx in range(2):
                start = axle_idx * per_axle
                if start + 1 < len(centers):
                    p1 = centers[start]
                    p2 = centers[start + 1]
                    _ = abs(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) - spacing) <= spacing_tolerance_mm
    elif len(centers) >= 2 and spacing > 0.0:
        p1 = centers[0]
        p2 = centers[1]
        _ = abs(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) - spacing) <= spacing_tolerance_mm

    load_cases: List[Dict[str, float]] = []
    for center_x, center_y in centers:
        patch = _center_to_bounded_patch(center_x, center_y, contact_x, contact_y, slab_a, slab_b)
        load_cases.append(
            {
                "x1": float(patch["x1"]),
                "x2": float(patch["x2"]),
                "y1": float(patch["y1"]),
                "y2": float(patch["y2"]),
                "q": float(q_value),
            }
        )

    return load_cases


def apply_implicit_inferences(params: Dict[str, Any], user_provided_keys: List[str]) -> Tuple[List[str], List[str]]:
    applied: List[str] = []
    notes: List[str] = []
    slab_a = float(params.get("a", DEFAULT_VALUES["a"]))
    slab_b = float(params.get("b", DEFAULT_VALUES["b"]))

    if MESH_TYPE_KEY in params and not any(k in user_provided_keys for k in NODE_PARAM_KEYS):
        mesh_type = normalize_mesh_type(params[MESH_TYPE_KEY]) or "coarse"
        inferred_x, inferred_y = infer_nodes_from_mesh(mesh_type, slab_a, slab_b)
        if params.get("x") != inferred_x:
            params["x"] = inferred_x
            applied.append("x")
        if params.get("y") != inferred_y:
            params["y"] = inferred_y
            applied.append("y")
        if "x" in applied or "y" in applied:
            notes.append(f"Inferred node coordinates from mesh type '{mesh_type}'.")

    load_keys = ["x1", "x2", "y1", "y2"]
    if LOAD_LOCATION_KEY in params and not any(k in user_provided_keys for k in load_keys):
        # Skip inference if multi-wheel load_cases already computed — don't overwrite them.
        if isinstance(params.get("load_cases"), list) and params["load_cases"]:
            pass
        else:
            location = normalize_load_location(params[LOAD_LOCATION_KEY]) or "interior"
            contact_lx = params.get("_contact_lx_mm")
            contact_ly = params.get("_contact_ly_mm")
            if contact_lx is not None and contact_ly is not None:
                inferred_load = infer_load_patch_from_contact_area(
                    location, slab_a, slab_b, float(contact_lx), float(contact_ly)
                )
            else:
                inferred_load = infer_load_patch_from_location(location, slab_a, slab_b)
            changed = False
            for key, value in inferred_load.items():
                if params.get(key) != value:
                    params[key] = value
                    applied.append(key)
                    changed = True
            if changed:
                notes.append(f"Inferred load patch coordinates for '{location}' location.")

    return applied, notes


def convert_to_standard_unit(value: float, from_unit: str, param_key: str) -> Tuple[float, str]:
    info = PARAM_INFO.get(param_key, {})
    alt_units = info.get("alt_units", {})
    standard_unit = info.get("unit", "")
    from_unit_lower = from_unit.lower().strip()
    if from_unit_lower in alt_units:
        multiplier = alt_units[from_unit_lower]
        converted_value = value * multiplier
        return converted_value, f"Converted {value} {from_unit} → {converted_value:.1f} {standard_unit}"
    return value, ""


def format_value_with_unit(value: Any, param_key: str) -> str:
    info = PARAM_INFO.get(param_key, {})
    unit = info.get("unit", "")
    if isinstance(value, list):
        rendered = ", ".join(_compact_value(v) for v in value)
        return f"[{rendered}]" if not unit or unit == "(no unit)" else f"[{rendered}] {unit}"
    if isinstance(value, float) and value == int(value):
        formatted = f"{int(value):,}"
    elif isinstance(value, float):
        formatted = f"{value:,.2f}".rstrip("0").rstrip(".")
    else:
        formatted = str(value)
    if unit and unit != "(no unit)" and unit != "(stiffness factor)":
        return f"{formatted} {unit}"
    return formatted


def validate_single_param(key: str, value: Any, all_params: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a single parameter value."""
    # Private/internal keys (prefixed with _) are computation artefacts; skip validation.
    if key.startswith("_"):
        return True, ""

    info = PARAM_INFO.get(key, {})

    if key in NODE_PARAM_KEYS:
        if not isinstance(value, list) or not value:
            return False, "Please provide at least two node coordinate values, for example [0, 1500, 3500]."
        numeric: List[float] = []
        for item in value:
            if not isinstance(item, (int, float)):
                return False, "Node coordinates must be numeric values."
            numeric.append(float(item))
        if len(numeric) < 2:
            return False, "Please provide at least two node coordinates."
        if any(v < 0 for v in numeric):
            return False, "Node coordinates cannot be negative."
        for i in range(1, len(numeric)):
            if numeric[i] <= numeric[i - 1]:
                return False, "Node coordinates must be in strictly increasing order."
        if key == "x" and "a" in all_params and numeric[-1] > float(all_params["a"]):
            return False, f"Last X node cannot exceed slab length ({all_params['a']} mm)."
        if key == "y" and "b" in all_params and numeric[-1] > float(all_params["b"]):
            return False, f"Last Y node cannot exceed slab width ({all_params['b']} mm)."
        return True, ""

    if key == MESH_TYPE_KEY:
        normalized = normalize_mesh_type(value)
        if normalized is None:
            return False, "Choose one mesh type: coarse, medium, or fine."
        return True, ""

    if key == LOAD_LOCATION_KEY:
        normalized = normalize_load_location(value)
        if normalized is None:
            return False, "Choose one load location type: corner, edge, or interior."
        return True, ""

    if not isinstance(value, (int, float)):
        return False, "Please provide a numeric value."

    if "max_value" in info and value > info["max_value"]:
        return False, f"This value is too high! The maximum allowed is {info['max_value']} {info.get('unit', '')}."
    if key in ["Emod", "a", "b", "t"] and value <= 0:
        return False, "This value must be greater than zero."
    if key in ["Kx", "Ky", "Kz"] and value < 0:
        return False, "This value cannot be negative."
    if key == "nu" and (value < 0.15 or value > 0.25):
        return False, "Poisson's ratio must be between 0.15 and 0.25 for concrete pavements (IS 456). Typical value is 0.15–0.20."
    if key == "x1" and "a" in all_params and value > all_params["a"]:
        return False, f"This can't be larger than your slab length ({all_params['a']} mm)."
    if key == "x2" and "a" in all_params and value > all_params["a"]:
        return False, f"This can't be larger than your slab length ({all_params['a']} mm)."
    if key == "y1" and "b" in all_params and value > all_params["b"]:
        return False, f"This can't be larger than your slab width ({all_params['b']} mm)."
    if key == "y2" and "b" in all_params and value > all_params["b"]:
        return False, f"This can't be larger than your slab width ({all_params['b']} mm)."
    if key == "x2" and "x1" in all_params and value <= all_params["x1"]:
        return False, f"This must be larger than the load start position ({all_params['x1']} mm)."
    if key == "y2" and "y1" in all_params and value <= all_params["y1"]:
        return False, f"This must be larger than the load start position ({all_params['y1']} mm)."
    if key == "q" and value < 0:
        return False, "Pressure can't be negative."
    return True, ""


def find_first_inconsistent_param(params: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Return (param_key, error_msg) for the first parameter that becomes invalid given current state."""
    keys_to_check = list(PARAM_ORDER)
    for extra_key in ["x", "y", "x1", "x2", "y1", "y2"]:
        if extra_key in params and extra_key not in keys_to_check:
            keys_to_check.append(extra_key)

    for key in keys_to_check:
        if key in params:
            ok, msg = validate_single_param(key, params[key], params)
            if not ok:
                return key, msg
    return None


def check_physical_feasibility(params: Dict[str, Any]) -> List[str]:
    """Return soft engineering-feasibility warnings (not hard errors).

    These are passed as inference_notes to the LLM narration so the assistant
    can mention them conversationally without blocking the workflow.
    """
    warnings_list: List[str] = []
    a = float(params.get("a", 0))
    b = float(params.get("b", 0))
    t = float(params.get("t", 0))
    q = float(params.get("q", 0))

    if 0 < a < 1000:
        warnings_list.append(f"Slab length {a:.0f} mm seems very small (typical minimum ~1000 mm).")
    if 0 < b < 1000:
        warnings_list.append(f"Slab width {b:.0f} mm seems very small (typical minimum ~1000 mm).")
    if 0 < t < 100:
        warnings_list.append(f"Slab thickness {t:.0f} mm is too thin (typical minimum 150 mm for pavements).")
    if t > 600:
        warnings_list.append(f"Slab thickness {t:.0f} mm is unusually large — please verify.")
    if 0 < q > 0.7:
        warnings_list.append(
            f"Contact pressure {q:.3f} MPa exceeds 0.7 MPa (typical upper bound for highway tyres). "
            "Please check load value and contact area."
        )
    if q > 2.0:
        warnings_list.append(
            f"Contact pressure {q:.3f} MPa is extremely high — possible unit error "
            "(e.g., kN entered as N, or area in m² instead of mm²)."
        )
    return warnings_list


def get_friendly_param_question(key: str, params: Dict[str, Any]) -> str:
    info = PARAM_INFO[key]

    if key == MESH_TYPE_KEY:
        return (
            "🕸️ **Mesh Density**\n\n"
            "Choose the solver mesh level. I'll infer node coordinates automatically.\n\n"
            "1. coarse\n"
            "2. medium\n"
            "3. fine\n\n"
            "Reply with the name or number (for example: 'medium' or '2')."
        )

    if key == LOAD_LOCATION_KEY:
        return (
            "📍 **Load Location Type**\n\n"
            "Choose where the load is applied. I'll infer x1, x2, y1, y2 automatically.\n\n"
            "1. corner\n"
            "2. edge\n"
            "3. interior\n\n"
            "Reply with the name or number (for example: 'corner' or '1')."
        )

    question = f"{info['icon']} **{info['name']}** ({info['technical_name']})\n\n"
    question += f"📝 *What this means:* {info['simple_explanation']}\n\n"
    question += f"📏 *Typical range:* {info['typical_range']}\n\n"
    question += f"💡 *Example:* {info['example']}\n\n"

    if key in ["x1", "x2"] and "a" in params:
        question += f"ℹ️ *Note: Your slab length is {params['a']} mm, so this should be less than that.*\n\n"
    if key in ["y1", "y2"] and "b" in params:
        question += f"ℹ️ *Note: Your slab width is {params['b']} mm, so this should be less than that.*\n\n"
    if key == "x2" and "x1" in params:
        question += f"ℹ️ *Note: This should be greater than Load Start ({params['x1']} mm).*\n\n"
    if key == "y2" and "y1" in params:
        question += f"ℹ️ *Note: This should be greater than Load Start ({params['y1']} mm).*\n\n"

    question += f"🔧 **Default value:** {format_value_with_unit(get_default_value(key), key)}\n\n"
    question += "👉 *Enter a value, or type 'default' or 'skip' to use the default.*"
    return question


def build_conversation_context(messages: List[Dict], params: Dict[str, Any], memory: Optional[str] = None) -> str:
    context_parts = []
    if memory:
        mem = str(memory).strip()
        if mem:
            # Keep memory bounded in the prompt context to avoid runaway token usage.
            if len(mem) > 6000:
                mem = "..." + mem[-6000:]
            context_parts.append("Earlier conversation memory (compacted):")
            context_parts.append(mem)
    if params:
        context_parts.append("Current parameters collected:")
        for key, value in params.items():
            info = PARAM_INFO.get(key, {})
            context_parts.append(f"- {info.get('name', key)}: {format_value_with_unit(value, key)}")

    # Include as much recent conversation as fits into a safe budget.
    max_recent_chars = 6000
    kept: List[Dict] = []
    used_chars = 0
    for msg in reversed(messages):
        content_raw = str(msg.get("content") or "")
        content = content_raw.strip()
        if len(content) > 400:
            content = content[:400] + "..."

        role_raw = str(msg.get("role") or "")
        role = "User" if role_raw == "user" else ("Assistant" if role_raw == "assistant" else "System")
        line = f"{role}: {content}"

        # +1 for newline join.
        add_len = len(line) + 1
        if kept and used_chars + add_len > max_recent_chars:
            break

        kept.append({"line": line})
        used_chars += add_len

    kept_lines = [item["line"] for item in reversed(kept)]
    if kept_lines:
        context_parts.append("\nConversation (most recent first-fit):")
        context_parts.extend(kept_lines)
    return "\n".join(context_parts)


def _compact_value(value: Any) -> str:
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _extract_first_number_with_unit(text: str) -> Tuple[Optional[float], Optional[str]]:
    s = text.strip()
    n = len(s)

    i = 0
    while i < n and not (s[i].isdigit() or s[i] == "-"):
        i += 1
    if i >= n:
        return None, None

    j = i
    seen_digit = False
    while j < n:
        ch = s[j]
        if ch.isdigit():
            seen_digit = True
            j += 1
            continue
        if ch in {"-", ".", ","}:
            j += 1
            continue
        break

    if not seen_digit:
        return None, None

    raw = s[i:j].replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None, None

    k = j
    while k < n and s[k].isspace():
        k += 1
    u = k
    while u < n and (s[u].isalpha() or s[u] == "%"):
        u += 1

    unit = s[k:u].lower() if u > k else None
    return value, unit


def _tokenize_for_numbers(text: str) -> List[str]:
    cleaned_chars: List[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {".", "-", "%"}:
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    return "".join(cleaned_chars).split()


_KNOWN_UNIT_TOKENS = {
    # length
    "mm",
    "millimeter",
    "millimeters",
    "cm",
    "m",
    "meter",
    "meters",
    "ft",
    "feet",
    "foot",
    "inch",
    "inches",
    # pressure
    "mpa",
    "kpa",
    "psi",
    # stiffness / modulus
    "gpa",
    # force (occasionally present in user text)
    "n",
    "kn",
}


_NUMBER_WITH_UNIT_RE = re.compile(r"^(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)([a-z%]+)?$")


def _parse_number_token_with_unit(token: str) -> Optional[Tuple[float, Optional[str]]]:
    """Parse a token that may contain a number with an attached unit.

    Examples:
    - "200" -> (200.0, None)
    - "200mm" -> (200.0, "mm")
    - "24.5mpa" -> (24.5, "mpa")
    - "1e5" -> (100000.0, None)
    """

    tok = (token or "").strip().lower()
    if tok in {"", ".", "-", "-."}:
        return None

    match = _NUMBER_WITH_UNIT_RE.match(tok)
    if not match:
        return None

    try:
        value = float(match.group(1))
    except ValueError:
        return None

    unit = match.group(2)
    return float(value), (unit.lower() if unit else None)


def _extract_all_numbers_with_units(text: str) -> List[Tuple[float, Optional[str]]]:
    """Return all numeric occurrences in token order, including inline units like 200mm."""

    tokens = _tokenize_for_numbers(text)
    out: List[Tuple[float, Optional[str]]] = []
    for i, tok in enumerate(tokens):
        parsed = _parse_number_token_with_unit(tok)
        if not parsed:
            continue
        value, unit = parsed
        if unit is None and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            if next_tok in _KNOWN_UNIT_TOKENS:
                unit = next_tok
        out.append((float(value), unit))
    return out


def _parse_node_list_candidates(text: str) -> Dict[str, List[float]]:
    import re

    out: Dict[str, List[float]] = {}
    normalized = text.lower()
    patterns = {
        "x": r"\b(?:node\s*)?x(?!\d)(?:\s*(?:coordinates?|coords?|nodes?))?\s*(?:=|:|are|is)?\s*([0-9.,\s-]+)",
        "y": r"\b(?:node\s*)?y(?!\d)(?:\s*(?:coordinates?|coords?|nodes?))?\s*(?:=|:|are|is)?\s*([0-9.,\s-]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, normalized)
        if not match:
            continue
        nums = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        values = [float(n) for n in nums]
        if len(values) >= 2:
            out[key] = values
    return out


def _find_phrase_indices(tokens: List[str], phrase_tokens: List[str]) -> List[int]:
    indices: List[int] = []
    n = len(phrase_tokens)
    if n == 0:
        return indices
    for i in range(0, len(tokens) - n + 1):
        if tokens[i : i + n] == phrase_tokens:
            indices.append(i)
    return indices


def _length_to_mm(value: float, unit: Optional[str]) -> float:
    if unit is None:
        return float(value)
    unit_lower = unit.strip().lower()
    multipliers = {
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "meter": 1000.0,
        "meters": 1000.0,
        "ft": 304.8,
        "feet": 304.8,
        "foot": 304.8,
        "inch": 25.4,
        "inches": 25.4,
    }
    return float(value) * multipliers.get(unit_lower, 1.0)


def _compute_contact_pressure_q_mpa(
    load_value: float,
    load_unit: str,
    contact_lx_mm: float,
    contact_ly_mm: float,
) -> float:
    """Compute tire contact pressure q in MPa.

    Uses the explicit conversion chain:
    - area(m²) = (lx_mm * ly_mm) / 1_000_000
    - pressure(MPa) = load(kN) / area(m²) / 1000

    Note: This is numerically equivalent to N/mm², but kept explicit to
    prevent unit-mismatch bugs when parsing mixed units.
    """

    _LOAD_TO_KN = {
        "kn": 1.0,
        "n": 0.001,
        "kgf": 0.00981,
        "kg": 0.00981,
        "t": 9.81,
        "ton": 9.81,
        "tons": 9.81,
        "tonne": 9.81,
        "tonnes": 9.81,
    }
    unit_lower = (load_unit or "n").strip().lower()
    load_kn = float(load_value) * _LOAD_TO_KN.get(unit_lower, 1.0)
    area_m2 = max((float(contact_lx_mm) * float(contact_ly_mm)) / 1_000_000.0, 1e-12)
    return load_kn / area_m2 / 1000.0


def _extract_assumed_values(text: str) -> Dict[str, float]:
    key_map = {
        "emod": "Emod",
        "nu": "nu",
        "a": "a",
        "b": "b",
        "t": "t",
        "kx": "Kx",
        "ky": "Ky",
        "kz": "Kz",
        "x1": "x1",
        "x2": "x2",
        "y1": "y1",
        "y2": "y2",
        "q": "q",
    }

    assumed: Dict[str, float] = {}
    # Parse each "assume ..." clause independently.
    for clause_match in re.finditer(r"\bassume\b[^.\n;]*", text):
        clause = clause_match.group(0)

        # Supports compact chained assignments like: kx=ky=0
        chain_match = re.search(
            r"\b(emod|nu|a|b|t|kx|ky|kz|x1|x2|y1|y2|q)\b\s*=\s*"
            r"\b(emod|nu|a|b|t|kx|ky|kz|x1|x2|y1|y2|q)\b\s*=\s*(-?\d+(?:\.\d+)?)",
            clause,
        )
        if chain_match:
            k1 = key_map[chain_match.group(1)]
            k2 = key_map[chain_match.group(2)]
            v = float(chain_match.group(3))
            assumed[k1] = v
            assumed[k2] = v

        # Supports explicit assignments like: assume kx=0, ky=0
        for key, value in re.findall(
            r"\b(emod|nu|a|b|t|kx|ky|kz|x1|x2|y1|y2|q)\b\s*=\s*(-?\d+(?:\.\d+)?)",
            clause,
        ):
            assumed[key_map[key]] = float(value)

    return assumed


def _extract_engineering_candidates(text: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    lowered = text.lower().replace("\n", " ")
    out: Dict[str, Any] = {}
    units: Dict[str, str] = {}

    # Capture slab dimensions like "5m x 5m" and prefer lines that mention slab/dimensions.
    dim_patterns = [
        r"(?:slab\s*(?:dimensions?|size)?|dimensions?)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?(?:\s*\([^)]*\))?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?(?:\s*\([^)]*\))?",
        r"\b(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)\s*(?:\([^)]*\))?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)\s*(?:\([^)]*\))?\b",
    ]
    for pattern in dim_patterns:
        dim_match = re.search(pattern, lowered)
        if dim_match:
            a_val = _length_to_mm(float(dim_match.group(1)), dim_match.group(2))
            b_val = _length_to_mm(float(dim_match.group(3)), dim_match.group(4))
            out["a"] = round(a_val, 3)
            out["b"] = round(b_val, 3)
            break

    thickness_match = re.search(
        r"(?:slab\s*)?thickness\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?",
        lowered,
    )
    if not thickness_match:
        thickness_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)\s*(?:thick|thickness)\b",
            lowered,
        )
    if thickness_match:
        out["t"] = round(_length_to_mm(float(thickness_match.group(1)), thickness_match.group(2)), 3)

    # Allow users to request a standard concrete Poisson ratio without a numeric value.
    if re.search(r"(?:standard|typical)\s*(?:poisson(?:'?s)?\s*ratio|\bnu\b)", lowered):
        out["nu"] = STANDARD_POISSON_RATIO

    fck_value: Optional[float] = None

    # Handles phrases like "concrete compressive strength 45 MPa" or "fck 40".
    grade_match = re.search(
        r"(?:concrete\s*(?:grade|compressive\s*strength)?|grade|fck)\s*[:=-]?\s*(?:m|c)?\s*(\d+(?:\.\d+)?)",
        lowered,
    )
    if grade_match:
        fck_value = float(grade_match.group(1))

    # Handles standard grade notation like M40, M-40, C30, C-30.
    if fck_value is None:
        grade_code_match = re.search(r"\b(?:m|c)\s*[-:]?\s*(\d+(?:\.\d+)?)\b(?:\s*grade|\s*concrete)?", lowered)
        if grade_code_match:
            fck_value = float(grade_code_match.group(1))

    if fck_value is not None:
        # IS 456 empirical relation: E (MPa) = 5000 * sqrt(fck in MPa).
        out["Emod"] = round(5000.0 * math.sqrt(fck_value), 3)

    # Capture subgrade/foundation modulus and apply in all directions by default.
    k_match = re.search(
        r"(?:subgrade|foundation|\bk\b)\s*(?:modulus)?\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(?:mpa\s*/\s*m|mpa/m)?",
        lowered,
    )
    if k_match:
        k_val = float(k_match.group(1))
        if "kz only" in lowered or "only kz" in lowered:
            out["Kz"] = k_val
            if "kx=ky=0" in lowered or "assume kx=ky=0" in lowered:
                out["Kx"] = 0.0
                out["Ky"] = 0.0
        else:
            out["Kx"] = k_val
            out["Ky"] = k_val
            out["Kz"] = k_val

    # Explicit phrase-based override for horizontal subgrade resistance.
    if re.search(r"no\s+horizontal\s+resistance|without\s+horizontal\s+resistance|horizontal\s+resistance\s*(?:is\s*)?0", lowered):
        out["Kx"] = 0.0
        out["Ky"] = 0.0

    for key in ["Kx", "Ky", "Kz"]:
        direct_match = re.search(rf"\b{key.lower()}\b\s*[:=-]?\s*(\d+(?:\.\d+)?)", lowered)
        if direct_match:
            out[key] = float(direct_match.group(1))

    load_match: Optional[re.Match[str]] = None
    _LOAD_UNIT_PAT = r"(kn|n|t\b|ton|tons|tonne|tonnes|kg|kgf)"
    load_patterns = [
        r"(?:each\s*wheel|per\s*wheel|wheel\s*load|(?:wheel\s*)?load)\s*[:=-]?\s*[^\d]{0,20}(\d+(?:\.\d+)?)\s*" + _LOAD_UNIT_PAT,
        r"\bload\b\s*(?:of\s*)?[:=-]?\s*(\d+(?:\.\d+)?)\s*" + _LOAD_UNIT_PAT + r"\b",
        r"(\d+(?:\.\d+)?)\s*" + _LOAD_UNIT_PAT + r"\s*(?:each\s*wheel|per\s*wheel|wheel\s*)?load\b",
    ]
    for pattern in load_patterns:
        candidate = re.search(pattern, lowered)
        if candidate:
            load_match = candidate
            break

    area_match: Optional[re.Match[str]] = None
    area_patterns = [
        r"(?:load\s*)?(?:tire|tyre)?\s*contact\s*(?:area|patch)?\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?",
        r"(?:contact\s*(?:area|patch)?\s*(?:of|is|=|:)?\s*)(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?",
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?\s*(?:contact|contact\s*area|contact\s*patch|tire\s*contact|tyre\s*contact)",
        r"(?:each\s*wheel|per\s*wheel|wheel\s*load|(?:wheel\s*)?load)[^.\n;]*?\bover\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?",
        r"\bover\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|meter|meters)?",
    ]
    for pattern in area_patterns:
        candidate = re.search(pattern, lowered)
        if candidate:
            area_match = candidate
            break

    area_lx_mm: Optional[float] = None
    area_ly_mm: Optional[float] = None
    if area_match:
        area_lx_mm = _length_to_mm(float(area_match.group(1)), area_match.group(2))
        area_ly_mm = _length_to_mm(float(area_match.group(3)), area_match.group(4))
        # Store as private params so apply_implicit_inferences() can use actual contact dims.
        out["_contact_lx_mm"] = area_lx_mm
        out["_contact_ly_mm"] = area_ly_mm

    # If contact dimensions are provided but "load" isn't explicitly labeled,
    # treat the first kN/N magnitude as the load.
    if load_match is None and area_lx_mm is not None and area_ly_mm is not None:
        load_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(kn|n|t\b|ton|tons|tonne|tonnes|kg|kgf)\b(?!\s*/)", lowered)

    generic_spacing_mm = _extract_spacing_mm(lowered)
    wheel_spacing_mm = _extract_wheel_spacing_mm(lowered)
    axle_spacing_mm = _extract_axle_spacing_mm(lowered)

    tandem_requested = bool(re.search(r"\btandem\s*axle\b|\btandem\b", lowered))
    dual_requested = bool(re.search(r"\bdual\s*wheel\b|\bdouble\s*wheel\b|\bdual\b", lowered))

    if wheel_spacing_mm is None:
        wheel_spacing_mm = generic_spacing_mm

    if tandem_requested:
        wheels_per_axle = 2 if dual_requested else 1
        number_of_wheels = 2 * wheels_per_axle
    else:
        wheels_per_axle = max(1, _detect_number_of_wheels(lowered))
        number_of_wheels = wheels_per_axle

    if tandem_requested and axle_spacing_mm is None:
        axle_spacing_mm = generic_spacing_mm

    inferred_location = infer_semantic_choices(text).get(LOAD_LOCATION_KEY)
    slab_a = float(out["a"]) if "a" in out else float(DEFAULT_VALUES["a"])
    slab_b = float(out["b"]) if "b" in out else float(DEFAULT_VALUES["b"])

    load_cases: List[Dict[str, float]] = []
    if area_lx_mm is not None and area_ly_mm is not None:
        load_case_pattern = re.compile(
            r"[^.\n;]*?(?:wheel\s*)?load[^\d]{0,20}(\d+(?:\.\d+)?)\s*(kn|n)[^.\n;]*?\b(?:at\s+)?(corner|edge|interior)\b"
        )
        for case_match in load_case_pattern.finditer(lowered):
            load_value = float(case_match.group(1))
            load_unit = (case_match.group(2) or "n").lower()
            location = normalize_load_location(case_match.group(3)) or "interior"
            q_val = round(
                _compute_contact_pressure_q_mpa(load_value, load_unit, area_lx_mm, area_ly_mm),
                5,
            )
            patch = infer_load_patch_from_contact_area(
                location,
                slab_a,
                slab_b,
                area_lx_mm,
                area_ly_mm,
                lowered,
            )
            load_cases.append(
                {
                    "x1": float(patch["x1"]),
                    "x2": float(patch["x2"]),
                    "y1": float(patch["y1"]),
                    "y2": float(patch["y2"]),
                    "q": float(q_val),
                }
            )

    # Multi-wheel groups (tandem, tridem, dual) should be decomposed around axle/group center.
    if number_of_wheels > 1 and load_match and area_lx_mm is not None and area_ly_mm is not None:
        load_value = float(load_match.group(1))
        load_unit = (load_match.group(2) or "n").lower()
        # If user says "axle load" / "total load", divide equally across all wheels.
        axle_total = bool(re.search(r"\baxle\s+load\b|\btotal\s+(?:axle\s+)?load\b", lowered))
        per_wheel_factor = (1.0 / number_of_wheels) if axle_total else 1.0
        q_val = round(
            _compute_contact_pressure_q_mpa(load_value * per_wheel_factor, load_unit, area_lx_mm, area_ly_mm),
            5,
        )

        group_location = inferred_location or (normalize_load_location(out.get("load_location")) if isinstance(out.get("load_location"), str) else None) or "interior"
        spacing_for_group = wheel_spacing_mm if isinstance(wheel_spacing_mm, (int, float)) else float(area_lx_mm)
        axle_spacing_for_group = (
            float(axle_spacing_mm)
            if isinstance(axle_spacing_mm, (int, float))
            else float(spacing_for_group)
        )
        load_cases = _build_group_wheel_load_cases(
            number_of_wheels=number_of_wheels,
            spacing_mm=float(spacing_for_group),
            slab_a=slab_a,
            slab_b=slab_b,
            contact_lx_mm=float(area_lx_mm),
            contact_ly_mm=float(area_ly_mm),
            load_location=group_location,
            text=lowered,
            q_value=float(q_val),
            is_tandem=tandem_requested,
            wheels_per_axle=wheels_per_axle,
            axle_spacing_mm=axle_spacing_for_group,
        )

    if load_cases:
        out["load_cases"] = load_cases
        out.update(load_cases[0])
    elif tandem_requested and inferred_location and load_match and area_lx_mm is not None and area_ly_mm is not None:
        load_value = float(load_match.group(1))
        load_unit = (load_match.group(2) or "n").lower()
        q_val = round(
            _compute_contact_pressure_q_mpa(load_value, load_unit, area_lx_mm, area_ly_mm),
            5,
        )

        spacing_for_group = wheel_spacing_mm if isinstance(wheel_spacing_mm, (int, float)) else float(area_lx_mm)
        axle_spacing_for_group = (
            float(axle_spacing_mm)
            if isinstance(axle_spacing_mm, (int, float))
            else float(spacing_for_group)
        )
        load_cases = _build_group_wheel_load_cases(
            number_of_wheels=number_of_wheels,
            spacing_mm=float(spacing_for_group),
            slab_a=slab_a,
            slab_b=slab_b,
            contact_lx_mm=float(area_lx_mm),
            contact_ly_mm=float(area_ly_mm),
            load_location=inferred_location,
            text=lowered,
            q_value=float(q_val),
            is_tandem=True,
            wheels_per_axle=wheels_per_axle,
            axle_spacing_mm=axle_spacing_for_group,
        )
        out["load_cases"] = load_cases
        out.update(load_cases[0])
    elif area_lx_mm is not None and area_ly_mm is not None:
        # Use _compute_patch_from_description so edge-distance descriptions ("1m from left...")
        # and "center of slab" are handled in addition to named location classes.
        patch = _compute_patch_from_description(
            text, slab_a, slab_b, area_lx_mm, area_ly_mm, load_location=inferred_location
        )
        if patch is not None:
            out.update(patch)

    explicit_q = re.search(r"(?:pressure|\bq\b)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mpa|kpa|psi)?", lowered)
    has_load_and_area = load_match is not None and area_lx_mm is not None and area_ly_mm is not None

    # Store raw load value so _detect_extraction_ambiguity() can see a load was given
    # even when contact area is missing (and q can't yet be computed).
    if load_match:
        raw_load_kn_val = float(load_match.group(1))
        raw_load_unit = (load_match.group(2) or "n").lower()
        _LOAD_TO_KN_LOCAL = {
            "kn": 1.0, "n": 0.001, "kgf": 0.00981, "kg": 0.00981,
            "t": 9.81, "ton": 9.81, "tons": 9.81, "tonne": 9.81, "tonnes": 9.81,
        }
        out["_raw_load_kn"] = raw_load_kn_val * _LOAD_TO_KN_LOCAL.get(raw_load_unit, 1.0)

    # Priority (as required):
    # 1) Load + area => compute pressure
    # 2) Explicit pressure => use provided
    # 3) Otherwise => leave missing (defaults handled elsewhere)
    if has_load_and_area and not load_cases:
        load_value = float(load_match.group(1))
        load_unit = (load_match.group(2) or "n").lower()
        out["q"] = round(
            _compute_contact_pressure_q_mpa(load_value, load_unit, float(area_lx_mm), float(area_ly_mm)),
            5,
        )
    elif explicit_q:
        out["q"] = float(explicit_q.group(1))
        if explicit_q.group(2):
            units["q"] = explicit_q.group(2)

    # Explicit user assumptions should override inferred values.
    assumed_values = _extract_assumed_values(lowered)
    if assumed_values:
        out.update(assumed_values)

    return out, units


def parse_multi_param_candidates(text: str) -> Dict[str, Tuple[float, Optional[str]]]:
    tokens = _tokenize_for_numbers(text)
    if not tokens:
        return {}

    numbers: List[Tuple[int, float, Optional[str]]] = []
    for i, tok in enumerate(tokens):
        parsed = _parse_number_token_with_unit(tok)
        if not parsed:
            continue
        val, unit = parsed
        if unit is None and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            if next_tok in _KNOWN_UNIT_TOKENS:
                unit = next_tok
        numbers.append((i, float(val), unit))

    if not numbers:
        return {}

    out: Dict[str, Tuple[float, Optional[str]]] = {}
    for key, aliases in PARAM_ALIASES.items():
        best_match: Optional[Tuple[float, Optional[str], int]] = None
        for alias in aliases:
            phrase_tokens = alias.split()
            starts = _find_phrase_indices(tokens, phrase_tokens)
            if not starts:
                continue
            for start in starts:
                end = start + len(phrase_tokens) - 1
                chosen: Optional[Tuple[int, float, Optional[str]]] = None
                for idx, val, unit in numbers:
                    if idx > end and idx - end <= 4:
                        chosen = (idx, val, unit)
                        break
                if chosen is None:
                    for idx, val, unit in reversed(numbers):
                        if idx < start and start - idx <= 3:
                            chosen = (idx, val, unit)
                            break
                if chosen is not None:
                    dist = abs(chosen[0] - end)
                    if best_match is None or dist < best_match[2]:
                        best_match = (chosen[1], chosen[2], dist)
        if best_match is not None:
            out[key] = (best_match[0], best_match[1])

    return out


def _progress_lines(params: Dict[str, Any], current_asking: Optional[str]) -> str:
    total = len(PARAM_ORDER)
    done = len([k for k in PARAM_ORDER if k in params])
    pct = int(round((done / total) * 100)) if total else 0
    remaining = total - done

    next_key = current_asking if current_asking else next((k for k in PARAM_ORDER if k not in params), None)

    lines = [
        "### Progress",
        f"- Completed: **{done}/{total} ({pct}%)**",
        f"- Remaining: **{remaining}**",
    ]

    if next_key and next_key in PARAM_INFO:
        lines.append(f"- Next focus: **{PARAM_INFO[next_key]['name']}**")

    remembered = [k for k in PARAM_ORDER if k in params][-4:]
    if remembered:
        snapshot = ", ".join(
            f"{PARAM_INFO[k]['name']}: {format_value_with_unit(float(params[k]), k) if isinstance(params[k], (int, float)) else _compact_value(params[k])}"
            for k in remembered
        )
        lines.append(f"- I remember: {snapshot}")

    return "\n".join(lines)


def with_progress_tail(message: str, params: Dict[str, Any], current_asking: Optional[str], include_progress: bool = True) -> str:
    if not include_progress:
        return message
    if "### Progress" in message:
        return message
    return f"{message.rstrip()}\n\n---\n\n{_progress_lines(params, current_asking)}\n\n---"


def generate_interactive_followup(params: Dict[str, Any]) -> str:
    missing = [k for k in PARAM_ORDER if k not in params]
    if not missing:
        return ""

    grouped: Dict[str, List[str]] = {}
    for key in missing:
        category = PARAM_INFO[key].get("category", "Other")
        grouped.setdefault(category, []).append(key)

    lines = [
        "You're doing great. If you know multiple details, share them in one message and I'll capture them together.",
        "",
        "You can give any of these next:",
    ]

    shown_categories = 0
    for category, keys in grouped.items():
        if shown_categories >= 3:
            break
        names = ", ".join(PARAM_INFO[k]["name"] for k in keys[:3])
        lines.append(f"- **{category}:** {names}")
        shown_categories += 1

    lines.extend(
        [
            "",
            "Tip: You can reply like \"length 4.5 m, width 3.5 m, mesh medium, load at edge\".",
            "If you'd like, say \"use defaults for remaining\" and I'll finish the rest safely.",
        ]
    )
    return "\n".join(lines)


def create_expert_system_prompt() -> str:
    param_catalog = {
        k: {
            "name": v["name"],
            "unit": v.get("unit", ""),
            "typical_range": v.get("typical_range", ""),
            "simple_explanation": v.get("simple_explanation", ""),
        }
        for k, v in PARAM_INFO.items()
        if not str(k).startswith("_") and k not in ("x", "y")
    }

    return (
        "You are a rigid pavement FEM configuration assistant with expert knowledge of "
        "IS 456, IRC:58, and pavement engineering.\n\n"
        "YOUR ROLE:\n"
        "- Extract parameters from the user's message\n"
        "- Compute ALL derived values yourself using the exact formulas below\n"
        "- Never guess or assume ambiguous inputs — ask exactly ONE clarification question if unclear\n"
        "- Return ONLY a JSON object matching the schema at the end of this prompt\n\n"

        "═══════════════════════════════════════════════════\n"
        "PARAMETER CATALOGUE\n"
        "═══════════════════════════════════════════════════\n"
        + json.dumps(param_catalog, indent=2)
        + "\n\n"

        "═══════════════════════════════════════════════════\n"
        "DISAMBIGUATION RULES — READ CAREFULLY\n"
        "═══════════════════════════════════════════════════\n"
        "1. DIMENSION PAIRS (e.g. '5000×2000', '350×300'):\n"
        "   A pair WITHOUT the words 'contact', 'tyre', 'tire', 'patch', or 'over' is AMBIGUOUS.\n"
        "   NEVER assume it is slab dimensions OR contact area without clear context.\n"
        "   If ambiguous → needs_clarification=true, ask: 'Is [N]×[M] mm the slab size or the tyre contact dimensions?'\n"
        "   CLEAR: 'slab 5m×3m' → a=5000, b=3000\n"
        "   CLEAR: 'contact area 350×300' → contact lx=350, ly=300\n"
        "   CLEAR: 'over 350×300 mm' or 'tyre contact 350×300' → contact area\n"
        "   AMBIGUOUS: '5000×2000' alone → ask\n\n"
        "2. LOAD WITHOUT CONTACT AREA:\n"
        "   If load (kN) is given but no contact dimensions → needs_clarification=true.\n"
        "   Ask: 'What are the tyre contact dimensions (length × width in mm)?'\n\n"
        "3. EDGE SPECIFICATION:\n"
        "   If user says 'edge load' without specifying which edge → needs_clarification=true.\n"
        "   Ask: 'Which edge — x=0 (near), x=a (far), y=0 (left), or y=b (right)?'\n"
        "   Exception: 'corner edge' implies corner; directional hints ('along the length') imply y=0/y=b.\n\n"

        "═══════════════════════════════════════════════════\n"
        "MANDATORY COMPUTATION FORMULAS\n"
        "═══════════════════════════════════════════════════\n"
        "1. ELASTIC MODULUS FROM GRADE (IS 456):\n"
        "   E (MPa) = 5000 × √fck   [fck = characteristic compressive strength in MPa]\n"
        "   M20→22361, M25→25000, M30→27386, M35→29580, M40→31623 MPa\n\n"
        "2. CONTACT PRESSURE q:\n"
        "   q (MPa) = P_kN × 1000 / (Lx_mm × Ly_mm)\n"
        "   Example: 80 kN, 350×300 mm → q = 80000 / 105000 = 0.7619 MPa\n\n"
        "3. POISSON RATIO DEFAULT (IRC:58): ν = 0.15 — apply silently, do not ask.\n\n"
        "4. SUBGRADE MODULUS K: one K value → Kx=Ky=Kz=K (unless user differentiates).\n\n"
        "5. NODE COORDINATE ARRAYS (always compute both x[] and y[]):\n"
        "   coarse → 11 nodes: x[i] = i × (a/10)  for i=0..10\n"
        "   medium → 16 nodes: x[i] = i × (a/15)  for i=0..15\n"
        "   fine   → 31 nodes: x[i] = i × (a/30)  for i=0..30\n"
        "   Round each value to 6 decimal places.\n"
        "   Example: medium, a=5000 → x=[0, 333.333333, 666.666667, ..., 5000.0]\n\n"
        "6. LOAD PATCH COORDINATES (single wheel, contact Lx × Ly mm):\n"
        "   corner:    x1=0,        x2=Lx,       y1=0,          y2=Ly\n"
        "   edge x=0:  x1=0,        x2=Lx,       y1=(b-Ly)/2,   y2=(b+Ly)/2\n"
        "   edge x=a:  x1=a-Lx,     x2=a,        y1=(b-Ly)/2,   y2=(b+Ly)/2\n"
        "   edge y=0:  x1=(a-Lx)/2, x2=(a+Lx)/2, y1=0,          y2=Ly\n"
        "   edge y=b:  x1=(a-Lx)/2, x2=(a+Lx)/2, y1=b-Ly,       y2=b\n"
        "   interior:  x1=(a-Lx)/2, x2=(a+Lx)/2, y1=(b-Ly)/2,   y2=(b+Ly)/2\n\n"
        "7. MULTI-WHEEL load_cases (each entry = one tyre patch {x1,x2,y1,y2,q}):\n"
        "   DUAL TYRES (side by side, across slab y-direction, wheel spacing s):\n"
        "     Wheel 1: y_center = b/2 - s/2 → y1=y_center-Ly/2, y2=y_center+Ly/2\n"
        "     Wheel 2: y_center = b/2 + s/2 → y1=y_center-Ly/2, y2=y_center+Ly/2\n"
        "     Both wheels: x1=(a-Lx)/2, x2=(a+Lx)/2\n"
        "   TANDEM AXLE (two axles along slab x-direction, axle spacing d):\n"
        "     Axle 1: x_center = a/2 - d/2 → x1=x_center-Lx/2, x2=x_center+Lx/2\n"
        "     Axle 2: x_center = a/2 + d/2 → x1=x_center-Lx/2, x2=x_center+Lx/2\n"
        "     Both axles: y1=(b-Ly)/2, y2=(b+Ly)/2\n"
        "   TANDEM + DUAL: 4 patches (2 axles × 2 tyre positions)\n"
        "   PER-WHEEL q: if axle total load P_total with N wheels → q = (P_total/N)×1000/(Lx×Ly)\n"
        "   Single wheel: load_cases=[], put x1/x2/y1/y2 directly in computed.\n"
        "   Multi-wheel: populate load_cases; ALSO copy first entry values to top-level x1/x2/y1/y2/q.\n\n"
        "8. UNIT CONVERSIONS:\n"
        "   Length: 1m=1000mm, 1cm=10mm, 1ft=304.8mm, 1inch=25.4mm\n"
        "   Load:   1kN=1000N, 1ton=9.81kN, 1tonne=9.81kN, 1kg=0.00981kN\n"
        "   Pressure: 1kPa=0.001MPa, 1psi=0.006895MPa, 1GPa=1000MPa\n\n"

        "═══════════════════════════════════════════════════\n"
        "WHEN TO ASK FOR CLARIFICATION\n"
        "═══════════════════════════════════════════════════\n"
        "ASK (needs_clarification=true) when:\n"
        "- Dimension pair given without 'contact'/'tyre'/'patch' context\n"
        "- Load given but no contact dimensions\n"
        "- Load location not specified or cannot be inferred\n"
        "- 'edge' without specifying which edge\n"
        "- Contradictory values (contact area > slab)\n\n"
        "DO NOT ASK when:\n"
        "- E is computable from grade given\n"
        "- q is computable from load + contact dims given\n"
        "- ν not stated → use 0.15 silently\n"
        "- mesh_type known → compute node arrays without asking\n"
        "- load_location + contact dims known → compute patch coords without asking\n\n"

        "═══════════════════════════════════════════════════\n"
        "RECOGNISING SPECIAL INTENTS\n"
        "═══════════════════════════════════════════════════\n"
        "'yes'/'ok'/'sure'/'that\\'s fine' when asked about a param → use_default=true\n"
        "'use all defaults'/'fill with defaults' → use_all_defaults=true, compute known derived values\n"
        "Greetings ('hi','hello') → conversation_only=true, warm greeting, no params extracted\n"
        "Help requests ('help','what can you do') → conversation_only=true, explain capabilities\n"
        "'what is [param]'/'explain [param]' → conversation_only=true, explain the parameter\n\n"

        "═══════════════════════════════════════════════════\n"
        "OUTPUT JSON SCHEMA — return ONLY this object, no other text\n"
        "═══════════════════════════════════════════════════\n"
        "{\n"
        '  "needs_clarification": false,\n'
        '  "clarification_question": null,\n'
        '  "friendly_response": "Brief warm confirmation of what was understood and computed.",\n'
        '  "use_default": false,\n'
        '  "use_all_defaults": false,\n'
        '  "conversation_only": false,\n'
        '  "understood": {\n'
        '    // Human-readable summary of what the user provided\n'
        '    // e.g. "slab_a_mm": 5000, "fck_mpa": 30, "load_kn": 80\n'
        '    // Omit keys the user did not mention\n'
        '  },\n'
        '  "computed": {\n'
        '    // Solver-ready final values — use EXACT key names shown\n'
        '    // Only include params that were provided or derivable from provided values\n'
        '    "a": <number|omit>,\n'
        '    "b": <number|omit>,\n'
        '    "t": <number|omit>,\n'
        '    "Emod": <number|omit>,\n'
        '    "nu": <number|omit>,\n'
        '    "Kx": <number|omit>,\n'
        '    "Ky": <number|omit>,\n'
        '    "Kz": <number|omit>,\n'
        '    "q": <number|omit>,\n'
        '    "x1": <number|omit>,\n'
        '    "x2": <number|omit>,\n'
        '    "y1": <number|omit>,\n'
        '    "y2": <number|omit>,\n'
        '    "mesh_type": <"coarse"|"medium"|"fine"|omit>,\n'
        '    "load_location": <"corner"|"edge"|"interior"|omit>,\n'
        '    "x": [<evenly spaced nodes from 0 to a>|omit],\n'
        '    "y": [<evenly spaced nodes from 0 to b>|omit],\n'
        '    "load_cases": []  // empty for single wheel; [{x1,x2,y1,y2,q},...] for multi-wheel\n'
        '  }\n'
        "}\n\n"
        "WHEN needs_clarification=true: set understood={}, computed={}, populate clarification_question.\n"
        "NEVER output anything outside the JSON object."
    )


def create_extraction_prompt(user_input: str, context: str, current_asking: Optional[str] = None) -> str:
    prompt = f"""### CONTEXT
{context}

### NODE CONTEXT
- Node coordinates are inferred from mesh_type:
    - coarse -> 10x10 elements
    - medium -> 15x15 elements
    - fine -> 30x30 elements
- Load coordinates are inferred from load_location:
    - corner, edge, interior
- Extract mesh_type/load_location when user mentions these classes.

### USER INPUT
"{user_input}"

### CURRENTLY ASKING ABOUT
{f"Parameter: {current_asking} ({PARAM_INFO[current_asking]['name']})" if current_asking else "General conversation - extract any parameters mentioned"}

### EXAMPLES (few-shot)

Input: "5m × 3.5m slab, 250mm thick, M30 concrete, K=60, 80kN load at edge over 350×300mm contact"
→ extracted_multiple: {{"a":5000,"b":3500,"t":250,"Emod":27386,"Kx":60,"Ky":60,"Kz":60,"load_location":"edge","q":0.762}}
→ friendly_response: "Great — I captured slab dimensions, M30 grade (E = 27,386 MPa from IS 456), K = 60, and computed contact pressure 0.762 MPa from the 80 kN load over 350 × 300 mm."

Input: "use medium mesh, load at corner with 200×200 contact"
→ extracted_multiple: {{"mesh_type":"medium","load_location":"corner"}}
→ friendly_response: "Medium mesh and corner load noted. I'll compute the load patch starting at the corner based on your 200 × 200 mm contact area."

Input: "tandem, two 60kN wheels, 300×300mm each, 1.2m spacing, interior"
→ extracted_multiple: {{"load_location":"interior","q":0.667}}
→ friendly_response: "Tandem setup noted — two 60 kN wheels at 1.2 m spacing, each on 300 × 300 mm contact (q = 0.667 MPa). I'll compute both wheel positions for the interior case."

### TASK
1. Decide user intent:
    - provide one value,
    - provide multiple values,
    - use default for current field,
    - use defaults for all remaining fields,
    - ask a question / need clarification.
2. Extract values and units if present.
3. Map values to correct parameter keys.
4. If uncertain, set needs_clarification=true.
5. Write friendly_response as a natural beginner-friendly assistant.

### OUTPUT FORMAT (JSON only, no other text):
{{
    "understood_value": <number|string or null if not providing a value>,
    "original_unit": "<unit mentioned by user or null>",
    "parameter_key": "<which parameter this is for>",
    "use_default": <true/false - if user wants default for CURRENT parameter>,
    "use_all_defaults": <true/false - if user wants defaults for ALL remaining parameters>,
    "needs_clarification": <true/false>,
    "friendly_response": "<your warm, friendly response to the user>",
    "extracted_multiple": {{<any additional parameters if user mentioned multiple>}}
}}

IMPORTANT DETECTION RULES:
- Prioritize extracting multiple parameters from one user message when possible.
- If a user message includes updates to existing values, use the newest values.
- Detect mesh_type from: coarse / medium / fine (or 1/2/3).
- Detect load_location from: corner / edge / interior (or 1/2/3).
- Never expose raw JSON, schema keys, or implementation details in friendly_response.
- Be warm, concise, and proactive.

JSON RESPONSE:"""
    return prompt


def build_llm_message_history(messages: List[Dict]) -> List:
    lc_messages = []
    recent = messages[-10:] if len(messages) > 10 else messages
    for msg in recent:
        content = msg["content"]
        if len(content) > 500:
            content = content[:500] + "..."
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=content))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages


def _invoke_llm_with_timeout(llm: ChatOllama, messages: List, timeout_seconds: float = LLM_INVOKE_TIMEOUT_SECONDS):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(llm.invoke, messages)
    try:
        timeout = float(timeout_seconds)
        if timeout <= 0:
            # Optional "no-timeout" mode for long cloud inference runs.
            return future.result()
        return future.result(timeout=timeout)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"LLM request timed out after {timeout:.1f}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _detect_extraction_ambiguity(
    extracted: Dict[str, Any],
    params: Dict[str, Any],
) -> Optional[str]:
    """Return a clarification question if extracted values are paired incompletely.

    Catches the two most common gaps that require the user's help before the
    assistant can compute a derived value (e.g., q from load + contact area).
    Returns None when no ambiguity is detected.
    """
    has_contact = "_contact_lx_mm" in extracted
    has_load_location = "load_location" in extracted or "load_location" in params
    has_computed_q = "q" in extracted

    if has_contact and not has_load_location:
        return (
            "I have the contact area dimensions — where on the slab does this load act? "
            "(corner, edge, or interior)"
        )

    has_raw_load = "_raw_load_kn" in extracted
    if has_raw_load and not has_contact and not has_computed_q:
        return (
            "Got the load value. Could you share the tyre contact dimensions "
            "(e.g., 300 × 250 mm)? I'll compute the contact pressure from those."
        )

    return None


def process_user_input_with_llm(
    user_input: str,
    llm: ChatOllama,
    context: str,
    params: Optional[Dict[str, Any]] = None,
    current_asking: Optional[str] = None,
) -> Dict[str, Any]:
    """Process user input through LLM for extraction and response generation."""

    # First: deterministic conversation handling (greetings, help, explanations).
    convo = handle_conversational_intent(user_input, params=params or {}, current_asking=current_asking)
    if convo is not None:
        return convo

    stripped_input = user_input.strip().lower()

    if current_asking and stripped_input in SINGLE_DEFAULT_PATTERNS and not _contains_default_keyword(user_input):
        return {
            "understood_value": None,
            "use_default": True,
            "use_all_defaults": False,
            "needs_clarification": False,
            "friendly_response": "Absolutely, we can use the suggested value for this one.",
            "original_unit": None,
            "parameter_key": current_asking,
            "extracted_multiple": {},
        }

    semantic_choices = infer_semantic_choices(user_input, current_asking)
    heuristic_candidates = parse_multi_param_candidates(user_input)
    node_list_candidates = _parse_node_list_candidates(user_input)
    engineering_candidates, engineering_units = _extract_engineering_candidates(user_input)

    extracted_multiple: Dict[str, Any] = {}
    extracted_units: Dict[str, str] = {}

    if semantic_choices:
        extracted_multiple.update(semantic_choices)
    if engineering_candidates:
        extracted_multiple.update(engineering_candidates)
    for key, (value, unit) in heuristic_candidates.items():
        # Prefer deterministic engineering parsing for domain-specific fields when present.
        if key not in extracted_multiple:
            extracted_multiple[key] = value
            if unit:
                extracted_units[key] = unit
    for key, values in node_list_candidates.items():
        extracted_multiple[key] = values
    extracted_units.update(engineering_units)

    if extracted_multiple:
        # friendly_response is intentionally None here — generate_autonomous_assistant_message()
        # will narrate the captured values naturally instead of a hardcoded static string.
        friendly: Optional[str] = None

        understood_value: Optional[float] = None
        original_unit: Optional[str] = None

        # If the user included multiple values but only explicitly keyed some of them,
        # preserve the remaining (unclaimed) number as the answer to the current question.
        if (
            current_asking
            and current_asking not in extracted_multiple
            and current_asking not in NODE_PARAM_KEYS
            and current_asking not in {MESH_TYPE_KEY, LOAD_LOCATION_KEY}
        ):
            occurrences = _extract_all_numbers_with_units(user_input)

            # Build numeric claims from extracted_multiple and try to match occurrences to those claims.
            claims: List[Tuple[str, float]] = [
                (key, float(val))
                for key, val in extracted_multiple.items()
                if isinstance(val, (int, float)) and key in PARAM_INFO
            ]
            used = [False] * len(claims)
            leftovers: List[Tuple[float, Optional[str]]] = []

            tol = 1e-9
            for value, unit in occurrences:
                matched = False
                for idx, (key, claim_val) in enumerate(claims):
                    if used[idx]:
                        continue
                    if abs(claim_val - float(value)) <= tol:
                        used[idx] = True
                        matched = True
                        break
                    if unit and key not in extracted_units:
                        converted_value, _ = convert_to_standard_unit(float(value), unit, key)
                        if abs(float(converted_value) - float(claim_val)) <= tol:
                            used[idx] = True
                            matched = True
                            break
                if not matched:
                    leftovers.append((float(value), unit))

            if leftovers:
                stripped = user_input.strip().lower()
                starts_with_number = bool(re.match(r"^-?\d", stripped))

                tokens = _tokenize_for_numbers(user_input)
                alias_hit = False
                for alias in PARAM_ALIASES.get(current_asking, []):
                    phrase_tokens = alias.split()
                    if _find_phrase_indices(tokens, phrase_tokens):
                        alias_hit = True
                        break

                expected_unit = str(PARAM_INFO.get(current_asking, {}).get("unit", "")).strip().lower()
                alt_units = PARAM_INFO.get(current_asking, {}).get("alt_units", {}) or {}
                compatible_units = {expected_unit} | {str(u).strip().lower() for u in alt_units.keys()}

                chosen: Optional[Tuple[float, Optional[str]]] = None
                for value, unit in leftovers:
                    if unit and unit.strip().lower() in compatible_units:
                        chosen = (float(value), unit)
                        break

                if chosen is None and (starts_with_number or alias_hit):
                    chosen = (float(leftovers[0][0]), leftovers[0][1])

                if chosen is not None:
                    understood_value = float(chosen[0])
                    original_unit = chosen[1]

        # Check for partial/ambiguous input pairs (e.g., contact area but no load location).
        ambiguity = _detect_extraction_ambiguity(extracted_multiple, params or {})
        needs_clarification = ambiguity is not None
        if ambiguity:
            friendly = ambiguity

        return {
            "understood_value": understood_value,
            "use_default": False,
            "use_all_defaults": False,
            "needs_clarification": needs_clarification,
            "friendly_response": friendly,
            "original_unit": original_unit,
            "parameter_key": current_asking,
            "extracted_multiple": extracted_multiple,
            "extracted_units": extracted_units,
            "conversation_only": False,
        }

    # Deterministic guided fallback: if we're asking a specific field and the
    # user provided a number, capture it without requiring an LLM call.
    if current_asking:
        num, unit = _extract_first_number_with_unit(user_input)
        if isinstance(num, (int, float)):
            return {
                "understood_value": float(num),
                "use_default": False,
                "use_all_defaults": False,
                "needs_clarification": False,
                "friendly_response": "Thanks, I captured that value.",
                "original_unit": unit,
                "parameter_key": current_asking,
                "extracted_multiple": {},
            }

    try:
        system_message = SystemMessage(content=create_expert_system_prompt())
        extraction_prompt = create_extraction_prompt(user_input, context, current_asking)
        human_message = HumanMessage(content=extraction_prompt)

        response = _invoke_llm_with_timeout(llm, [system_message, human_message])

        if response and response.content:
            content = response.content.strip()

            if content.startswith("```"):
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1 :]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start : end + 1]

            try:
                result = json.loads(content)

                result.setdefault("understood_value", None)
                result.setdefault("use_default", False)
                result.setdefault("use_all_defaults", False)
                result.setdefault("needs_clarification", False)
                result.setdefault("friendly_response", "")
                result.setdefault("original_unit", None)
                result.setdefault("parameter_key", current_asking)
                result.setdefault("extracted_multiple", {})

                return result
            except json.JSONDecodeError:
                if current_asking:
                    if current_asking in heuristic_candidates:
                        num, unit = heuristic_candidates[current_asking]
                    else:
                        num, unit = _extract_first_number_with_unit(user_input)
                    if num is not None and isinstance(num, (int, float)):
                        return {
                            "understood_value": num,
                            "use_default": False,
                            "use_all_defaults": False,
                            "needs_clarification": False,
                            "friendly_response": "Thanks, I captured that value.",
                            "original_unit": unit,
                            "parameter_key": current_asking,
                            "extracted_multiple": {},
                        }
                return {
                    "understood_value": None,
                    "use_default": False,
                    "use_all_defaults": False,
                    "needs_clarification": True,
                    "friendly_response": "I'm not quite sure I understood that. Could you try again? You can enter a number, or say 'default' to use the suggested value. 😊",
                }

        return {"needs_clarification": True, "friendly_response": "I didn't catch that. Could you please try again?"}

    except TimeoutError:
        return {
            "needs_clarification": True,
            "friendly_response": "I’m having trouble reaching the model right now. If you share the key numbers (slab size, thickness, concrete grade, K value, wheel load/contact area), I can still capture them — or you can say ‘let’s begin’ for guided setup.",
        }
    except Exception as exc:
        err_text = str(exc).strip()
        lowered_err = err_text.lower()

        if "model" in lowered_err and "not found" in lowered_err:
            return {
                "needs_clarification": True,
                "friendly_response": (
                    "I couldn't use the selected Ollama model because it is not installed. "
                    "Please choose an available model in settings, or try again and I'll continue with a fallback model if available."
                ),
            }

        if "unauthorized" in lowered_err or "status code: 401" in lowered_err:
            return {
                "needs_clarification": True,
                "friendly_response": (
                    "Your Ollama model request is unauthorized (401). "
                    "This usually means the selected cloud model needs authentication. "
                    "Please switch to a locally available model in settings or configure Ollama cloud auth, then try again."
                ),
            }

        if "cannot reach ollama" in lowered_err or "connection" in lowered_err:
            return {
                "needs_clarification": True,
                "friendly_response": (
                    "I couldn't reach your Ollama server. Please ensure Ollama is running and the URL is correct, then try again."
                ),
            }

        return {
            "needs_clarification": True,
            "friendly_response": (
                "I hit an internal processing issue while understanding that message. "
                "Please try again in one sentence with key values (for example: length 5m, width 5m, thickness 200mm)."
            ),
        }


def generate_welcome_message() -> str:
    """Generate a warm welcome message."""
    return """# Welcome to the Pavement Configuration Assistant

This assistant helps you set up parameters for analyzing a concrete road pavement. You can answer in plain language; I’ll capture values and use safe defaults when needed.

### What we’ll configure
I’ll ask you about:
- **Slab size** (length, width, thickness)
- **Mesh density** (coarse / medium / fine)
- **Concrete properties**
- **Subgrade/soil support**
- **Load location** (corner / edge / interior)

### How it works
- One question at a time
- Brief explanation of what each parameter means
- Reply with a value, or type **"default"** to accept the suggested value
- If you know multiple values, you can provide them in one message

### Base units
Please use the following units:
- **Elastic Modulus (E):** MPa
- **Poisson's Ratio (ν):** dimensionless
- **Slab Length (a):** mm
- **Slab Width (b):** mm
- **Slab Thickness (t):** mm
- **Soil Stiffness (Kx, Ky, Kz):** no unit (stiffness factor)
- **Load coordinates (x1, x2, y1, y2):** mm
- **Tire pressure (q):** MPa
- **Node coordinates (x, y):** mm

---

**To begin:** describe your pavement, or type **"let's begin"** for guided setup.

*Example: "4 m square slab, medium mesh, load at edge"*

---

### Progress
- Completed: **0/11 (0%)**
- Remaining: **11**
- Next focus: **Slab Length**
"""


def generate_completion_message(params: Dict[str, Any], user_provided: List[str]) -> str:
    msg = """# 🎉 All Done! Configuration Complete!

Great job! We've successfully configured all the parameters for your pavement analysis. Here's a summary of what we set up:

"""
    for category, keys in PARAM_CATEGORIES.items():
        msg += f"\n### {category}\n"
        for key in keys:
            info = PARAM_INFO[key]
            value = params.get(key, get_default_value(key))
            source = "✏️ You provided" if key in user_provided else "⚙️ Default used"
            msg += f"- **{info['icon']} {info['name']}:** {format_value_with_unit(value, key)} ({source})\n"

    msg += """
---
### 📥 Download Your Configuration
Use the download buttons below to save your configuration!
"""
    return msg


def build_final_configuration(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build the final nested JSON payload expected by downstream consumers."""
    merged = {**DEFAULT_VALUES, **params}

    x_nodes = params.get("x") if isinstance(params.get("x"), list) else NODE_DEFAULT_VALUES["x"]
    y_nodes = params.get("y") if isinstance(params.get("y"), list) else NODE_DEFAULT_VALUES["y"]

    load_cases_input = params.get("load_cases")
    normalized_load_cases: List[Dict[str, float]] = []
    if isinstance(load_cases_input, list):
        for case in load_cases_input:
            if not isinstance(case, dict):
                continue
            required = ["x1", "x2", "y1", "y2", "q"]
            if not all(k in case and isinstance(case[k], (int, float)) for k in required):
                continue
            normalized_load_cases.append({k: float(case[k]) for k in required})

    if not normalized_load_cases:
        normalized_load_cases = [
            {
                "x1": float(merged["x1"]),
                "x2": float(merged["x2"]),
                "y1": float(merged["y1"]),
                "y2": float(merged["y2"]),
                "q": float(merged["q"]),
            }
        ]

    return {
        "nodes": {
            "x": [float(v) for v in x_nodes],
            "y": [float(v) for v in y_nodes],
        },
        "slab": {
            "Emod": float(merged["Emod"]),
            "nu": float(merged["nu"]),
            "t": float(merged["t"]),
        },
        "subgrade": {
            "Kx": float(merged["Kx"]),
            "Ky": float(merged["Ky"]),
            "Kz": float(merged["Kz"]),
        },
        "loads": {
            "x1": [case["x1"] for case in normalized_load_cases],
            "x2": [case["x2"] for case in normalized_load_cases],
            "y1": [case["y1"] for case in normalized_load_cases],
            "y2": [case["y2"] for case in normalized_load_cases],
            "q": [case["q"] for case in normalized_load_cases],
        },
    }


def _clean_llm_text(text: str) -> str:
    content = (text or "").strip()
    if not content:
        return ""

    # Strip common fenced blocks.
    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1 :]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    return content


def _render_param_value_for_prompt(key: str, value: Any) -> str:
    try:
        return format_value_with_unit(value, key)
    except Exception:
        return str(value)


def _render_known_params_for_prompt(params: Dict[str, Any]) -> str:
    if not params:
        return "- (none yet)"

    lines: List[str] = []
    ordered = list(PARAM_ORDER)
    for extra in ["x", "y", "x1", "x2", "y1", "y2"]:
        if extra in params and extra not in ordered:
            ordered.append(extra)

    for key in ordered:
        if key not in params:
            continue
        if key not in PARAM_INFO:
            continue
        info = PARAM_INFO.get(key, {})
        name = str(info.get("name", key))
        lines.append(f"- {name}: {_render_param_value_for_prompt(key, params.get(key))}")

    return "\n".join(lines) if lines else "- (none yet)"


def _render_applied_for_prompt(params: Dict[str, Any], applied_keys: List[str]) -> str:
    if not applied_keys:
        return "- (no parameter updates detected in this message)"

    lines: List[str] = []
    for key in applied_keys:
        if key not in PARAM_INFO:
            continue
        info = PARAM_INFO.get(key, {})
        name = str(info.get("name", key))
        lines.append(f"- {name}: {_render_param_value_for_prompt(key, params.get(key))}")

    return "\n".join(lines)


def _next_question_context(current_asking: Optional[str]) -> str:
    if not current_asking or current_asking not in PARAM_INFO:
        return "(none)"

    info = PARAM_INFO[current_asking]
    name = str(info.get("name", current_asking))
    tech = str(info.get("technical_name", "")).strip()
    unit = str(info.get("unit", "")).strip()
    typical = str(info.get("typical_range", "")).strip()
    example = str(info.get("example", "")).strip()
    default_rendered = _render_param_value_for_prompt(current_asking, get_default_value(current_asking))

    extra = ""
    if current_asking == "Emod":
        extra = "Alternative: tell me concrete grade (e.g., M30 / fck=30 MPa) and I can compute Elastic Modulus." 
    elif current_asking == "q":
        extra = "Alternative: give wheel load (kN) + contact patch size (e.g., 200×300 mm) and I can compute tire pressure." 

    parts = [
        f"Parameter needed next: {name}",
        f"Technical name: {tech}" if tech else "",
        f"Unit: {unit}" if unit and unit != "(no unit)" else "",
        f"Typical range: {typical}" if typical else "",
        f"Example: {example}" if example else "",
        f"Safe default available: {default_rendered}",
        extra,
    ]
    return "\n".join(p for p in parts if p)


def create_autonomous_response_system_prompt() -> str:
    return (
        "You are an expert rigid pavement (concrete pavement) assistant helping users configure "
        "finite element analysis inputs for IS-code-based pavement design.\n\n"
        "ENGINEERING KNOWLEDGE YOU MAY USE (to explain computations — never to invent values):\n"
        "- IS 456 elastic modulus: E (MPa) = 5000 × √fck (fck in MPa). E.g., M30 → E = 27,386 MPa; M40 → E = 31,623 MPa.\n"
        "- Poisson's ratio for concrete: typical 0.15–0.20 (IS 456 allows 0.15–0.25).\n"
        "- Contact pressure: q (MPa) = load (kN) / contact_area (m²) / 1000. "
        "E.g., 80 kN on 0.35 × 0.30 m → q = 80 / 0.105 / 1000 = 0.762 MPa.\n"
        "- Corner load: patch starts at slab corner → x1=0, x2=contact_lx, y1=0, y2=contact_ly.\n"
        "- Edge load: one edge of the patch is flush with the slab boundary; other coordinates from contact dims.\n"
        "- Interior load: patch centred within the slab with clearance from all edges.\n"
        "- Subgrade modulus K (dimensionless factor here): soft soil 20–30, medium 40–80, hard/stabilised 100–300.\n"
        "- Mesh levels: coarse = 10×10 elements, medium = 15×15, fine = 30×30.\n\n"
        "HARD RULES:\n"
        "- Do NOT invent or guess numeric values. Use only confirmed values given in the prompt.\n"
        "- Do NOT expose internal keys, JSON structure, schema names, or code details.\n"
        "- If you computed a value from user data (E from grade, q from load+area, patch from contact+location), "
        "explain the computation briefly in plain language.\n"
        "- If something is missing or ambiguous, ask ONE focused follow-up question.\n"
        "- If there are validation or feasibility issues, address them before moving on.\n"
        "- Keep responses warm, concise, and practical. Use Markdown sparingly.\n\n"
        "PANEL AWARENESS:\n"
        "- You have full read access to the user's right-side configuration panel. "
        "All collected values are listed under CONFIRMED PARAMETERS in every prompt.\n"
        "- If a user asks 'what have I set?', 'what is my current config?', or 'what's left?', "
        "answer directly from CONFIRMED PARAMETERS and MISSING REQUIRED INPUTS.\n"
        "- You can update any parameter at any time — even after configuration is complete. "
        "Simply acknowledge the new value; the change will be applied automatically.\n"
        "- After applying an edit in complete mode, confirm the specific change and remind the user "
        "to re-run analysis if prior results already exist.\n"
    )


def create_autonomous_response_prompt(
    *,
    memory: Optional[str] = None,
    user_input: str,
    mode: str,
    params: Dict[str, Any],
    applied_keys: List[str],
    inference_notes: List[str],
    conversion_notes: List[str],
    problems: List[str],
    current_asking: Optional[str],
    conversation_only: bool,
) -> str:
    missing = [k for k in PARAM_ORDER if k not in params]
    missing_names = [str(PARAM_INFO[k].get("name", k)) for k in missing if k in PARAM_INFO]

    explain_key = _extract_param_key_from_text(user_input) or current_asking
    explanation_seed = ""
    if conversation_only and explain_key and explain_key in PARAM_INFO:
        # Provide a safe reference block that the model may paraphrase.
        explanation_seed = _parameter_explanation(explain_key)

    segments: List[str] = []

    if memory:
        mem = str(memory).strip()
        if mem:
            if len(mem) > 6000:
                mem = "..." + mem[-6000:]
            segments.append(f"### EARLIER CONVERSATION MEMORY (compacted)\n{mem}\n\n")

    segments.extend([
        f"### USER MESSAGE\n{user_input}\n\n",
        f"### MODE\n{mode}\n\n",
        f"### CONVERSATION-ONLY\n{conversation_only}\n\n",
        f"### CONFIRMED PARAMETERS (current)\n{_render_known_params_for_prompt(params)}\n\n",
        f"### UPDATES CAPTURED THIS TURN\n{_render_applied_for_prompt(params, applied_keys)}\n\n",
        "### UNIT CONVERSIONS (already applied)\n"
        + ("\n".join(f"- {n}" for n in conversion_notes) if conversion_notes else "- (none)")
        + "\n\n",
        "### ENGINEERING INFERENCES (already applied)\n"
        + ("\n".join(f"- {n}" for n in inference_notes) if inference_notes else "- (none)")
        + "\n\n",
        "### VALIDATION ISSUES\n" + ("\n".join(f"- {p}" for p in problems) if problems else "- (none)") + "\n\n",
        "### MISSING REQUIRED INPUTS\n"
        + ("\n".join(f"- {n}" for n in missing_names) if missing_names else "- (none)")
        + "\n\n",
        f"### NEXT QUESTION CONTEXT\n{_next_question_context(current_asking)}\n\n",
    ])

    if explanation_seed:
        segments.append(
            "### REFERENCE EXPLANATION (ground truth; paraphrase this)\n" + explanation_seed + "\n\n"
        )

    segments.append(
        "### TASK\n"
        "Write the assistant reply following these priorities:\n"
        "1. Confirm what was captured this turn (use human-readable names and values, not internal keys).\n"
        "2. If a value was COMPUTED from user data (E from concrete grade, q from load+area, "
        "patch coordinates from contact dims + location), explain that computation in one plain-language line.\n"
        "3. If there are validation or feasibility issues, address the specific issue and ask the user to correct it.\n"
        "4. If something is ambiguous (contact area given but no load location, or load given but no contact dims), "
        "ask ONE focused clarification question.\n"
        "5. If not complete and no issues, ask ONE focused question for the most important missing input.\n"
        "6. If complete:\n"
        "   a. If UPDATES CAPTURED THIS TURN is non-empty → confirm the specific change(s) by name and value, "
        "and note that any prior analysis results should be re-run with the updated inputs.\n"
        "   b. If the user asked about current values or what has been set → summarise the relevant "
        "CONFIRMED PARAMETERS clearly in plain language.\n"
        "   c. Otherwise → confirm all inputs are ready and direct the user to the side panel to run or export analysis.\n\n"
        "PROACTIVE BEHAVIOURS (apply where relevant, don't overdo):\n"
        "- If user provided load (kN) but no contact dims → ask for tyre contact dimensions.\n"
        "- If user gave concrete grade → confirm the computed E value from IS 456.\n"
        "- If user said 'corner' or 'edge' load with a contact area → confirm the computed patch coordinates.\n"
        "- After concrete grade is given, if Poisson's ratio is still missing → suggest IS 456 default of 0.18.\n\n"
        "Do not output JSON.\n"
    )

    return "".join(segments)


def create_vtk_analysis_system_prompt() -> str:
    return (
        "You are an expert rigid pavement analysis assistant. "
        "The user has just completed a Finite Element Analysis of their concrete pavement slab.\n\n"
        "You have full access to the analysis results provided in the prompt under VTK ANALYSIS RESULTS. "
        "Use them to answer any question the user has about the simulation outcome.\n\n"
        "GUIDELINES:\n"
        "- When answering questions about the analysis: cite only values present in VTK ANALYSIS RESULTS "
        "or SLAB CONFIGURATION, and include field names with engineering units.\n"
        "- If the user asks about a result field that is not listed, say it was not available in the output.\n"
        "- When plot images are attached or listed under GENERATED PLOTS, visually analyse them and "
        "embed them in your response using Markdown image syntax: ![description](url).\n"
        "- Keep answers concise and engineering-focused. Use Markdown tables or bullets where helpful.\n"
        "- If the user wants to change a configuration parameter, acknowledge the request and let them "
        "know the side panel or chat can be used to update values and re-run analysis.\n"
        "- If the user asks a general question unrelated to this analysis, answer it briefly and "
        "naturally — you are a helpful assistant, not just a report reader.\n"
    )


def _render_slab_params_for_vtk(params: Dict[str, Any]) -> str:
    labels = {
        "Emod": ("Elastic Modulus (E)", "MPa"),
        "nu": ("Poisson's Ratio", ""),
        "t": ("Slab Thickness", "mm"),
        "Kx": ("Subgrade Kx", ""),
        "Ky": ("Subgrade Ky", ""),
        "Kz": ("Subgrade Kz", ""),
        "a": ("Slab Length", "mm"),
        "b": ("Slab Width", "mm"),
        "q": ("Load Pressure", "MPa"),
        "load_location": ("Load Location", ""),
        "mesh_type": ("Mesh Type", ""),
    }
    lines = []
    for key, (label, unit) in labels.items():
        val = params.get(key)
        if val is not None:
            lines.append(f"- {label}: {val} {unit}".rstrip())
    return "\n".join(lines) if lines else "(no configuration data)"


def _resolve_local_plot_paths(plot_files: List[str]) -> List[Path]:
    """Resolve frontend/backend plot references to local artifact files.

    Supports values like:
    - /artifacts/plot.png
    - http://localhost:8000/artifacts/plot.png
    - absolute filesystem paths
    """
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "output_python_square"
    resolved: List[Path] = []

    for raw in plot_files or []:
        text = (raw or "").strip()
        if not text:
            continue

        candidate: Optional[Path] = None

        if text.startswith("http://") or text.startswith("https://"):
            parsed = urlparse(text)
            basename = Path(parsed.path).name
            if basename:
                candidate = artifacts_dir / basename
        elif text.startswith("/artifacts/"):
            basename = Path(text).name
            if basename:
                candidate = artifacts_dir / basename
        else:
            p = Path(text)
            if p.is_absolute():
                candidate = p
            else:
                candidate = (repo_root / p).resolve()

        if candidate and candidate.exists() and candidate.is_file():
            resolved.append(candidate)

    # Preserve order while removing duplicates.
    unique: List[Path] = []
    seen = set()
    for path in resolved:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def generate_vtk_agent_response(
    *,
    llm: Optional[ChatOllama],
    fallback_llm: Optional[ChatOllama] = None,
    ollama_url: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    user_input: str,
    vtk_stats_text: str,
    slab_params: Dict[str, Any],
    plot_files: List[str],
    fallback_message: str = "",
    timeout_seconds: float = LLM_NARRATION_TIMEOUT_SECONDS,
) -> str:
    """Use the LLM to answer a post-analysis question with full VTK context.

    Tries in priority order:
    1. kimi multimodal (ollama_url + model + images)
    2. kimi text-only (llm)
    3. user's local model (fallback_llm)
    """
    if llm is None and not (ollama_url and model) and fallback_llm is None:
        return fallback_message

    plots_section = ""
    if plot_files:
        plot_lines = [f"![Plot {i+1}]({url})" for i, url in enumerate(plot_files)]
        plots_section = "\n\n### GENERATED PLOTS\n" + "\n".join(plot_lines)

    prompt = (
        f"### USER QUESTION\n{user_input}\n\n"
        f"### SLAB CONFIGURATION\n{_render_slab_params_for_vtk(slab_params)}\n\n"
        f"### VTK ANALYSIS RESULTS\n{vtk_stats_text}"
        f"{plots_section}\n\n"
        "### TASK\n"
        "Answer the user's question using the data above. "
        "If plots are listed and relevant, embed them inline with the Markdown image syntax already provided. "
        "Be concise and engineering-focused. Do not output JSON."
    )

    system_prompt_text = create_vtk_analysis_system_prompt()

    def _try_chat_llm(chat_llm: ChatOllama) -> Optional[str]:
        system = SystemMessage(content=system_prompt_text)
        human = HumanMessage(content=prompt)
        resp = _invoke_llm_with_timeout(chat_llm, [system, human], timeout_seconds=timeout_seconds)
        if resp and getattr(resp, "content", None):
            return _clean_llm_text(str(resp.content)) or None
        return None

    # 1. Primary: kimi multimodal — sends plot images so the model can visually inspect them.
    local_plot_paths = []
    try:
        local_plot_paths = _resolve_local_plot_paths(plot_files)
    except Exception:
        pass

    if ollama_url and model and local_plot_paths:
        try:
            multimodal_text = invoke_ollama_multimodal_chat(
                base_url=ollama_url,
                model=model,
                system_prompt=system_prompt_text,
                user_prompt=prompt,
                image_paths=local_plot_paths,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
            cleaned_multi = _clean_llm_text(multimodal_text)
            if cleaned_multi:
                return cleaned_multi
        except Exception:
            pass

    # 2. Secondary: kimi text-only (no images).
    if llm is not None:
        try:
            result = _try_chat_llm(llm)
            if result:
                return result
        except Exception as exc:
            err_text = str(exc).strip().lower()
            if ("unauthorized" in err_text or "401" in err_text) and fallback_llm is None:
                return (
                    "Couldn't query the model (401 Unauthorized). "
                    "Add an API key in Settings or switch to a local model.\n\n" + fallback_message
                )

    # 3. Tertiary: user's configured local model.
    if fallback_llm is not None:
        try:
            result = _try_chat_llm(fallback_llm)
            if result:
                return result
        except Exception:
            pass

    return fallback_message


def generate_autonomous_assistant_message(
    *,
    llm: Optional[ChatOllama],
    memory: Optional[str] = None,
    user_input: str,
    mode: str,
    params: Dict[str, Any],
    applied_keys: List[str],
    inference_notes: List[str],
    conversion_notes: List[str],
    problems: List[str],
    current_asking: Optional[str],
    conversation_only: bool,
    fallback_message: str,
    timeout_seconds: float = LLM_NARRATION_TIMEOUT_SECONDS,
) -> str:
    """Generate the user-visible assistant message.

    This keeps deterministic extraction/validation/calculation as the source of truth, and uses the LLM
    only to narrate the state (what was captured, inferred, what is missing, and the next question).
    """

    if llm is None:
        return fallback_message

    try:
        system = SystemMessage(content=create_autonomous_response_system_prompt())
        prompt = create_autonomous_response_prompt(
            memory=memory,
            user_input=user_input,
            mode=mode,
            params=params,
            applied_keys=applied_keys,
            inference_notes=inference_notes,
            conversion_notes=conversion_notes,
            problems=problems,
            current_asking=current_asking,
            conversation_only=conversation_only,
        )
        human = HumanMessage(content=prompt)
        response = _invoke_llm_with_timeout(llm, [system, human], timeout_seconds=timeout_seconds)
        if response and getattr(response, "content", None):
            cleaned = _clean_llm_text(str(response.content))
            return cleaned or fallback_message
        return fallback_message
    except TimeoutError:
        return fallback_message
    except Exception as exc:
        err_text = str(exc).strip()
        lowered_err = err_text.lower()

        if "unauthorized" in lowered_err or "status code: 401" in lowered_err:
            return (
                "I couldn’t use the selected Ollama model because the request is unauthorized (401). "
                "If you selected a cloud model, add an API key in Settings (left panel), or switch to a local model.\n\n"
                + fallback_message
            )

        if "cannot reach ollama" in lowered_err or "connection" in lowered_err:
            return (
                "I couldn’t reach your Ollama server. Please ensure Ollama is running and the URL is correct.\n\n"
                + fallback_message
            )

        return fallback_message
