import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

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
        "typical_range": "0.15 - 0.2 for concrete",
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

LLM_INVOKE_TIMEOUT_SECONDS = 8.0

USE_ALL_DEFAULTS_PATTERNS = [
    "use defaults",
    "use all defaults",
    "defaults for remaining",
    "defaults for the rest",
    "use defaults for remaining",
    "use defaults for the rest",
    "fill remaining with defaults",
    "fill rest with defaults",
    "skip the rest",
    "skip remaining",
    "i don't know the rest",
    "thats all i have",
]

SINGLE_DEFAULT_PATTERNS = {
    "default",
    "skip",
    "use default",
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
    return None


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
    location = normalize_load_location(load_location) or "interior"
    slab_a = float(a)
    slab_b = float(b)
    lx = max(1.0, float(contact_lx_mm))
    ly = max(1.0, float(contact_ly_mm))
    lowered = (context_text or "").lower()

    if location == "corner":
        return {
            "x1": 0.0,
            "x2": round(min(slab_a, lx), 3),
            "y1": 0.0,
            "y2": round(min(slab_b, ly), 3),
        }

    if location == "edge":
        if "right edge" in lowered or "at right edge" in lowered or "on right edge" in lowered:
            y_center = slab_b / 2.0
            y1 = max(0.0, y_center - (ly / 2.0))
            y2 = min(slab_b, y_center + (ly / 2.0))
            return {
                "x1": round(max(0.0, slab_a - lx), 3),
                "x2": round(slab_a, 3),
                "y1": round(y1, 3),
                "y2": round(y2, 3),
            }

        if "left edge" in lowered or "at left edge" in lowered or "on left edge" in lowered:
            y_center = slab_b / 2.0
            y1 = max(0.0, y_center - (ly / 2.0))
            y2 = min(slab_b, y_center + (ly / 2.0))
            return {
                "x1": 0.0,
                "x2": round(min(slab_a, lx), 3),
                "y1": round(y1, 3),
                "y2": round(y2, 3),
            }

        if "top edge" in lowered or "at top edge" in lowered or "on top edge" in lowered:
            x_center = slab_a / 2.0
            x1 = max(0.0, x_center - (lx / 2.0))
            x2 = min(slab_a, x_center + (lx / 2.0))
            return {
                "x1": round(x1, 3),
                "x2": round(x2, 3),
                "y1": round(max(0.0, slab_b - ly), 3),
                "y2": round(slab_b, 3),
            }

        x_center = slab_a / 2.0
        x1 = max(0.0, x_center - (lx / 2.0))
        return {
            "x1": round(x1, 3),
            "x2": round(min(slab_a, x1 + lx), 3),
            "y1": 0.0,
            "y2": round(min(slab_b, ly), 3),
        }

    x1 = max(0.0, (slab_a - lx) / 2.0)
    y1 = max(0.0, (slab_b - ly) / 2.0)
    return {
        "x1": round(x1, 3),
        "x2": round(min(slab_a, x1 + lx), 3),
        "y1": round(y1, 3),
        "y2": round(min(slab_b, y1 + ly), 3),
    }


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
    mesh_type: str,
) -> List[Dict[str, float]]:
    location = normalize_load_location(load_location) or "interior"
    distances = _extract_edge_distances_mm(text)
    edge_lock_x = False
    edge_lock_y = False

    cx = slab_a / 2.0
    cy = slab_b / 2.0

    if "left" in distances:
        cx = distances["left"]
    if "right" in distances:
        cx = slab_a - distances["right"]
    if "bottom" in distances:
        cy = distances["bottom"]
    if "top" in distances:
        cy = slab_b - distances["top"]

    explicit_x_distance = "left" in distances or "right" in distances
    explicit_y_distance = "bottom" in distances or "top" in distances

    if location == "edge":
        if "bottom" in text and "bottom" not in distances:
            cy = contact_ly_mm / 2.0
            edge_lock_y = True
        elif "top" in text and "top" not in distances:
            cy = slab_b - (contact_ly_mm / 2.0)
            edge_lock_y = True
        elif "left" in text and "left" not in distances:
            cx = contact_lx_mm / 2.0
            edge_lock_x = True
        elif "right" in text and "right" not in distances:
            cx = slab_a - (contact_lx_mm / 2.0)
            edge_lock_x = True
        elif all(side not in text for side in ["top", "bottom", "left", "right"]):
            cy = contact_ly_mm / 2.0
            edge_lock_y = True

    if location == "corner":
        if "top" in text and "right" in text:
            cx = slab_a - (contact_lx_mm / 2.0)
            cy = slab_b - (contact_ly_mm / 2.0)
        elif "top" in text:
            cx = contact_lx_mm / 2.0
            cy = slab_b - (contact_ly_mm / 2.0)
        elif "right" in text:
            cx = slab_a - (contact_lx_mm / 2.0)
            cy = contact_ly_mm / 2.0
        else:
            cx = contact_lx_mm / 2.0
            cy = contact_ly_mm / 2.0

    axis = "y"
    if any(k in distances for k in ["left", "right"]) or ("left edge" in text or "right edge" in text):
        axis = "y"
    elif any(k in distances for k in ["top", "bottom"]) or ("top edge" in text or "bottom edge" in text):
        axis = "x"
    elif location == "edge":
        axis = "x"

    level = normalize_mesh_type(mesh_type) or "coarse"
    elements = max(1, int(MESH_LEVEL_ELEMENTS.get(level, MESH_LEVEL_ELEMENTS["coarse"])))
    mesh_dx = float(slab_a) / elements
    mesh_dy = float(slab_b) / elements

    contact_x = float(contact_lx_mm)
    contact_y = float(contact_ly_mm)
    if level == "fine":
        # Fine mesh guardrail: contact patch should be at least one mesh step.
        contact_x = max(contact_x, mesh_dx)
        contact_y = max(contact_y, mesh_dy)

    half_x = contact_x / 2.0
    half_y = contact_y / 2.0

    wheels = max(1, int(number_of_wheels))
    spacing = max(0.0, float(spacing_mm))
    offsets = [((i - (wheels - 1) / 2.0) * spacing) for i in range(wheels)]

    # Corner boundary condition takes highest priority over mesh alignment.
    # For corner loading, keep first wheel exactly touching both edges.
    if location == "corner":
        first_cx = half_x
        first_cy = half_y

        if wheels == 1:
            patch = _center_to_patch(first_cx, first_cy, contact_x, contact_y)
            return [
                {
                    "x1": float(patch["x1"]),
                    "x2": float(patch["x2"]),
                    "y1": float(patch["y1"]),
                    "y2": float(patch["y2"]),
                    "q": float(q_value),
                }
            ]

        centers: List[Tuple[float, float]] = [(first_cx, first_cy)]

        # Dual/tridem extension from first wheel uses center spacing (not rectangle spacing).
        # Prefer x-direction first; if not feasible, extend in y-direction.
        def _can_place_x(index: int) -> bool:
            cx_try = first_cx + (index * spacing)
            return (cx_try + half_x) <= slab_a

        def _can_place_y(index: int) -> bool:
            cy_try = first_cy + (index * spacing)
            return (cy_try + half_y) <= slab_b

        use_x_axis = _can_place_x(wheels - 1)
        if not use_x_axis and not _can_place_y(wheels - 1):
            # If spacing is too large for both directions, place as far as possible in x while
            # preserving first wheel boundary condition.
            use_x_axis = True

        for i in range(1, wheels):
            if use_x_axis:
                cx_i = first_cx + (i * spacing)
                cx_i = min(max(cx_i, half_x), slab_a - half_x)
                centers.append((cx_i, first_cy))
            else:
                cy_i = first_cy + (i * spacing)
                cy_i = min(max(cy_i, half_y), slab_b - half_y)
                centers.append((first_cx, cy_i))

        load_cases: List[Dict[str, float]] = []
        for cxi, cyi in centers:
            patch = _center_to_patch(cxi, cyi, contact_x, contact_y)
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

    # Preserve spacing exactly by constraining and snapping only the group center.
    if axis == "x":
        min_off = min(offsets)
        max_off = max(offsets)
        min_cx = half_x - min_off
        max_cx = slab_a - half_x - max_off
        min_cy = half_y
        max_cy = slab_b - half_y
    else:
        min_off = min(offsets)
        max_off = max(offsets)
        min_cx = half_x
        max_cx = slab_a - half_x
        min_cy = half_y - min_off
        max_cy = slab_b - half_y - max_off

    # If spacing is too large for slab, collapse to nearest feasible center while keeping wheel offsets fixed.
    if min_cx > max_cx:
        mid = slab_a / 2.0
        min_cx = mid
        max_cx = mid
    if min_cy > max_cy:
        mid = slab_b / 2.0
        min_cy = mid
        max_cy = mid

    cx = min(max(cx, min_cx), max_cx)
    cy = min(max(cy, min_cy), max_cy)

    x_nodes = _linspace_nodes(slab_a, elements)
    y_nodes = _linspace_nodes(slab_b, elements)
    snapped_cx = _nearest_mesh_point(cx, x_nodes)
    snapped_cy = _nearest_mesh_point(cy, y_nodes)

    # Preserve user-specified interior distances first; only snap if mesh deviation is small.
    if explicit_x_distance and abs(snapped_cx - cx) >= (mesh_dx / 2.0):
        snapped_cx = cx
    if explicit_y_distance and abs(snapped_cy - cy) >= (mesh_dy / 2.0):
        snapped_cy = cy

    # Boundary condition priority: edge-constrained axis should not move to mesh nodes.
    if edge_lock_x:
        snapped_cx = cx
    if edge_lock_y:
        snapped_cy = cy

    # Keep spacing priority: after snap, shift center only as needed to remain inside boundaries.
    cx = min(max(snapped_cx, min_cx), max_cx)
    cy = min(max(snapped_cy, min_cy), max_cy)

    load_cases: List[Dict[str, float]] = []
    for offset in offsets:
        wheel_cx = cx + offset if axis == "x" else cx
        wheel_cy = cy + offset if axis == "y" else cy

        patch = _center_to_patch(wheel_cx, wheel_cy, contact_x, contact_y)
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
        location = normalize_load_location(params[LOAD_LOCATION_KEY]) or "interior"
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


def check_use_all_defaults_intent(user_input: str) -> bool:
    lower_input = user_input.lower().strip()
    compact = " ".join(lower_input.split())
    return any(pattern in compact for pattern in USE_ALL_DEFAULTS_PATTERNS)


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
    if key == "nu" and (value < 0.15 or value > 0.2):
        return False, "For concrete pavements, use a Poisson's ratio between 0.15 and 0.2."
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


def build_conversation_context(messages: List[Dict], params: Dict[str, Any]) -> str:
    context_parts = []
    if params:
        context_parts.append("Current parameters collected:")
        for key, value in params.items():
            info = PARAM_INFO.get(key, {})
            context_parts.append(f"- {info.get('name', key)}: {format_value_with_unit(value, key)}")

    recent_messages = messages[-6:] if len(messages) > 6 else messages
    if recent_messages:
        context_parts.append("\nRecent conversation:")
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            context_parts.append(f"{role}: {content}")
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
    }
    return float(value) * multipliers.get(unit_lower, 1.0)


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

    load_match = re.search(
        r"(?:each\s*wheel|per\s*wheel|wheel\s*load|(?:wheel\s*)?load)\s*[:=-]?\s*[^\d]{0,20}(\d+(?:\.\d+)?)\s*(kn|n)",
        lowered,
    )

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

    spacing_mm = _extract_spacing_mm(lowered)
    tandem_requested = bool(
        re.search(r"\btandem\s*axle\b|\btandem\b|\bdual\s*wheel\b|\bdouble\s*wheel\b", lowered)
    )
    number_of_wheels = _detect_number_of_wheels(lowered)

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
            load_n = load_value * 1000.0 if load_unit == "kn" else load_value
            area_mm2 = max(area_lx_mm * area_ly_mm, 1e-9)
            q_val = round(load_n / area_mm2, 5)
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
        load_n = load_value * 1000.0 if load_unit == "kn" else load_value
        area_mm2 = max(area_lx_mm * area_ly_mm, 1e-9)
        q_val = round(load_n / area_mm2, 5)

        group_location = inferred_location or (normalize_load_location(out.get("load_location")) if isinstance(out.get("load_location"), str) else None) or "interior"
        spacing_for_group = spacing_mm if isinstance(spacing_mm, (int, float)) else float(area_lx_mm)
        mesh_for_group = infer_semantic_choices(text).get(MESH_TYPE_KEY) or (str(out.get(MESH_TYPE_KEY)) if MESH_TYPE_KEY in out else "coarse")
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
            mesh_type=mesh_for_group,
        )

    if load_cases:
        out["load_cases"] = load_cases
        out.update(load_cases[0])
    elif tandem_requested and inferred_location and load_match and area_lx_mm is not None and area_ly_mm is not None:
        load_value = float(load_match.group(1))
        load_unit = (load_match.group(2) or "n").lower()
        load_n = load_value * 1000.0 if load_unit == "kn" else load_value
        area_mm2 = max(area_lx_mm * area_ly_mm, 1e-9)
        q_val = round(load_n / area_mm2, 5)

        base_patch = infer_load_patch_from_contact_area(
            inferred_location,
            slab_a,
            slab_b,
            area_lx_mm,
            area_ly_mm,
            lowered,
        )
        second_patch = _offset_load_patch(
            base_patch,
            spacing_mm if isinstance(spacing_mm, (int, float)) else float(area_lx_mm),
            slab_a,
            slab_b,
        )
        load_cases = [
            {
                "x1": float(base_patch["x1"]),
                "x2": float(base_patch["x2"]),
                "y1": float(base_patch["y1"]),
                "y2": float(base_patch["y2"]),
                "q": float(q_val),
            },
            {
                "x1": float(second_patch["x1"]),
                "x2": float(second_patch["x2"]),
                "y1": float(second_patch["y1"]),
                "y2": float(second_patch["y2"]),
                "q": float(q_val),
            },
        ]
        out["load_cases"] = load_cases
        out.update(load_cases[0])
    elif inferred_location and area_lx_mm is not None and area_ly_mm is not None:
        patch = infer_load_patch_from_contact_area(
            inferred_location,
            slab_a,
            slab_b,
            area_lx_mm,
            area_ly_mm,
            lowered,
        )
        out.update(patch)

    explicit_q = re.search(r"(?:pressure|\bq\b)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mpa|kpa|psi)?", lowered)
    if explicit_q:
        out["q"] = float(explicit_q.group(1))
        if explicit_q.group(2):
            units["q"] = explicit_q.group(2)
    elif load_match and area_lx_mm is not None and area_ly_mm is not None:
        load_value = float(load_match.group(1))
        load_unit = (load_match.group(2) or "n").lower()
        load_n = load_value * 1000.0 if load_unit == "kn" else load_value

        area_mm2 = max(area_lx_mm * area_ly_mm, 1e-9)
        out["q"] = round(load_n / area_mm2, 5)

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
        "### 📊 Progress",
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
    if "### 📊 Progress" in message:
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


def create_conversational_system_prompt() -> str:
    parameter_catalog = {k: {**v, "default": get_default_value(k)} for k, v in PARAM_INFO.items()}

    return (
    """You are a production-grade Pavement Configuration Assistant for non-technical users.

PRIMARY GOAL:
Help a beginner provide all required inputs through a natural, supportive conversation.

PERSONA AND TONE:
- Warm, calm, encouraging, and human.
- Explain in plain language first, technical term second (if needed).
- Never sound robotic or command-like.
- Proactively guide with short suggestions and examples.

BEHAVIOR RULES:
- Never expose internal JSON structure, keys, or implementation details to users.
- Treat user messages as conversation, not form fields.
- If the user gives multiple values, extract all relevant ones.
- If the user corrects something, trust the latest value.
- If user is unsure, offer defaults kindly.
- If ambiguous, ask one focused clarification question.

EXTRACTION RULES:
- Extract numbers and units from free text.
- Recognize simple confirmations for defaults on the current field.
- Recognize requests to use defaults for all remaining fields.
- If a value is given but the key is implicit, infer from current question context.

FRIENDLY RESPONSE STYLE:
- Start with a brief acknowledgement.
- Confirm what you understood.
- If useful, include one concise real-world hint.
- Keep it compact and clear.

PARAMETER CATALOG:
"""
    + json.dumps(parameter_catalog, indent=2)
    + "\n\nNODE PARAMETER CATALOG (always included in final JSON):\n"
    + json.dumps(NODE_PARAM_INFO, indent=2)
    + "\n\nNODE DEFAULT VALUES:\n"
    + json.dumps(NODE_DEFAULT_VALUES, indent=2)
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
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"LLM request timed out after {timeout_seconds:.1f}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


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

    if current_asking and stripped_input in SINGLE_DEFAULT_PATTERNS:
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
        friendly = "Great details, thanks. I captured multiple values together."
        if semantic_choices and len(extracted_multiple) == len(semantic_choices):
            friendly = "Perfect, I captured that classification and will infer the detailed coordinates automatically."

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

        return {
            "understood_value": understood_value,
            "use_default": False,
            "use_all_defaults": False,
            "needs_clarification": False,
            "friendly_response": friendly,
            "original_unit": original_unit,
            "parameter_key": current_asking,
            "extracted_multiple": extracted_multiple,
            "extracted_units": extracted_units,
        }

    if check_use_all_defaults_intent(user_input):
        return {
            "understood_value": None,
            "use_default": False,
            "use_all_defaults": True,
            "needs_clarification": False,
            "friendly_response": "Got it! I'll use the default values for all remaining parameters.",
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
        system_message = SystemMessage(content=create_conversational_system_prompt())
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
    return """# 👋 Hello! Welcome to the Pavement Configuration Assistant!

I'm here to help you set up parameters for analyzing a concrete road pavement. Don't worry if you're not a technical expert - I'll explain everything in simple terms!

### 🎯 What we're doing:
We're going to configure a few settings for your concrete slab (the road surface). I'll ask you about:
- **The size of the slab** (how big it is)
- **The mesh density** (coarse / medium / fine)
- **The concrete properties** (how strong it is)  
- **The ground underneath** (how supportive the soil is)
- **Where the load acts** (corner / edge / interior)

### 💡 How this works:
- I'll ask you one question at a time
- For each question, I'll explain what it means in plain English
- You can give me a value, or just say **\"default\"** to use the suggested value
- If you know multiple values already, feel free to tell me all at once!

### 📏 Base Units:
Please enter values in these units.
- **Elastic Modulus (E):** MPa
- **Poisson's Ratio (ν):** dimensionless (no unit)
- **Slab Length (a):** mm
- **Slab Width (b):** mm
- **Slab Thickness (t):** mm
- **Soil Stiffness (Kx, Ky, Kz):** no unit (stiffness factor)
- **Load coordinates (x1, x2, y1, y2):** mm
- **Tire pressure (q):** MPa
- **Node coordinates (x, y):** mm

---

**Ready to start?** Just tell me about your pavement, or type **\"let's begin\"** and I'll guide you step by step! 

*Example: \"I have a 4 meter square slab, medium mesh, load at edge\" or \"let's begin\"*

---

### 📊 Progress
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
