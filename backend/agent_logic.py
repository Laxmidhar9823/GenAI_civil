import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

DEFAULT_VALUES = {
    "Emod": 24000.0,
    "nu": 0.3,
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
}

NODE_DEFAULT_VALUES = {
    "x": [0.0, 1500.0, 3500.0],
    "y": [0.0, 1500.0, 3500.0],
}

NODE_PARAM_INFO = {
    "x": {
        "name": "Node Coordinates (X)",
        "technical_name": "Mesh Node Locations in X-direction",
        "unit": "mm",
        "category": "Mesh / Nodes",
        "simple_explanation": "Reference mesh coordinates along slab length used by the solver grid.",
        "typical_range": "0 to slab length",
        "example": "[0, 1500, 3500]",
        "icon": "🧭",
        "default": [0.0, 1500.0, 3500.0],
    },
    "y": {
        "name": "Node Coordinates (Y)",
        "technical_name": "Mesh Node Locations in Y-direction",
        "unit": "mm",
        "category": "Mesh / Nodes",
        "simple_explanation": "Reference mesh coordinates across slab width used by the solver grid.",
        "typical_range": "0 to slab width",
        "example": "[0, 1500, 3500]",
        "icon": "🧭",
        "default": [0.0, 1500.0, 3500.0],
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
        "typical_range": "0.15 - 0.25 for concrete",
        "example": "For most concrete, use 0.15 to 0.2",
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
    "q": {
        "name": "Tire Pressure on Ground",
        "technical_name": "Tyre Contact Pressure",
        "unit": "MPa",
        "alt_units": {"kpa": 0.001, "psi": 0.00689476},
        "category": "Load",
        "simple_explanation": "How hard the tire presses on the concrete. This depends on vehicle weight and tire size.",
        "typical_range": "0.1 - 0.7 MPa (maximum 0.7)",
        "example": "Car: ~0.2 MPa, Truck: ~0.5-0.7 MPa",
        "icon": "🛞",
        "max_value": 0.7,
    },
}

PARAM_INFO.update(NODE_PARAM_INFO)

NODE_PARAM_KEYS = {"x", "y"}


def get_default_value(key: str) -> Any:
    if key in NODE_PARAM_KEYS:
        return list(NODE_DEFAULT_VALUES[key])
    return DEFAULT_VALUES[key]

PARAM_CATEGORIES = {
    "Mesh / Nodes": ["x", "y"],
    "Slab Size": ["a", "b", "t"],
    "Material Properties": ["Emod", "nu"],
    "Ground Support": ["Kx", "Ky", "Kz"],
    "Load Position": ["x1", "x2", "y1", "y2"],
    "Load": ["q"],
}

PARAM_ORDER = ["a", "b", "x", "y", "t", "Emod", "nu", "Kx", "Ky", "Kz", "x1", "x2", "y1", "y2", "q"]

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
    "fine",
    "ok",
    "okay",
    "yes",
    "sure",
}

PARAM_ALIASES = {
    "a": ["a", "length", "slab length", "x length", "x dimension"],
    "b": ["b", "width", "slab width", "y length", "y dimension"],
    "t": ["t", "thickness", "slab thickness", "depth"],
    "Emod": ["emod", "modulus", "modulus of elasticity", "elasticity"],
    "nu": ["nu", "poisson", "poissons ratio", "poisson ratio"],
    "Kx": ["kx", "stiffness x", "foundation x"],
    "Ky": ["ky", "stiffness y", "foundation y"],
    "Kz": ["kz", "stiffness z", "foundation z"],
    "x1": ["x1", "load start x", "start x"],
    "x2": ["x2", "load end x", "end x"],
    "y1": ["y1", "load start y", "start y"],
    "y2": ["y2", "load end y", "end y"],
    "q": ["q", "pressure", "tyre pressure", "tire pressure", "contact pressure"],
    "x": ["node x", "x nodes", "x coordinates", "mesh x", "grid x"],
    "y": ["node y", "y nodes", "y coordinates", "mesh y", "grid y"],
}


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

    if not isinstance(value, (int, float)):
        return False, "Please provide a numeric value."

    if "max_value" in info and value > info["max_value"]:
        return False, f"This value is too high! The maximum allowed is {info['max_value']} {info.get('unit', '')}."
    if key in ["Emod", "a", "b", "t", "Kx", "Ky", "Kz"] and value <= 0:
        return False, "This value must be greater than zero."
    if key == "nu" and (value < 0 or value > 0.5):
        return False, "This value should be between 0 and 0.5."
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
    for key in PARAM_ORDER:
        if key in params:
            ok, msg = validate_single_param(key, params[key], params)
            if not ok:
                return key, msg
    return None


def get_friendly_param_question(key: str, params: Dict[str, Any]) -> str:
    info = PARAM_INFO[key]
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


def _parse_numeric_token(token: str) -> Optional[float]:
    if token in {"", ".", "-", "-."}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


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


def parse_multi_param_candidates(text: str) -> Dict[str, Tuple[float, Optional[str]]]:
    tokens = _tokenize_for_numbers(text)
    if not tokens:
        return {}

    numbers: List[Tuple[int, float, Optional[str]]] = []
    for i, tok in enumerate(tokens):
        val = _parse_numeric_token(tok)
        if val is None:
            continue
        unit = tokens[i + 1] if i + 1 < len(tokens) and tokens[i + 1].isalpha() else None
        numbers.append((i, val, unit))

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
            "Tip: You can reply like \"length 4.5 m, width 3.5 m, thickness 220 mm\".",
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
- Ask the user for node X and Y coordinates like other parameters.
- If user does not provide them, use defaults safely.
- Node X defaults: {NODE_DEFAULT_VALUES['x']}
- Node Y defaults: {NODE_DEFAULT_VALUES['y']}

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
    "understood_value": <number or null if not providing a value>,
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
    current_asking: Optional[str] = None,
) -> Dict[str, Any]:
    """Process user input through LLM for extraction and response generation."""

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

    heuristic_candidates = parse_multi_param_candidates(user_input)
    node_list_candidates = _parse_node_list_candidates(user_input)
    if len(heuristic_candidates) >= 2 or node_list_candidates:
        extracted_multiple: Dict[str, Any] = {}
        for key, (value, _) in heuristic_candidates.items():
            extracted_multiple[key] = value
        for key, values in node_list_candidates.items():
            extracted_multiple[key] = values
        return {
            "understood_value": None,
            "use_default": False,
            "use_all_defaults": False,
            "needs_clarification": False,
            "friendly_response": "Great details, thanks. I captured multiple values together.",
            "original_unit": None,
            "parameter_key": current_asking,
            "extracted_multiple": extracted_multiple,
            "extracted_units": {k: u for k, (_, u) in heuristic_candidates.items() if u},
        }

    if check_use_all_defaults_intent(user_input):
        return {
            "understood_value": None,
            "use_default": False,
            "use_all_defaults": True,
            "needs_clarification": False,
            "friendly_response": "Got it! I'll use the default values for all remaining parameters.",
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
            "friendly_response": "The model is taking too long to respond right now. Please try again, use 'let's begin' for guided mode, or provide multiple numeric values in one message.",
        }
    except Exception:
        return {
            "needs_clarification": True,
            "friendly_response": "I had a small hiccup processing that. Could you try again? 😊",
        }


def generate_welcome_message() -> str:
    """Generate a warm welcome message."""
    return """# 👋 Hello! Welcome to the Pavement Configuration Assistant!

I'm here to help you set up parameters for analyzing a concrete road pavement. Don't worry if you're not a technical expert - I'll explain everything in simple terms!

### 🎯 What we're doing:
We're going to configure a few settings for your concrete slab (the road surface). I'll ask you about:
- **The size of the slab** (how big it is)
- **The concrete properties** (how strong it is)  
- **The ground underneath** (how supportive the soil is)
- **Where the vehicle load is** (where tires press on the road)

### 💡 How this works:
- I'll ask you one question at a time
- For each question, I'll explain what it means in plain English
- You can give me a value, or just say **\"default\"** to use the suggested value
- If you know multiple values already, feel free to tell me all at once!

---

**Ready to start?** Just tell me about your pavement, or type **\"let's begin\"** and I'll guide you step by step! 

*Example: \"I have a 4 meter square slab\" or \"let's begin\"*

---

### 📊 Progress
- Completed: **0/15 (0%)**
- Remaining: **15**
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
            value = params[key]
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
            "x1": [float(merged["x1"])],
            "x2": [float(merged["x2"])],
            "y1": [float(merged["y1"])],
            "y2": [float(merged["y2"])],
            "q": [float(merged["q"])],
        },
    }
