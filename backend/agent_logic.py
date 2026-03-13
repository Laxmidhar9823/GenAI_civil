import json
import re
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

PARAM_CATEGORIES = {
    "Slab Size": ["a", "b", "t"],
    "Material Properties": ["Emod", "nu"],
    "Ground Support": ["Kx", "Ky", "Kz"],
    "Load Position": ["x1", "x2", "y1", "y2"],
    "Load": ["q"],
}

PARAM_ORDER = ["a", "b", "t", "Emod", "nu", "Kx", "Ky", "Kz", "x1", "x2", "y1", "y2", "q"]

USE_ALL_DEFAULTS_PATTERNS = [
    "use all defaults", "fill all defaults", "default all", "skip all",
    "use defaults", "all defaults", "defaults for all", "default for all",
    "default for the rest", "defaults for the rest", "use default for the rest",
    "use defaults for the rest", "default for rest", "defaults for rest",
    "take default for the rest", "take defaults for the rest",
    "take default for rest", "take defaults for rest",
    "fill in the rest", "fill the rest", "fill rest",
    "default for remaining", "defaults for remaining", "use default for remaining",
    "use defaults for remaining", "default remaining", "defaults remaining",
    "take default for remaining", "take defaults for remaining",
    "not sure about remaining", "not sure remaining",
    "default for everything else", "defaults for everything else",
    "use default for everything", "use defaults for everything",
    "default everything else", "defaults everything else",
    "fill in everything else", "fill everything else",
    "skip the rest", "skip rest", "skip remaining", "skip everything",
    "skip all remaining", "skip the remaining",
    "just use defaults", "just defaults", "go with defaults",
    "proceed with defaults", "continue with defaults",
    "put the defaults", "put defaults", "just put defaults",
    "yes use defaults", "yes defaults", "yeah defaults",
    "ok use defaults", "okay defaults", "sure defaults",
    "sure use the defaults", "alright use defaults", "alright defaults",
    "fill with defaults", "fill remaining with defaults", "fill rest with defaults",
    "auto fill", "autofill", "auto-fill",
    "complete with defaults", "complete it with defaults", "finish with defaults",
    "i'm done", "im done", "that's all", "thats all", "that's it", "thats it",
    "nothing else", "no more values", "don't know the rest", "dont know the rest",
    "i don't know", "i dont know", "no idea", "not sure about the rest",
    "no clue", "idk", "have no idea", "only know those",
    "that's all i have", "thats all i have", "that's all the info", "all the info i have",
    "only know these", "only have these", "don't have more", "dont have more",
]


def check_use_all_defaults_intent(user_input: str) -> bool:
    lower_input = user_input.lower().strip()

    # Guardrail: avoid accidental "fill everything with defaults" when the user is merely confused
    # (e.g. "I don't know what Emod is"). We only consider this intent if the message references
    # defaults/rest/remaining/autofill/skip-all semantics.
    if not (
        re.search(r"\bdefault(s)?\b", lower_input)
        or re.search(r"\b(rest|remaining)\b", lower_input)
        or "everything else" in lower_input
        or re.search(r"\bautofill\b|\bauto[- ]fill\b", lower_input)
        or re.search(r"\bskip\b\s+\b(all|everything)\b", lower_input)
    ):
        return False

    for pattern in USE_ALL_DEFAULTS_PATTERNS:
        if pattern in lower_input:
            return True

    flexible_patterns = [
        r"\b(use|take|go with|apply|set|put)\s+(the\s+)?default(s)?(\s+values?)?\s+(for\s+)?(the\s+)?(rest|remaining|everything|all|others?)(\s+(of\s+)?(the\s+)?(parameters?|values?|settings?|fields?))?\b",
        r"\bdefault(s)?(\s+values?)?\s+(for\s+)?(the\s+)?(rest|remaining|everything|all|others?)(\s+(of\s+)?(the\s+)?(parameters?|values?|settings?|fields?))?\b",
        r"\bskip\s+(the\s+)?(rest|remaining|everything|all|others?)\b",
        r"\b(fill|complete|finish)\s+(the\s+)?(rest\s+)?(it\s+)?(with\s+)?default(s)?\b",
        r"\bi\s+(don'?t|do not)\s+(know|have)\s+(the\s+)?(rest|remaining|other)\b",
        r"\bjust\s+(use\s+)?default(s)?\b",
        r"\bfill\s+(in\s+)?(the\s+)?(rest|remaining|everything)\b",
        r"\b(rest|remaining)\s+(with\s+)?default(s)?\b",
        r"\b(yeah|ok|sure|alright)\s+(just\s+)?(fill|use|put)\s+(in\s+)?(the\s+)?(rest|defaults?|everything)\b",
        r"\bi\s+(only\s+)?(know|have)\s+(those|these)\s*(values?)?\b",
        r"\bidk\s+(the\s+)?(rest|remaining)?\b",
        r"\bno\s+clue\s+(about\s+)?(the\s+)?(rest|remaining|others?|these)?\b",
        r"\bnot\s+sure\s+(about\s+)?(the\s+)?(rest|remaining|others?)?\b",
    ]

    return any(re.search(pattern, lower_input) for pattern in flexible_patterns)


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


def format_value_with_unit(value: float, param_key: str) -> str:
    info = PARAM_INFO.get(param_key, {})
    unit = info.get("unit", "")
    if isinstance(value, float) and value == int(value):
        formatted = f"{int(value):,}"
    elif isinstance(value, float):
        formatted = f"{value:,.2f}".rstrip("0").rstrip(".")
    else:
        formatted = str(value)
    if unit and unit != "(no unit)" and unit != "(stiffness factor)":
        return f"{formatted} {unit}"
    return formatted


def validate_single_param(key: str, value: float, all_params: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a single parameter value."""
    info = PARAM_INFO.get(key, {})

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
            ok, msg = validate_single_param(key, float(params[key]), params)
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

    question += f"🔧 **Default value:** {format_value_with_unit(DEFAULT_VALUES[key], key)}\n\n"
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


def create_conversational_system_prompt() -> str:
    return (
        """You are a friendly Pavement Configuration Assistant helping non-technical users set up parameters for rigid pavement analysis.

Your personality:
- Warm, patient, and encouraging
- Explain technical concepts in simple, everyday language
- Use analogies when helpful
- Never use jargon without explanation
- Celebrate progress and reassure users

Your tasks:
1. Help users understand what each parameter means
2. Extract parameter values from natural language
3. Handle unit conversions (meters to mm, etc.)
4. Validate values and explain why if something is wrong
5. Guide users through the configuration process

IMPORTANT RULES FOR RESPONSES:
- Always acknowledge what the user said
- If they provide a value, confirm you understood it
- If there's an error, explain it kindly and suggest a fix
- Keep responses concise but warm
- Use emojis sparingly to be friendly

PARAMETER INFORMATION:
"""
        + json.dumps({k: {**v, "default": DEFAULT_VALUES[k]} for k, v in PARAM_INFO.items()}, indent=2)
    )


def create_extraction_prompt(user_input: str, context: str, current_asking: Optional[str] = None) -> str:
    prompt = f"""### CONTEXT
{context}

### USER INPUT
"{user_input}"

### CURRENTLY ASKING ABOUT
{f"Parameter: {current_asking} ({PARAM_INFO[current_asking]['name']})" if current_asking else "General conversation - extract any parameters mentioned"}

### TASK
1. Determine if the user is providing a parameter value
2. If yes, extract the value and unit (if mentioned)
3. Check if they want to use default (words like "default", "skip", "use default", "that's fine", "ok")
4. Generate a friendly response

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
- Single default: "default", "skip", "ok", "fine", "sure", "yes" = use_default: true (for current parameter only)
- ALL defaults: "default for the rest", "use defaults for remaining", "skip all", "defaults for everything", "I don't know the rest", "that's all I have" = use_all_defaults: true
- If user says they don't know remaining values or want to finish = use_all_defaults: true
- Look for unit mentions: "meters", "m", "cm", "MPa", "kPa", "psi", "feet", "inches"
- If user says "4 meters", extract 4 with unit "m" or "meters"
- Be warm and encouraging!

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


def process_user_input_with_llm(
    user_input: str,
    llm: ChatOllama,
    context: str,
    current_asking: Optional[str] = None,
) -> Dict[str, Any]:
    """Process user input through LLM for extraction and response generation."""

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

        response = llm.invoke([system_message, human_message])

        if response and response.content:
            content = response.content.strip()

            if "```json" in content:
                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*", "", content)
            elif "```" in content:
                content = re.sub(r"```\s*", "", content)

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                content = json_match.group()

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
                return {
                    "understood_value": None,
                    "use_default": False,
                    "use_all_defaults": False,
                    "needs_clarification": True,
                    "friendly_response": "I'm not quite sure I understood that. Could you try again? You can enter a number, or say 'default' to use the suggested value. 😊",
                }

        return {"needs_clarification": True, "friendly_response": "I didn't catch that. Could you please try again?"}

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
