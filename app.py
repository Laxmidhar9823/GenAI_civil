import streamlit as st
import json
import re
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# =============================================================================
# DEFAULT VALUES AND CONFIGURATIONS
# =============================================================================

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
    "q": 0.1
}

AVAILABLE_MODELS = ["llama3.1:8b", "deepseek-r1:8b"]

# =============================================================================
# LAYMAN-FRIENDLY PARAMETER DESCRIPTIONS
# =============================================================================

PARAM_INFO = {
    "Emod": {
        "name": "Concrete Stiffness",
        "technical_name": "Modulus of Elasticity",
        "unit": "MPa",
        "category": "Material Properties",
        "simple_explanation": "How stiff or rigid the concrete is. Higher values mean stiffer concrete that bends less under load.",
        "typical_range": "20,000 - 35,000 MPa for concrete pavements",
        "example": "For normal concrete roads, use around 24,000-30,000 MPa",
        "icon": "🧱"
    },
    "nu": {
        "name": "Concrete Flexibility Ratio",
        "technical_name": "Poisson's Ratio",
        "unit": "(no unit)",
        "category": "Material Properties",
        "simple_explanation": "How much the concrete squeezes sideways when pressed down. It's always between 0 and 0.5.",
        "typical_range": "0.15 - 0.25 for concrete",
        "example": "For most concrete, use 0.15 to 0.2",
        "icon": "📐"
    },
    "a": {
        "name": "Slab Length",
        "technical_name": "Slab Dimension (X-direction)",
        "unit": "mm",
        "alt_units": {"m": 1000, "meter": 1000, "meters": 1000, "cm": 10, "ft": 304.8, "feet": 304.8, "inch": 25.4, "inches": 25.4},
        "category": "Slab Size",
        "simple_explanation": "The length of your concrete slab (one side of the rectangle).",
        "typical_range": "3,000 - 6,000 mm (3 to 6 meters)",
        "example": "A typical road slab is about 4.5 meters (4500 mm) long",
        "icon": "📏"
    },
    "b": {
        "name": "Slab Width",
        "technical_name": "Slab Dimension (Y-direction)",
        "unit": "mm",
        "alt_units": {"m": 1000, "meter": 1000, "meters": 1000, "cm": 10, "ft": 304.8, "feet": 304.8, "inch": 25.4, "inches": 25.4},
        "category": "Slab Size",
        "simple_explanation": "The width of your concrete slab (the other side of the rectangle).",
        "typical_range": "3,000 - 4,500 mm (3 to 4.5 meters)",
        "example": "A typical road lane is about 3.5 meters (3500 mm) wide",
        "icon": "📐"
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
        "icon": "📊"
    },
    "Kx": {
        "name": "Ground Support (Horizontal)",
        "technical_name": "Foundation Stiffness (X)",
        "unit": "(stiffness factor)",
        "category": "Ground Support",
        "simple_explanation": "How well the ground underneath supports the slab horizontally. Higher means stronger support.",
        "typical_range": "30 - 100",
        "example": "Soft soil: 30-50, Medium soil: 50-70, Hard soil: 70-100",
        "icon": "🌍"
    },
    "Ky": {
        "name": "Ground Support (Sideways)",
        "technical_name": "Foundation Stiffness (Y)",
        "unit": "(stiffness factor)",
        "category": "Ground Support",
        "simple_explanation": "How well the ground underneath supports the slab sideways.",
        "typical_range": "30 - 100",
        "example": "Usually same as horizontal support (Kx)",
        "icon": "🌍"
    },
    "Kz": {
        "name": "Ground Support (Vertical)",
        "technical_name": "Foundation Stiffness (Z)",
        "unit": "(stiffness factor)",
        "category": "Ground Support",
        "simple_explanation": "How well the ground underneath pushes back when the slab is pressed down.",
        "typical_range": "30 - 100",
        "example": "Usually same as other directions",
        "icon": "🌍"
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
        "icon": "🚗"
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
        "icon": "🚗"
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
        "icon": "🚗"
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
        "icon": "🚗"
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
        "max_value": 0.7
    }
}

# =============================================================================
# UI-ONLY DISPLAY LABEL OVERRIDES (do not change keys or logic)
# =============================================================================

DISPLAY_LABEL_OVERRIDES = {
    "nu": "Poisson's Ratio (ν)",
    "Emod": "Elastic Modulus (E)",
    "Kx": "Soil Stiffness (Horizontal)",
    "Ky": "Soil Stiffness (Transverse)",
    "Kz": "Soil Stiffness (Vertical)",
}


def get_param_display_name(param_key: str) -> str:
    """Return UI display label for a parameter key (UI-only)."""
    if param_key in DISPLAY_LABEL_OVERRIDES:
        return DISPLAY_LABEL_OVERRIDES[param_key]
    info = PARAM_INFO.get(param_key, {})
    return info.get("name", param_key)

# Group parameters by category for guided flow
PARAM_CATEGORIES = {
    "Slab Size": ["a", "b", "t"],
    "Material Properties": ["Emod", "nu"],
    "Ground Support": ["Kx", "Ky", "Kz"],
    "Load Position": ["x1", "x2", "y1", "y2"],
    "Load": ["q"]
}

# Order for asking parameters (most intuitive order for layman)
PARAM_ORDER = ["a", "b", "t", "Emod", "nu", "Kx", "Ky", "Kz", "x1", "x2", "y1", "y2", "q"]

# Patterns to detect "use all defaults" intent
USE_ALL_DEFAULTS_PATTERNS = [
    # Exact matches
    "use all defaults", "fill all defaults", "default all", "skip all",
    "use defaults", "all defaults", "defaults for all", "default for all",
    # "Rest" patterns
    "default for the rest", "defaults for the rest", "use default for the rest",
    "use defaults for the rest", "default for rest", "defaults for rest",
    "take default for the rest", "take defaults for the rest",
    "take default for rest", "take defaults for rest",
    "fill in the rest", "fill the rest", "fill rest",
    # "Remaining" patterns  
    "default for remaining", "defaults for remaining", "use default for remaining",
    "use defaults for remaining", "default remaining", "defaults remaining",
    "take default for remaining", "take defaults for remaining",
    "not sure about remaining", "not sure remaining",
    # "Everything else" patterns
    "default for everything else", "defaults for everything else",
    "use default for everything", "use defaults for everything",
    "default everything else", "defaults everything else",
    "fill in everything else", "fill everything else",
    # "Skip" patterns
    "skip the rest", "skip rest", "skip remaining", "skip everything",
    "skip all remaining", "skip the remaining",
    # Short forms
    "just use defaults", "just defaults", "go with defaults",
    "proceed with defaults", "continue with defaults",
    "put the defaults", "put defaults", "just put defaults",
    # Confirmations that mean "use defaults for all"
    "yes use defaults", "yes defaults", "yeah defaults",
    "ok use defaults", "okay defaults", "sure defaults",
    "sure use the defaults", "alright use defaults", "alright defaults",
    # Fill patterns
    "fill with defaults", "fill remaining with defaults", "fill rest with defaults",
    "auto fill", "autofill", "auto-fill",
    "complete with defaults", "complete it with defaults", "finish with defaults",
    # Done patterns
    "i'm done", "im done", "that's all", "thats all", "that's it", "thats it",
    "nothing else", "no more values", "don't know the rest", "dont know the rest",
    "i don't know", "i dont know", "no idea", "not sure about the rest",
    # Uncertainty patterns
    "no clue", "idk", "have no idea", "only know those",
    "that's all i have", "thats all i have", "that's all the info", "all the info i have",
    "only know these", "only have these", "don't have more", "dont have more",
]


def check_use_all_defaults_intent(user_input: str) -> bool:
    """Check if user wants to use defaults for all remaining parameters."""
    lower_input = user_input.lower().strip()
    
    # Check exact and substring matches
    for pattern in USE_ALL_DEFAULTS_PATTERNS:
        if pattern in lower_input:
            return True
    
    # Check with regex for more flexible matching
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
    
    for pattern in flexible_patterns:
        if re.search(pattern, lower_input):
            return True
    
    return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ollama_llm(base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.3,
        num_predict=1000,
    )


def check_ollama_connection(base_url: str = "http://localhost:11434", model_name: str = "llama3.1:8b") -> Tuple[bool, bool, list]:
    import requests
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


def convert_to_standard_unit(value: float, from_unit: str, param_key: str) -> Tuple[float, str]:
    """Convert a value from alternative units to standard unit (mm or MPa)."""
    info = PARAM_INFO.get(param_key, {})
    alt_units = info.get("alt_units", {})
    standard_unit = info.get("unit", "")
    
    from_unit_lower = from_unit.lower().strip()
    
    # Check if conversion needed
    if from_unit_lower in alt_units:
        multiplier = alt_units[from_unit_lower]
        converted_value = value * multiplier
        return converted_value, f"Converted {value} {from_unit} → {converted_value:.1f} {standard_unit}"
    
    return value, ""


def format_value_with_unit(value: float, param_key: str) -> str:
    """Format a value nicely with its unit."""
    info = PARAM_INFO.get(param_key, {})
    unit = info.get("unit", "")
    
    if isinstance(value, float) and value == int(value):
        formatted = f"{int(value):,}"
    elif isinstance(value, float):
        formatted = f"{value:,.2f}".rstrip('0').rstrip('.')
    else:
        formatted = str(value)
    
    if unit and unit != "(no unit)" and unit != "(stiffness factor)":
        return f"{formatted} {unit}"
    return formatted


def validate_single_param(key: str, value: float, all_params: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a single parameter value."""
    info = PARAM_INFO.get(key, {})
    
    # Check maximum value constraints
    if "max_value" in info and value > info["max_value"]:
        return False, f"This value is too high! The maximum allowed is {info['max_value']} {info.get('unit', '')}."
    
    # Check positive values
    if key in ["Emod", "a", "b", "t", "Kx", "Ky", "Kz"] and value <= 0:
        return False, f"This value must be greater than zero."
    
    # Check Poisson's ratio range
    if key == "nu" and (value < 0 or value > 0.5):
        return False, f"This value should be between 0 and 0.5."
    
    # Check load positions against slab dimensions
    if key == "x1" and "a" in all_params and value > all_params["a"]:
        return False, f"This can't be larger than your slab length ({all_params['a']} mm)."
    if key == "x2" and "a" in all_params and value > all_params["a"]:
        return False, f"This can't be larger than your slab length ({all_params['a']} mm)."
    if key == "y1" and "b" in all_params and value > all_params["b"]:
        return False, f"This can't be larger than your slab width ({all_params['b']} mm)."
    if key == "y2" and "b" in all_params and value > all_params["b"]:
        return False, f"This can't be larger than your slab width ({all_params['b']} mm)."
    
    # Check x1 < x2 and y1 < y2
    if key == "x2" and "x1" in all_params and value <= all_params["x1"]:
        return False, f"This must be larger than the load start position ({all_params['x1']} mm)."
    if key == "y2" and "y1" in all_params and value <= all_params["y1"]:
        return False, f"This must be larger than the load start position ({all_params['y1']} mm)."
    
    # Check q is positive
    if key == "q" and value < 0:
        return False, f"Pressure can't be negative."
    
    return True, ""


def get_progress_bar(params: Dict[str, Any]) -> str:
    """Create a visual progress bar."""
    total = len(DEFAULT_VALUES)
    filled = len(params)
    percentage = (filled / total) * 100
    
    filled_blocks = int(percentage / 10)
    empty_blocks = 10 - filled_blocks
    
    bar = "█" * filled_blocks + "░" * empty_blocks
    return f"[{bar}] {filled}/{total} parameters ({percentage:.0f}%)"


def display_current_status(params: Dict[str, Any], stage: str = "collecting"):
    """Display a friendly, easy-to-understand status card."""
    
    st.markdown("---")
    st.markdown("### 📊 Current Configuration Status")
    
    # Progress bar
    progress = get_progress_bar(params)
    st.markdown(f"**Progress:** `{progress}`")
    
    if params:
        st.markdown("#### ✅ Values You've Provided:")
        
        # Group by category
        for category, keys in PARAM_CATEGORIES.items():
            category_params = {k: params[k] for k in keys if k in params}
            if category_params:
                st.markdown(f"**{category}:**")
                cols = st.columns(min(len(category_params), 4))
                for idx, (key, value) in enumerate(category_params.items()):
                    info = PARAM_INFO[key]
                    with cols[idx % 4]:
                        st.metric(
                            label=f"{info['icon']} {get_param_display_name(key)}",
                            value=format_value_with_unit(value, key)
                        )
    
    # Show what's still needed
    missing = [k for k in PARAM_ORDER if k not in params]
    if missing:
        st.markdown("#### ⏳ Still Needed:")
        missing_text = ", ".join([f"{PARAM_INFO[k]['icon']} {get_param_display_name(k)}" for k in missing[:5]])
        if len(missing) > 5:
            missing_text += f" (+{len(missing) - 5} more)"
        st.markdown(missing_text)
    
    st.markdown("---")


def get_friendly_param_question(key: str, params: Dict[str, Any]) -> str:
    """Generate a friendly question for a specific parameter."""
    info = PARAM_INFO[key]
    display_name = get_param_display_name(key)
    
    question = f"{info['icon']} **{display_name}** ({info['technical_name']})\n\n"
    question += f"📝 *What this means:* {info['simple_explanation']}\n\n"
    question += f"📏 *Typical range:* {info['typical_range']}\n\n"
    question += f"💡 *Example:* {info['example']}\n\n"
    
    # Add context if related to slab size
    if key in ["x1", "x2"] and "a" in params:
        question += f"ℹ️ *Note: Your slab length is {params['a']} mm, so this should be less than that.*\n\n"
    if key in ["y1", "y2"] and "b" in params:
        question += f"ℹ️ *Note: Your slab width is {params['b']} mm, so this should be less than that.*\n\n"
    if key == "x2" and "x1" in params:
        question += f"ℹ️ *Note: This should be greater than Load Start ({params['x1']} mm).*\n\n"
    if key == "y2" and "y1" in params:
        question += f"ℹ️ *Note: This should be greater than Load Start ({params['y1']} mm).*\n\n"
    
    # Show default option
    default_val = DEFAULT_VALUES[key]
    question += f"🔧 **Default value:** {format_value_with_unit(default_val, key)}\n\n"
    question += f"👉 *Enter a value, or type 'default' or 'skip' to use the default.*"
    
    return question


def build_conversation_context(messages: List[Dict], params: Dict[str, Any]) -> str:
    """Build context string from conversation history."""
    context_parts = []
    
    # Add parameter summary
    if params:
        context_parts.append("Current parameters collected:")
        for key, value in params.items():
            info = PARAM_INFO.get(key, {})
            context_parts.append(f"- {info.get('name', key)}: {format_value_with_unit(value, key)}")
    
    # Add recent conversation (last 6 messages)
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    if recent_messages:
        context_parts.append("\nRecent conversation:")
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)


# =============================================================================
# LLM INTERACTION
# =============================================================================

def create_conversational_system_prompt() -> str:
    return """You are a friendly Pavement Configuration Assistant helping non-technical users set up parameters for rigid pavement analysis.

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
""" + json.dumps({k: {**v, "default": DEFAULT_VALUES[k]} for k, v in PARAM_INFO.items()}, indent=2)


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


def process_user_input_with_llm(
    user_input: str, 
    llm: ChatOllama, 
    context: str, 
    current_asking: Optional[str] = None
) -> Dict[str, Any]:
    """Process user input through LLM for extraction and response generation."""
    
    # First, check if user wants all defaults using our pattern matching as a backup
    if check_use_all_defaults_intent(user_input):
        return {
            "understood_value": None,
            "use_default": False,
            "use_all_defaults": True,
            "needs_clarification": False,
            "friendly_response": "Got it! I'll use the default values for all remaining parameters."
        }
    
    try:
        system_message = SystemMessage(content=create_conversational_system_prompt())
        extraction_prompt = create_extraction_prompt(user_input, context, current_asking)
        human_message = HumanMessage(content=extraction_prompt)
        
        response = llm.invoke([system_message, human_message])
        
        if response and response.content:
            # Parse the response
            content = response.content.strip()
            
            # Try to extract JSON
            if "```json" in content:
                content = re.sub(r'```json\s*', '', content)
                content = re.sub(r'```\s*', '', content)
            elif "```" in content:
                content = re.sub(r'```\s*', '', content)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            try:
                result = json.loads(content)
                
                # Ensure all expected keys exist with defaults
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
                # If JSON parsing fails, return a helpful message
                return {
                    "understood_value": None,
                    "use_default": False,
                    "use_all_defaults": False,
                    "needs_clarification": True,
                    "friendly_response": "I'm not quite sure I understood that. Could you try again? You can enter a number, or say 'default' to use the suggested value. 😊"
                }
        
        return {"needs_clarification": True, "friendly_response": "I didn't catch that. Could you please try again?"}
    
    except Exception as e:
        return {
            "needs_clarification": True, 
            "friendly_response": f"I had a small hiccup processing that. Could you try again? 😊"
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
- You can give me a value, or just say **"default"** to use the suggested value
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

**Ready to start?** Just tell me about your pavement, or type **"let's begin"** and I'll guide you step by step! 

*Example: "I have a 4 meter square slab" or "let's begin"*
"""


def generate_completion_message(params: Dict[str, Any], user_provided: List[str]) -> str:
    """Generate a friendly completion message."""
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


# =============================================================================
# DISPLAY FUNCTIONS  
# =============================================================================

def display_params_table(params: Dict[str, Any], user_keys: List[str]):
    """Display parameters in a professional, grouped table."""
    
    for category, keys in PARAM_CATEGORIES.items():
        category_params = [(k, params.get(k)) for k in keys if k in params]
        if not category_params:
            continue
            
        st.markdown(f"#### {category}")
        
        rows = []
        for key, value in category_params:
            info = PARAM_INFO[key]
            source = "✏️ You provided" if key in user_keys else "⚙️ Default"
            rows.append({
                "": info["icon"],
                "Setting": info["name"],
                "Value": format_value_with_unit(value, key),
                "Source": source
            })
        
        df = pd.DataFrame(rows)
        st.table(df)


def display_final_output(params: Dict[str, Any], user_keys: List[str]):
    """Display final output with download options."""
    
    st.markdown("### 📋 Final Configuration")
    display_params_table(params, user_keys)
    
    # Create downloadable formats
    json_str = json.dumps(params, indent=2)
    
    # Readable report
    lines = ["=" * 60]
    lines.append("    RIGID PAVEMENT ANALYSIS CONFIGURATION")
    lines.append("=" * 60)
    lines.append("")
    
    for category, keys in PARAM_CATEGORIES.items():
        lines.append(f"\n  {category}")
        lines.append("  " + "-" * 40)
        for key in keys:
            if key in params:
                info = PARAM_INFO[key]
                value = params[key]
                source = "[User]" if key in user_keys else "[Default]"
                lines.append(f"    {info['name']}: {format_value_with_unit(value, key)} {source}")
    
    lines.append("")
    lines.append("=" * 60)
    report = "\n".join(lines)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Report (TXT)",
            data=report,
            file_name="pavement_config_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="📥 Download Data (JSON)",
            data=json_str,
            file_name="pavement_config.json",
            mime="application/json",
            use_container_width=True,
        )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Pavement Configuration Assistant",
        page_icon="🛣️",
        layout="wide"
    )
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "params" not in st.session_state:
        st.session_state.params = {}
    if "user_provided_keys" not in st.session_state:
        st.session_state.user_provided_keys = []
    if "current_asking" not in st.session_state:
        st.session_state.current_asking = None
    if "mode" not in st.session_state:
        st.session_state.mode = "welcome"  # welcome, guided, free, complete
    if "welcomed" not in st.session_state:
        st.session_state.welcomed = False
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("🔌 Connection")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        
        st.subheader("🤖 Model")
        selected_model = st.selectbox("Choose Model", options=AVAILABLE_MODELS, index=0)
        custom_model = st.text_input("Or enter custom model name", value="")
        active_model = custom_model.strip() if custom_model.strip() else selected_model
        
        # Connection status
        ollama_connected, model_available, available_models = check_ollama_connection(ollama_url, active_model)
        
        if ollama_connected:
            st.success("✅ Ollama connected")
            if model_available:
                st.success(f"✅ {active_model} ready")
            else:
                st.warning(f"⚠️ Model {active_model} not found")
                if available_models:
                    st.info(f"Available: {', '.join(available_models[:5])}")
        else:
            st.error("❌ Cannot connect to Ollama")
        
        st.markdown("---")
        
        # Show current progress
        if st.session_state.params:
            st.markdown("### 📊 Progress")
            progress = len(st.session_state.params) / len(DEFAULT_VALUES)
            st.progress(progress)
            st.markdown(f"**{len(st.session_state.params)}/{len(DEFAULT_VALUES)}** parameters set")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.messages = []
            st.session_state.params = {}
            st.session_state.user_provided_keys = []
            st.session_state.current_asking = None
            st.session_state.mode = "welcome"
            st.session_state.welcomed = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 Quick Reference")
        st.markdown("""
        **Typical Values:**
        - Slab: 3-6m × 3-4.5m × 200mm
        - Elasticity: 24,000-30,000 MPa
        - Tire Pressure: 0.1-0.7 MPa
        
        **Units Accepted:**
        - Length: mm, cm, m, feet, inch
        - Pressure: MPa, kPa, psi
        """)
    
    # Main content area
    st.title("🛣️ Pavement Configuration Assistant")
    st.markdown("*Your friendly guide to setting up pavement analysis parameters*")
    
    # Check if Ollama is ready
    ollama_ready = ollama_connected and model_available
    llm = get_ollama_llm(ollama_url, active_model) if ollama_ready else None
    
    # Display welcome message if first time
    if not st.session_state.welcomed:
        with st.chat_message("assistant"):
            st.markdown(generate_welcome_message())
        st.session_state.messages.append({
            "role": "assistant",
            "content": generate_welcome_message()
        })
        st.session_state.welcomed = True
    
    # Display chat history
    for message in st.session_state.messages[1:]:  # Skip first welcome (already shown above)
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Show current status if we have parameters
    if st.session_state.params and st.session_state.mode != "complete":
        display_current_status(st.session_state.params)
    
    # Chat input
    user_input = st.chat_input(
        "Type your response here..." if ollama_ready else "Please connect to Ollama first",
        disabled=not ollama_ready
    )
    
    if user_input and ollama_ready and llm:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process input
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                # Build context
                context = build_conversation_context(
                    st.session_state.messages, 
                    st.session_state.params
                )
                
                # Check for special commands
                lower_input = user_input.lower().strip()
                
                # User wants to start guided mode
                if lower_input in ["let's begin", "lets begin", "start", "begin", "guide me", "help me"]:
                    st.session_state.mode = "guided"
                    # Find first missing parameter
                    missing = [k for k in PARAM_ORDER if k not in st.session_state.params]
                    if missing:
                        st.session_state.current_asking = missing[0]
                        response = f"Great! Let's go through this step by step. 😊\n\n"
                        response += get_friendly_param_question(missing[0], st.session_state.params)
                    else:
                        st.session_state.mode = "complete"
                        response = generate_completion_message(
                            st.session_state.params, 
                            st.session_state.user_provided_keys
                        )
                
                # User wants to use all defaults for remaining - check with comprehensive pattern matching
                elif check_use_all_defaults_intent(user_input):
                    missing = [k for k in PARAM_ORDER if k not in st.session_state.params]
                    for key in missing:
                        st.session_state.params[key] = DEFAULT_VALUES[key]
                    st.session_state.mode = "complete"
                    
                    if missing:
                        response = f"Perfect! I've filled in the remaining {len(missing)} values with defaults. 👍\n\n"
                    else:
                        response = "Great! All parameters were already set. 👍\n\n"
                    response += generate_completion_message(
                        st.session_state.params, 
                        st.session_state.user_provided_keys
                    )
                
                else:
                    # Process with LLM
                    result = process_user_input_with_llm(
                        user_input, llm, context, st.session_state.current_asking
                    )
                    
                    response = result.get("friendly_response", "")
                    
                    # Handle extracted value
                    if result.get("understood_value") is not None:
                        value = result["understood_value"]
                        param_key = result.get("parameter_key") or st.session_state.current_asking
                        
                        if param_key:
                            # Handle unit conversion
                            original_unit = result.get("original_unit")
                            if original_unit:
                                converted_value, conversion_msg = convert_to_standard_unit(
                                    value, original_unit, param_key
                                )
                                if conversion_msg:
                                    response = f"📐 {conversion_msg}\n\n" + response
                                value = converted_value
                            
                            # Validate
                            is_valid, error_msg = validate_single_param(
                                param_key, value, st.session_state.params
                            )
                            
                            if is_valid:
                                st.session_state.params[param_key] = value
                                if param_key not in st.session_state.user_provided_keys:
                                    st.session_state.user_provided_keys.append(param_key)
                                
                                info = PARAM_INFO[param_key]
                                response = f"✅ Got it! **{info['name']}** is set to **{format_value_with_unit(value, param_key)}**.\n\n"
                                
                                # Move to next parameter if in guided mode
                                missing = [k for k in PARAM_ORDER if k not in st.session_state.params]
                                if missing:
                                    st.session_state.current_asking = missing[0]
                                    response += f"Moving on to the next setting...\n\n"
                                    response += get_friendly_param_question(missing[0], st.session_state.params)
                                else:
                                    st.session_state.mode = "complete"
                                    response += generate_completion_message(
                                        st.session_state.params,
                                        st.session_state.user_provided_keys
                                    )
                            else:
                                response = f"⚠️ Hmm, that value has an issue: {error_msg}\n\n"
                                response += f"Could you try a different value? Or say 'default' to use **{format_value_with_unit(DEFAULT_VALUES[param_key], param_key)}**."
                    
                    # Handle use_all_defaults (LLM detected user wants all remaining defaults)
                    elif result.get("use_all_defaults"):
                        missing = [k for k in PARAM_ORDER if k not in st.session_state.params]
                        for key in missing:
                            st.session_state.params[key] = DEFAULT_VALUES[key]
                        st.session_state.mode = "complete"
                        
                        if missing:
                            response = f"Perfect! I've filled in the remaining {len(missing)} values with defaults. 👍\n\n"
                        else:
                            response = "Great! All parameters were already set. 👍\n\n"
                        response += generate_completion_message(
                            st.session_state.params, 
                            st.session_state.user_provided_keys
                        )
                    
                    # Handle use_default for single/current parameter
                    elif result.get("use_default") and st.session_state.current_asking:
                        param_key = st.session_state.current_asking
                        default_val = DEFAULT_VALUES[param_key]
                        st.session_state.params[param_key] = default_val
                        
                        info = PARAM_INFO[param_key]
                        response = f"👍 No problem! Using the default value of **{format_value_with_unit(default_val, param_key)}** for {info['name']}.\n\n"
                        
                        # Move to next
                        missing = [k for k in PARAM_ORDER if k not in st.session_state.params]
                        if missing:
                            st.session_state.current_asking = missing[0]
                            response += get_friendly_param_question(missing[0], st.session_state.params)
                        else:
                            st.session_state.mode = "complete"
                            response += generate_completion_message(
                                st.session_state.params,
                                st.session_state.user_provided_keys
                            )
                    
                    # Handle multiple extracted parameters
                    if result.get("extracted_multiple"):
                        for key, val in result["extracted_multiple"].items():
                            if key in DEFAULT_VALUES and key not in st.session_state.params:
                                # Handle unit conversion if needed
                                is_valid, _ = validate_single_param(key, val, st.session_state.params)
                                if is_valid:
                                    st.session_state.params[key] = val
                                    if key not in st.session_state.user_provided_keys:
                                        st.session_state.user_provided_keys.append(key)
                
                # Display response
                st.markdown(response)
                
                # Store in history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Show status after update
                if st.session_state.params and st.session_state.mode != "complete":
                    display_current_status(st.session_state.params)
                
                # Show final output if complete
                if st.session_state.mode == "complete":
                    final_params = {**DEFAULT_VALUES, **st.session_state.params}
                    display_final_output(final_params, st.session_state.user_provided_keys)
    
    elif user_input and not ollama_ready:
        st.warning("⚠️ Please ensure Ollama is running with the selected model.")


if __name__ == "__main__":
    main()
