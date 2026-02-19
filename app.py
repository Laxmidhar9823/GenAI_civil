import streamlit as st
import json
import re
import pandas as pd
from typing import Dict, Any, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

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

FIELD_DESCRIPTIONS = """
Field Descriptions:
1. Emod - Modulus of elasticity of rigid pavement (in MPa)
2. nu - Poisson's ratio (dimensionless, typically 0.1 to 0.5)
3. a - Slab dimension in x-direction (in mm)
4. b - Slab dimension in y-direction (in mm) - For square slabs, a = b
5. t - Thickness of the slab (in mm)
6. Kx - Stiffness constant in x-direction
7. Ky - Stiffness constant in y-direction
8. Kz - Stiffness constant in z-direction
9. x1 - Start position of load in x-direction (in mm)
10. x2 - End position of load in x-direction (in mm)
11. y1 - Start position of load in y-direction (in mm)
12. y2 - End position of load in y-direction (in mm)
13. q - Tyre contact pressure (in MPa, max 0.7)

Constraints:
- x1, x2, y1, y2 cannot be greater than a and b respectively
- Tyre pressure q cannot exceed 0.7 MPa
- All length dimensions are in mm
- Modulus of elasticity (Emod) is in MPa
"""

AVAILABLE_MODELS = ["llama3.1:8b", "deepseek-r1:8b"]

PARAM_DESCRIPTIONS = {
    "Emod": {"name": "Modulus of Elasticity", "unit": "MPa", "category": "Material Properties"},
    "nu":   {"name": "Poisson's Ratio", "unit": "—", "category": "Material Properties"},
    "a":    {"name": "Slab Length (X-direction)", "unit": "mm", "category": "Slab Geometry"},
    "b":    {"name": "Slab Width (Y-direction)", "unit": "mm", "category": "Slab Geometry"},
    "t":    {"name": "Slab Thickness", "unit": "mm", "category": "Slab Geometry"},
    "Kx":   {"name": "Foundation Stiffness (X)", "unit": "—", "category": "Foundation Properties"},
    "Ky":   {"name": "Foundation Stiffness (Y)", "unit": "—", "category": "Foundation Properties"},
    "Kz":   {"name": "Foundation Stiffness (Z)", "unit": "—", "category": "Foundation Properties"},
    "x1":   {"name": "Load Start Position (X)", "unit": "mm", "category": "Load Configuration"},
    "x2":   {"name": "Load End Position (X)", "unit": "mm", "category": "Load Configuration"},
    "y1":   {"name": "Load Start Position (Y)", "unit": "mm", "category": "Load Configuration"},
    "y2":   {"name": "Load End Position (Y)", "unit": "mm", "category": "Load Configuration"},
    "q":    {"name": "Tyre Contact Pressure", "unit": "MPa", "category": "Load Configuration"},
}


def display_params_table(params: Dict[str, Any], title: str = "Configuration Parameters", highlight_keys: list = None):
    """Display parameters as a professional, readable Streamlit table grouped by category."""
    if highlight_keys is None:
        highlight_keys = []

    # Group parameters by category
    categories = {}
    for key in params:
        if key not in PARAM_DESCRIPTIONS:
            continue
        cat = PARAM_DESCRIPTIONS[key]["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(key)

    st.markdown(f"#### 📄 {title}")
    st.markdown("---")

    for cat, keys in categories.items():
        st.markdown(f"**{cat}**")
        rows = []
        for key in keys:
            desc = PARAM_DESCRIPTIONS[key]
            value = params[key]
            # Format numeric values nicely
            if isinstance(value, float):
                if value == int(value):
                    display_val = f"{int(value):,}"
                else:
                    display_val = f"{value:,.4f}".rstrip('0').rstrip('.')
            elif isinstance(value, int):
                display_val = f"{value:,}"
            else:
                display_val = str(value)

            source = "✏️ User" if key in highlight_keys else "⚙️ Default"
            rows.append({
                "Parameter": f"`{key}`",
                "Description": desc["name"],
                "Value": display_val,
                "Unit": desc["unit"],
                "Source": source,
            })

        df = pd.DataFrame(rows)
        st.table(df)


def format_readable_output(params: Dict[str, Any], user_keys: list = None) -> str:
    """Create a professional plain-text report of the parameters."""
    if user_keys is None:
        user_keys = []

    lines = []
    lines.append("=" * 70)
    lines.append("       RIGID PAVEMENT ANALYSIS — PARAMETER CONFIGURATION")
    lines.append("=" * 70)
    lines.append("")

    # Group by category
    categories = {}
    for key in params:
        if key not in PARAM_DESCRIPTIONS:
            continue
        cat = PARAM_DESCRIPTIONS[key]["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(key)

    for cat, keys in categories.items():
        lines.append(f"  ┌─ {cat}")
        lines.append(f"  │")
        for key in keys:
            desc = PARAM_DESCRIPTIONS[key]
            value = params[key]
            if isinstance(value, float) and value == int(value):
                display_val = f"{int(value):,}"
            elif isinstance(value, float):
                display_val = f"{value:,.4f}".rstrip('0').rstrip('.')
            elif isinstance(value, int):
                display_val = f"{value:,}"
            else:
                display_val = str(value)

            source = "[User]" if key in user_keys else "[Default]"
            unit_str = f" {desc['unit']}" if desc['unit'] != '—' else ""
            lines.append(f"  │   {key:>4s}  ({desc['name']}) : {display_val}{unit_str}  {source}")
        lines.append(f"  │")

    lines.append("=" * 70)
    lines.append("  Note: [User] = provided by user  |  [Default] = system default")
    lines.append("=" * 70)
    return "\n".join(lines)


def display_final_output(params: Dict[str, Any], user_keys: list = None):
    """Display the final output with professional table + download buttons."""
    if user_keys is None:
        user_keys = list(params.keys())

    display_params_table(params, title="Final Pavement Configuration", highlight_keys=user_keys)

    readable_report = format_readable_output(params, user_keys)
    json_str = json.dumps(params, indent=2)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            label="📥 Download Report (TXT)",
            data=readable_report,
            file_name="pavement_config_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            label="📥 Download Data (JSON)",
            data=json_str,
            file_name="pavement_config.json",
            mime="application/json",
            use_container_width=True,
        )


def get_message_for_final_output(params: Dict[str, Any], user_keys: list = None) -> str:
    """Return a markdown-friendly summary for chat history."""
    if user_keys is None:
        user_keys = list(params.keys())

    lines = ["✅ **Final Pavement Configuration**\n"]
    lines.append("| Parameter | Description | Value | Unit | Source |")
    lines.append("|-----------|-------------|-------|------|--------|")
    for key in params:
        if key not in PARAM_DESCRIPTIONS:
            continue
        desc = PARAM_DESCRIPTIONS[key]
        value = params[key]
        if isinstance(value, float) and value == int(value):
            display_val = f"{int(value):,}"
        elif isinstance(value, float):
            display_val = f"{value:,.4f}".rstrip('0').rstrip('.')
        elif isinstance(value, int):
            display_val = f"{value:,}"
        else:
            display_val = str(value)
        source = "User" if key in user_keys else "Default"
        lines.append(f"| `{key}` | {desc['name']} | {display_val} | {desc['unit']} | {source} |")

    return "\n".join(lines)


def display_extracted_params(params: Dict[str, Any]):
    """Display extracted parameters in a readable table (before final output)."""
    rows = []
    for key in params:
        if key not in PARAM_DESCRIPTIONS:
            continue
        desc = PARAM_DESCRIPTIONS[key]
        value = params[key]
        if isinstance(value, float) and value == int(value):
            display_val = f"{int(value):,}"
        elif isinstance(value, float):
            display_val = f"{value:,.4f}".rstrip('0').rstrip('.')
        elif isinstance(value, int):
            display_val = f"{value:,}"
        else:
            display_val = str(value)
        rows.append({
            "Parameter": f"`{key}`",
            "Description": desc["name"],
            "Value": display_val,
            "Unit": desc["unit"],
        })
    if rows:
        st.markdown("**Extracted Parameters:**")
        df = pd.DataFrame(rows)
        st.table(df)


def display_missing_params(missing_keys: list):
    """Display missing parameters with their default values in a readable table."""
    rows = []
    for key in missing_keys:
        if key not in PARAM_DESCRIPTIONS:
            continue
        desc = PARAM_DESCRIPTIONS[key]
        value = DEFAULT_VALUES[key]
        if isinstance(value, float) and value == int(value):
            display_val = f"{int(value):,}"
        elif isinstance(value, float):
            display_val = f"{value:,.4f}".rstrip('0').rstrip('.')
        elif isinstance(value, int):
            display_val = f"{value:,}"
        else:
            display_val = str(value)
        rows.append({
            "Parameter": f"`{key}`",
            "Description": desc["name"],
            "Default Value": display_val,
            "Unit": desc["unit"],
        })
    if rows:
        st.markdown("**Missing Parameters (will use defaults):**")
        df = pd.DataFrame(rows)
        st.table(df)


def get_ollama_llm(base_url: str = "http://localhost:11434", model: str = "gemma3:12b"):
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.1,
        num_predict=500,
    )


def check_ollama_connection(base_url: str = "http://localhost:11434", model_name: str = "gemma3:12b") -> Tuple[bool, bool, list]:
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


def create_extraction_prompt(user_input: str) -> str:
    return f"""### ROLE
You are a Pavement Engineering Assistant that extracts parameters for Rigid Pavement Analysis from natural language.

### REQUIRED OUTPUT KEYS
{{
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
}}

### RULES
1. Convert lengths to mm (1m=1000mm) and pressures to MPa
2. Synonyms:
   - Elasticity/Young's modulus/concrete strength -> Emod
   - Foundation stiffness/soil support -> Kx, Ky, Kz
   - Poisson -> nu
   - Slab size -> a, b
   - Thickness -> t
   - Pressure -> q
3. Load geometry: "starts at X" -> x1, "ends at X" -> x2
4. Values: "half max" -> q=0.35, "max pressure" -> q=0.7

### USER INPUT
"{user_input}"

### TASK
Extract ONLY the parameters explicitly mentioned or clearly implied by the user.
Return a JSON object containing ONLY the keys that were mentioned.
Do NOT include keys that the user did not mention.

### OUTPUT FORMAT
Return ONLY a valid JSON object. No explanations, no markdown, no code blocks.
If no parameters found, return: {{}}

### EXAMPLES
- Input: "elasticity 25000 and pressure about half the max" -> {{"Emod": 25000.0, "q": 0.35}}
- Input: "4 meter square slab" -> {{"a": 4000, "b": 4000}}
- Input: "stiffness 75 for all directions" -> {{"Kx": 75, "Ky": 75, "Kz": 75}}
- Input: "concrete is pretty stiff, around 25000 MPa, tyre pressure half the max allowed" -> {{"Emod": 25000.0, "q": 0.35}}

RESPOND WITH ONLY THE JSON OBJECT:"""


def create_clarification_prompt(user_input: str, context: str) -> str:
    return f"""You are an expert assistant for extracting rigid pavement parameters.

{FIELD_DESCRIPTIONS}

Previous context: {context}
User's response: "{user_input}"

Extract parameter values from the response.
If user agrees to defaults (yes/ok/sure/y), return: {{"use_defaults": true}}
Otherwise extract the values they provided.

Return ONLY valid JSON."""


def parse_model_response(response_text: str) -> Dict[str, Any]:
    try:
        cleaned = response_text.strip()
        
        if "```json" in cleaned:
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
        elif "```" in cleaned:
            cleaned = re.sub(r'```\s*', '', cleaned)
        
        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()
        
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def extract_parameters(user_input: str, llm: ChatOllama, is_clarification: bool = False, context: str = "") -> Dict[str, Any]:
    try:
        system_message = SystemMessage(content="You are an expert assistant for extracting rigid pavement parameters. Always respond with ONLY valid JSON, no explanations.")
        prompt = create_clarification_prompt(user_input, context) if is_clarification else create_extraction_prompt(user_input)
        human_message = HumanMessage(content=prompt)
        response = llm.invoke([system_message, human_message])
        
        if response and response.content:
            return parse_model_response(response.content)
        return {}
    except Exception as e:
        st.error(f"Error calling Ollama: {str(e)}")
        return {}


def validate_parameters(params: Dict[str, Any]) -> Tuple[bool, list]:
    errors = []
    a = params.get("a", DEFAULT_VALUES["a"])
    b = params.get("b", DEFAULT_VALUES["b"])
    
    if "q" in params and params["q"] > 0.7:
        errors.append(f"Tyre contact pressure (q) cannot exceed 0.7 MPa. Got: {params['q']} MPa")
    
    if "q" in params and params["q"] < 0:
        errors.append(f"Tyre contact pressure (q) must be positive. Got: {params['q']} MPa")
    
    if "x1" in params and params["x1"] > a:
        errors.append(f"x1 ({params['x1']} mm) cannot be greater than slab dimension a ({a} mm)")
    if "x2" in params and params["x2"] > a:
        errors.append(f"x2 ({params['x2']} mm) cannot be greater than slab dimension a ({a} mm)")
    
    if "y1" in params and params["y1"] > b:
        errors.append(f"y1 ({params['y1']} mm) cannot be greater than slab dimension b ({b} mm)")
    if "y2" in params and params["y2"] > b:
        errors.append(f"y2 ({params['y2']} mm) cannot be greater than slab dimension b ({b} mm)")
    
    if "x1" in params and "x2" in params and params["x1"] >= params["x2"]:
        errors.append(f"x1 ({params['x1']} mm) must be less than x2 ({params['x2']} mm)")
    
    if "y1" in params and "y2" in params and params["y1"] >= params["y2"]:
        errors.append(f"y1 ({params['y1']} mm) must be less than y2 ({params['y2']} mm)")
    
    if "nu" in params and (params["nu"] < 0 or params["nu"] > 0.5):
        errors.append(f"Poisson's ratio (nu) should be between 0 and 0.5. Got: {params['nu']}")
    
    for key in ["Emod", "a", "b", "t", "Kx", "Ky", "Kz"]:
        if key in params and params[key] <= 0:
            errors.append(f"{key} must be positive. Got: {params[key]}")
    
    return len(errors) == 0, errors


def get_missing_fields(params: Dict[str, Any]) -> list:
    return [key for key in DEFAULT_VALUES.keys() if key not in params]


def merge_with_defaults(params: Dict[str, Any]) -> Dict[str, Any]:
    result = DEFAULT_VALUES.copy()
    for key, value in params.items():
        if key in result:
            result[key] = value
    return result


def format_json_output(params: Dict[str, Any]) -> str:
    return json.dumps(params, indent=2)


def main():
    st.set_page_config(
        page_title="Rigid Pavement Parameter Agent",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ Rigid Pavement Parameter Configuration Agent")
    st.markdown("""
    This agent helps you configure parameters for rigid pavement analysis. 
    Simply describe your requirements in natural language, and the agent will generate the appropriate JSON configuration.
    """)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Ollama Connection")
        default_url = "http://localhost:11434"
        ollama_url = st.text_input(
            "Ollama URL", 
            value=default_url,
            help="For WSL users: try 'http://host.docker.internal:11434' or your Windows IP"
        )
        
        st.subheader("Model Selection")
        selected_model = st.selectbox(
            "Choose Model",
            options=AVAILABLE_MODELS,
            index=0,
            help="qwen2.5 and llama3.1 are recommended for best accuracy"
        )
        
        custom_model = st.text_input(
            "Or enter custom model name",
            value="",
            help="Leave empty to use selected model above"
        )
        
        active_model = custom_model.strip() if custom_model.strip() else selected_model
        
        ollama_connected, model_available, available_models = check_ollama_connection(ollama_url, active_model)
        
        if ollama_connected:
            st.success("✅ Ollama is running")
            if model_available:
                st.success(f"✅ {active_model} available")
            else:
                st.warning(f"⚠️ {active_model} not found.\nRun: `ollama pull {active_model}`")
                if available_models:
                    st.info(f"Available models: {', '.join(available_models[:5])}")
        else:
            st.error("❌ Cannot connect to Ollama")
            st.info("**Tips:**\n- Ensure Ollama is running\n- For WSL: Use your Windows IP (e.g., `http://172.x.x.x:11434`)\n- Run `ipconfig` in Windows to find your IP")
        
        st.markdown("---")
        st.header("📋 Parameter Reference")
        st.markdown("""
        | Parameter | Description | Unit |
        |-----------|-------------|------|
        | Emod | Modulus of Elasticity | MPa |
        | nu | Poisson's Ratio | - |
        | a, b | Slab Dimensions | mm |
        | t | Thickness | mm |
        | Kx, Ky, Kz | Stiffness Constants | - |
        | x1, x2 | Load Position (X) | mm |
        | y1, y2 | Load Position (Y) | mm |
        | q | Tyre Pressure | MPa |
        """)
        
        st.markdown("---")
        st.header("📌 Constraints")
        st.markdown("""
        - **q** ≤ 0.7 MPa
        - **x1, x2** ≤ a
        - **y1, y2** ≤ b
        - **x1** < **x2**
        - **y1** < **y2**
        """)
        
        st.markdown("---")
        st.header("🔢 Default Values")
        st.json(DEFAULT_VALUES)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "extracted_params" not in st.session_state:
        st.session_state.extracted_params = {}
    if "awaiting_confirmation" not in st.session_state:
        st.session_state.awaiting_confirmation = False
    if "missing_fields" not in st.session_state:
        st.session_state.missing_fields = []
    if "final_output" not in st.session_state:
        st.session_state.final_output = None
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    ollama_ready = ollama_connected and model_available
    llm = get_ollama_llm(ollama_url, active_model) if ollama_ready else None
    
    user_input = st.chat_input(
        "Describe your pavement parameters (e.g., 'I need a 4m square slab with elasticity 30000 MPa and tyre pressure 0.5')",
        disabled=not ollama_ready
    )
    
    if user_input and ollama_ready and llm:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner(f"Processing your input with {active_model}..."):
                if st.session_state.awaiting_confirmation:
                    context = f"Missing fields: {st.session_state.missing_fields}"
                    response = extract_parameters(user_input, llm, is_clarification=True, context=context)
                    
                    if response.get("use_defaults", False):
                        final_params = merge_with_defaults(st.session_state.extracted_params)
                        st.session_state.final_output = final_params
                        st.session_state.awaiting_confirmation = False
                        user_keys = list(st.session_state.extracted_params.keys())
                        
                        st.markdown("✅ Great! Using default values for missing fields.")
                        display_final_output(final_params, user_keys=user_keys)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": get_message_for_final_output(final_params, user_keys=user_keys)
                        })
                    else:
                        if response:
                            st.session_state.extracted_params.update(response)
                        
                        still_missing = get_missing_fields(st.session_state.extracted_params)
                        
                        if still_missing:
                            st.markdown("I've noted your input. The following fields are still missing:")
                            display_missing_params(still_missing)
                            st.markdown("Would you like to use these default values, or would you prefer to specify them?")
                            msg = "I've noted your input. Some fields are still missing. Would you like to use default values?"
                            st.session_state.messages.append({"role": "assistant", "content": msg})
                            st.session_state.missing_fields = still_missing
                        else:
                            final_params = st.session_state.extracted_params
                            is_valid, errors = validate_parameters(final_params)
                            
                            if is_valid:
                                st.session_state.final_output = final_params
                                st.session_state.awaiting_confirmation = False
                                user_keys = list(st.session_state.extracted_params.keys())
                                
                                st.markdown("✅ All parameters provided and validated!")
                                display_final_output(final_params, user_keys=user_keys)
                                
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": get_message_for_final_output(final_params, user_keys=user_keys)
                                })
                            else:
                                error_msg = "⚠️ **Validation Errors:**\n" + "\n".join([f"- {e}" for e in errors])
                                error_msg += "\n\nPlease provide corrected values."
                                st.markdown(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    extracted = extract_parameters(user_input, llm)
                    
                    if not extracted:
                        msg = "I couldn't extract any parameters from your input. Could you please provide more specific values?\n\n"
                        msg += "**Example:** 'Set the slab size to 4 meters, elasticity to 30000 MPa, and tyre pressure to 0.5 MPa'"
                        st.markdown(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                    else:
                        is_valid, errors = validate_parameters(extracted)
                        
                        if not is_valid:
                            error_msg = "⚠️ **Validation Errors:**\n" + "\n".join([f"- {e}" for e in errors])
                            error_msg += "\n\nPlease provide corrected values."
                            st.markdown(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        else:
                            st.session_state.extracted_params = extracted
                            missing = get_missing_fields(extracted)
                            
                            if missing:
                                st.session_state.awaiting_confirmation = True
                                st.session_state.missing_fields = missing
                                
                                st.markdown("✅ I've extracted the following parameters:")
                                display_extracted_params(extracted)
                                display_missing_params(missing)
                                st.markdown("Would you like to use these default values for the missing fields? *(Yes/No, or provide the missing values)*")
                                msg = "Extracted some parameters. Missing fields shown with defaults. Awaiting confirmation."
                                st.session_state.messages.append({"role": "assistant", "content": msg})
                            else:
                                final_params = extracted
                                st.session_state.final_output = final_params
                                user_keys = list(extracted.keys())
                                
                                st.markdown("✅ All parameters extracted and validated!")
                                display_final_output(final_params, user_keys=user_keys)
                                
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": get_message_for_final_output(final_params, user_keys=user_keys)
                                })
    
    elif user_input and not ollama_ready:
        st.warning("⚠️ Please ensure Ollama is running and gemma3:12b model is available.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Start New Configuration", use_container_width=True):
            st.session_state.messages = []
            st.session_state.extracted_params = {}
            st.session_state.awaiting_confirmation = False
            st.session_state.missing_fields = []
            st.session_state.final_output = None
            st.rerun()


if __name__ == "__main__":
    main()
