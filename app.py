import streamlit as st
import json
import re
from typing import Dict, Any, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Default values from sample_output.json
DEFAULT_VALUES = {
    "Emod": 24000.0,  # Modulus of elasticity (MPa)
    "nu": 0.3,        # Poisson's ratio
    "a": 3500,        # Slab dimension (mm)
    "b": 3500,        # Slab dimension (mm) - equal to 'a' for square slab
    "t": 200.0,       # Thickness (mm)
    "Kx": 50,         # Stiffness constant x
    "Ky": 50,         # Stiffness constant y
    "Kz": 50,         # Stiffness constant z
    "x1": 100.0,      # Load start position x (mm)
    "x2": 1000.0,     # Load end position x (mm)
    "y1": 200.0,      # Load start position y (mm)
    "y2": 2000.0,     # Load end position y (mm)
    "q": 0.1          # Tyre contact pressure (MPa)
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

AVAILABLE_MODELS = [
    "llama3.1:8b",
    "deepseek-r1:8b",
]


def get_ollama_llm(base_url: str = "http://localhost:11434", model: str = "gemma3:12b"):
    """Create a LangChain ChatOllama instance."""
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.1,
        num_predict=500,
    )


def check_ollama_connection(base_url: str = "http://localhost:11434", model_name: str = "gemma3:12b") -> Tuple[bool, bool, list]:
    """Check if Ollama is running and if the model is available. Returns (connected, model_available, available_models)."""
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
    """Create a prompt for the model to extract parameters from user input."""
    return f"""### ROLE
You are a Pavement Engineering Assistant that extracts parameters for Rigid Pavement Analysis from natural language.

### REQUIRED OUTPUT KEYS (with defaults)
{{
  "Emod": 24000.0,      // Modulus of elasticity (MPa)
  "nu": 0.3,            // Poisson's ratio
  "a": 3500,            // Slab dimension X (mm)
  "b": 3500,            // Slab dimension Y (mm)
  "t": 200.0,           // Slab thickness (mm)
  "Kx": 50,             // Stiffness constant X
  "Ky": 50,             // Stiffness constant Y
  "Kz": 50,             // Stiffness constant Z
  "x1": 100.0,          // Load start X (mm)
  "x2": 1000.0,         // Load end X (mm)
  "y1": 200.0,          // Load start Y (mm)
  "y2": 2000.0,         // Load end Y (mm)
  "q": 0.1              // Tyre contact pressure (MPa, Max 0.7)
}}

### INTERPRETATION RULES
1. **Units:** Convert ALL lengths to **mm** (1m=1000mm, 1cm=10mm) and ALL pressures to **MPa**.
2. **Synonyms:**
   - "Concrete strength", "stiffness of concrete", "Young's modulus", "elasticity", "E" -> **Emod**
   - "Foundation stiffness", "soil support", "spring constant", "subgrade reaction", "stiffness" -> **Kx, Ky, Kz** (apply to all three if unspecified)
   - "Lateral expansion", "Poisson" -> **nu**
   - "Slab size", "panel dimensions" -> **a** and **b** (if square or single value, a=b)
   - "Depth", "height of slab", "thickness" -> **t**
   - "pressure", "tyre pressure", "tire pressure", "contact pressure" -> **q**
3. **Load Geometry:**
   - "Load starts at X" -> **x1**, "Load ends at X" -> **x2**
   - "Load width W starting at S" -> x1 = S, x2 = S + W
4. **Relative/Contextual Values (CRITICAL):**
   - "Half max pressure" or "half the maximum" -> q = 0.35 (half of 0.7)
   - "Max allowed pressure" or "maximum pressure" -> q = 0.7
   - "About half" or "around 50%" of max -> calculate accordingly
   - "pretty stiff" with a number -> use that number for Emod

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
    """Create a prompt for the model to handle clarification responses."""
    return f"""You are an expert assistant for extracting rigid pavement parameters.

{FIELD_DESCRIPTIONS}

Previous context: {context}
User's response: "{user_input}"

Task: The user is responding to a clarification question. Extract any parameter values from their response.
Return ONLY a valid JSON object with the extracted parameters.
If the user agrees to use defaults (says "yes", "ok", "sure", "default", "use default", "y", "yeah", "yep", etc.), return: {{"use_defaults": true}}
If the user disagrees or wants to provide values, extract those values.

Return ONLY valid JSON. No explanations, no markdown, no code blocks."""


def parse_model_response(response_text: str) -> Dict[str, Any]:
    """Parse the model response and extract JSON."""
    try:
        # Clean the response - remove markdown code blocks if present
        cleaned = response_text.strip()
        
        # Remove markdown code blocks
        if "```json" in cleaned:
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
        elif "```" in cleaned:
            cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()
        
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def extract_parameters(user_input: str, llm: ChatOllama, is_clarification: bool = False, context: str = "") -> Dict[str, Any]:
    """Use LangChain with Ollama to extract parameters from user input."""
    try:
        system_message = SystemMessage(content="You are an expert assistant for extracting rigid pavement parameters. Always respond with ONLY valid JSON, no explanations.")
        
        if is_clarification:
            prompt = create_clarification_prompt(user_input, context)
        else:
            prompt = create_extraction_prompt(user_input)
        
        human_message = HumanMessage(content=prompt)
        response = llm.invoke([system_message, human_message])
        
        if response and response.content:
            return parse_model_response(response.content)
        return {}
    except Exception as e:
        st.error(f"Error calling Ollama: {str(e)}")
        return {}


def validate_parameters(params: Dict[str, Any]) -> Tuple[bool, list]:
    """Validate the extracted parameters against constraints."""
    errors = []
    
    # Get slab dimensions for constraint checking
    a = params.get("a", DEFAULT_VALUES["a"])
    b = params.get("b", DEFAULT_VALUES["b"])
    
    # Validate q (tyre pressure)
    if "q" in params and params["q"] > 0.7:
        errors.append(f"Tyre contact pressure (q) cannot exceed 0.7 MPa. Got: {params['q']} MPa")
    
    # Validate q is positive
    if "q" in params and params["q"] < 0:
        errors.append(f"Tyre contact pressure (q) must be positive. Got: {params['q']} MPa")
    
    # Validate x1, x2 against a
    if "x1" in params and params["x1"] > a:
        errors.append(f"x1 ({params['x1']} mm) cannot be greater than slab dimension a ({a} mm)")
    if "x2" in params and params["x2"] > a:
        errors.append(f"x2 ({params['x2']} mm) cannot be greater than slab dimension a ({a} mm)")
    
    # Validate y1, y2 against b
    if "y1" in params and params["y1"] > b:
        errors.append(f"y1 ({params['y1']} mm) cannot be greater than slab dimension b ({b} mm)")
    if "y2" in params and params["y2"] > b:
        errors.append(f"y2 ({params['y2']} mm) cannot be greater than slab dimension b ({b} mm)")
    
    # Validate x1 < x2
    if "x1" in params and "x2" in params and params["x1"] >= params["x2"]:
        errors.append(f"x1 ({params['x1']} mm) must be less than x2 ({params['x2']} mm)")
    
    # Validate y1 < y2
    if "y1" in params and "y2" in params and params["y1"] >= params["y2"]:
        errors.append(f"y1 ({params['y1']} mm) must be less than y2 ({params['y2']} mm)")
    
    # Validate Poisson's ratio
    if "nu" in params and (params["nu"] < 0 or params["nu"] > 0.5):
        errors.append(f"Poisson's ratio (nu) should be between 0 and 0.5. Got: {params['nu']}")
    
    # Validate positive values
    for key in ["Emod", "a", "b", "t", "Kx", "Ky", "Kz"]:
        if key in params and params[key] <= 0:
            errors.append(f"{key} must be positive. Got: {params[key]}")
    
    return len(errors) == 0, errors


def get_missing_fields(params: Dict[str, Any]) -> list:
    """Identify which fields are missing from the extracted parameters."""
    return [key for key in DEFAULT_VALUES.keys() if key not in params]


def merge_with_defaults(params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge extracted parameters with default values."""
    result = DEFAULT_VALUES.copy()
    for key, value in params.items():
        if key in result:
            result[key] = value
    return result


def format_json_output(params: Dict[str, Any]) -> str:
    """Format the parameters as a pretty JSON string."""
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
    
    # Sidebar for connection status and information
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Configurable Ollama URL - important for WSL users connecting to Windows
        st.subheader("Ollama Connection")
        default_url = "http://localhost:11434"
        ollama_url = st.text_input(
            "Ollama URL", 
            value=default_url,
            help="For WSL users: try 'http://host.docker.internal:11434' or your Windows IP"
        )
        
        # Model selection
        st.subheader("Model Selection")
        selected_model = st.selectbox(
            "Choose Model",
            options=AVAILABLE_MODELS,
            index=0,  # Default to first (best) model
            help="qwen2.5 and llama3.1 are recommended for best accuracy"
        )
        
        # Allow custom model name
        custom_model = st.text_input(
            "Or enter custom model name",
            value="",
            help="Leave empty to use selected model above"
        )
        
        # Use custom model if provided
        active_model = custom_model.strip() if custom_model.strip() else selected_model
        
        # Check Ollama connection
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
    
    # Initialize session state
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
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Check if Ollama is ready
    ollama_ready = ollama_connected and model_available
    
    # Create LLM instance if ready
    llm = get_ollama_llm(ollama_url, active_model) if ollama_ready else None
    
    # Chat input
    user_input = st.chat_input(
        "Describe your pavement parameters (e.g., 'I need a 4m square slab with elasticity 30000 MPa and tyre pressure 0.5')",
        disabled=not ollama_ready
    )
    
    if user_input and ollama_ready and llm:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner(f"Processing your input with {active_model}..."):
                if st.session_state.awaiting_confirmation:
                    # Handle confirmation response
                    context = f"Missing fields: {st.session_state.missing_fields}"
                    response = extract_parameters(user_input, llm, is_clarification=True, context=context)
                    
                    if response.get("use_defaults", False):
                        # User agreed to use defaults
                        final_params = merge_with_defaults(st.session_state.extracted_params)
                        st.session_state.final_output = final_params
                        st.session_state.awaiting_confirmation = False
                        
                        output_msg = "✅ Great! Using default values for missing fields.\n\n**Generated JSON Configuration:**"
                        st.markdown(output_msg)
                        st.json(final_params)
                        
                        # Provide download button
                        json_str = format_json_output(final_params)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name="pavement_config.json",
                            mime="application/json"
                        )
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": output_msg + f"\n```json\n{json_str}\n```"
                        })
                    else:
                        # User provided additional values
                        if response:
                            st.session_state.extracted_params.update(response)
                        
                        # Check for still missing fields
                        still_missing = get_missing_fields(st.session_state.extracted_params)
                        
                        if still_missing:
                            missing_with_defaults = {k: DEFAULT_VALUES[k] for k in still_missing}
                            msg = f"I've noted your input. The following fields are still missing:\n\n"
                            msg += f"**Missing fields and their defaults:**\n```json\n{json.dumps(missing_with_defaults, indent=2)}\n```\n\n"
                            msg += "Would you like to use these default values, or would you prefer to specify them?"
                            st.markdown(msg)
                            st.session_state.messages.append({"role": "assistant", "content": msg})
                            st.session_state.missing_fields = still_missing
                        else:
                            # All fields provided
                            final_params = st.session_state.extracted_params
                            is_valid, errors = validate_parameters(final_params)
                            
                            if is_valid:
                                st.session_state.final_output = final_params
                                st.session_state.awaiting_confirmation = False
                                
                                output_msg = "✅ All parameters provided and validated!\n\n**Generated JSON Configuration:**"
                                st.markdown(output_msg)
                                st.json(final_params)
                                
                                json_str = format_json_output(final_params)
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json_str,
                                    file_name="pavement_config.json",
                                    mime="application/json"
                                )
                                
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": output_msg + f"\n```json\n{json_str}\n```"
                                })
                            else:
                                error_msg = "⚠️ **Validation Errors:**\n" + "\n".join([f"- {e}" for e in errors])
                                error_msg += "\n\nPlease provide corrected values."
                                st.markdown(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    # Initial extraction
                    extracted = extract_parameters(user_input, llm)
                    
                    if not extracted:
                        msg = "I couldn't extract any parameters from your input. Could you please provide more specific values?\n\n"
                        msg += "**Example:** 'Set the slab size to 4 meters, elasticity to 30000 MPa, and tyre pressure to 0.5 MPa'"
                        st.markdown(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                    else:
                        # Validate extracted parameters
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
                                
                                msg = f"✅ I've extracted the following parameters:\n```json\n{json.dumps(extracted, indent=2)}\n```\n\n"
                                missing_with_defaults = {k: DEFAULT_VALUES[k] for k in missing}
                                msg += f"**The following fields are missing and will use default values:**\n```json\n{json.dumps(missing_with_defaults, indent=2)}\n```\n\n"
                                msg += "Would you like to use these default values for the missing fields? (Yes/No, or provide the missing values)"
                                st.markdown(msg)
                                st.session_state.messages.append({"role": "assistant", "content": msg})
                            else:
                                # All fields provided
                                final_params = extracted
                                st.session_state.final_output = final_params
                                
                                output_msg = "✅ All parameters extracted and validated!\n\n**Generated JSON Configuration:**"
                                st.markdown(output_msg)
                                st.json(final_params)
                                
                                json_str = format_json_output(final_params)
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json_str,
                                    file_name="pavement_config.json",
                                    mime="application/json"
                                )
                                
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": output_msg + f"\n```json\n{json_str}\n```"
                                })
    
    elif user_input and not ollama_ready:
        st.warning("⚠️ Please ensure Ollama is running and gemma3:12b model is available.")
    
    # Reset button
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
