# Rigid Pavement Parameter Configuration Agent

A Streamlit-based intelligent agent that uses Ollama with gemma3:12b model to parse natural language input and generate JSON configuration for rigid pavement analysis.

## Features

- 🤖 **Natural Language Processing**: Understands layman terms and technical jargon
- ✅ **Input Validation**: Validates constraints and provides meaningful error messages
- 📋 **Default Values**: Uses sensible defaults for missing parameters
- 🔄 **Interactive Flow**: Asks for confirmation before using defaults
- 📥 **JSON Export**: Download the generated configuration as a JSON file
- 🏠 **Local AI**: Runs completely locally using Ollama - no API keys needed!

## Prerequisites

1. **Install Ollama**: Download from [ollama.com](https://ollama.com/)

2. **Pull the gemma3:12b model**:
```bash
ollama pull gemma3:12b
```

3. **Ensure Ollama is running**:
```bash
ollama serve
```

## Installation

1. Clone or navigate to the project directory:
```bash
cd /home/surriya_gokul/GenAI_civil
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Ensure Ollama is running with gemma3:12b model (sidebar will show status)
2. Type your requirements in natural language in the chat input
3. The agent will extract parameters and ask about missing values
4. Confirm to use default values or provide additional details
5. Download the generated JSON configuration

## Example Inputs

- "I need a 4m square slab with elasticity 30000 MPa"
- "Set the tyre pressure to 0.5 MPa and load area from 100 to 800mm in both directions"
- "Use a 5 meter slab with Poisson ratio 0.25 and stiffness constants all equal to 100"
- "Modulus of elasticity 28000, slab size 4.5m, thickness 250mm"

## Parameter Reference

| Parameter | Description | Unit | Default | Constraints |
|-----------|-------------|------|---------|-------------|
| Emod | Modulus of Elasticity | MPa | 24000 | > 0 |
| nu | Poisson's Ratio | - | 0.3 | 0 to 0.5 |
| a | Slab Dimension (X) | mm | 3500 | > 0 |
| b | Slab Dimension (Y) | mm | 3500 | > 0 |
| t | Thickness | mm | 200 | > 0 |
| Kx | Stiffness Constant (X) | - | 50 | > 0 |
| Ky | Stiffness Constant (Y) | - | 50 | > 0 |
| Kz | Stiffness Constant (Z) | - | 50 | > 0 |
| x1 | Load Start (X) | mm | 100 | ≤ a |
| x2 | Load End (X) | mm | 1000 | ≤ a, > x1 |
| y1 | Load Start (Y) | mm | 200 | ≤ b |
| y2 | Load End (Y) | mm | 2000 | ≤ b, > y1 |
| q | Tyre Pressure | MPa | 0.1 | ≤ 0.7 |

## Constraints

- `x1`, `x2`, `y1`, `y2` cannot exceed slab dimensions `a` and `b`
- Tyre pressure `q` cannot exceed 0.7 MPa
- `x1` must be less than `x2`
- `y1` must be less than `y2`
- For square slabs, `a` = `b`

## License

MIT License
