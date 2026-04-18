# prompt-lab

Side-by-side prompt testing across local Ollama models.

Run a single prompt against multiple models simultaneously and compare results in the terminal with timing info.

## Features

- **Parallel execution** — queries all specified models at the same time
- **Side-by-side layout** — rich columns per model with response and timing
- **Named templates** — save and reuse prompts by name
- **File input** — read prompts from text files
- **`--models all`** — automatically targets every installed Ollama model

## Installation

```bash
cd prompt-lab
pip install -e .
```

## Usage

```bash
# Run a prompt against specific models
prompt-lab run "Explain quantum entanglement simply" --models mistral,llama3,phi3

# Run against all installed models
prompt-lab run "What is 2+2?" --models all

# Run from a file
prompt-lab run -f my_prompt.txt --models mistral,llama3

# Save a named template
prompt-lab save explain "Explain the following in simple terms:"

# List saved templates
prompt-lab list

# Use a saved template
prompt-lab run --template explain --models mistral

# Delete a template
prompt-lab delete explain

# Custom Ollama host
prompt-lab run "Hello" --models mistral --host http://192.168.1.10:11434
```

## Output

Each model response appears in its own panel with:
- Model name and response time in the panel title
- Full response text
- Color-coded border (green = success, red = error)
- Timing summary table at the bottom

## Requirements

- Python >= 3.10
- [Ollama](https://ollama.ai) running locally (`ollama serve`)
- At least one model installed (`ollama pull mistral`)
