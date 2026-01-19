# RLHF Data Agent

A Python-based agent for generating synthetic preference-ranked datasets for RLHF (Reinforcement Learning from Human Feedback) and DPO (Direct Preference Optimization) training. Specialized in ML/Data coding tasks using libraries like pandas, numpy, sklearn, and PyTorch. Built with Claude Code for REPPO pod publishing—verifiable, scalable, and designed to drive real fees in decentralized AI networks.

## Overview

This tool creates high-signal datasets with prompts, chosen/rejected responses, and heuristic scores (60% efficiency, 40% clarity). Each data point is hashed (SHA-256) for blockchain compatibility and data integrity verification. Run locally or deploy as a web app for custom generations. Perfect for AI model fine-tuning in domains like data analysis, optimization, debugging, explanation, and code generation.

For each prompt, the agent:
1. Generates two responses with different optimization targets (efficient vs. verbose)
2. Scores both responses using heuristic metrics
3. Ranks them to create chosen/rejected pairs
4. Exports in formats compatible with popular training libraries

## Key Features

- **Domains**: Pandas (data manipulation), NumPy (numerical computing), Scikit-learn (machine learning), PyTorch (deep learning)
- **Task Types**: Optimize, Debug, Explain, Generate, Refactor
- **Model Support**: Local/free models like Llama 3.2 (via Ollama) or API-based (Claude, GPT)—no API key needed for local runs
- **Generation**: Batch processing with progress tracking, estimated time/cost display ($0 for local models)
- **Export Options**: JSON (with SHA-256 hashes), CSV, JSONL (HuggingFace format)
- **UI**: Streamlit-based web interface—preview datasets, stop generation mid-way
- **Scalability**: Handles 1-1000+ samples with rate limiting
- **REPPO Integration**: Designed for pod hosting—generate custom datasets as a paid service

## Screenshots

*(Add screenshots of the interface showing model selection, generation settings, progress bar, and dataset preview)*

## Installation

```bash
# Clone the repository
git clone https://github.com/yumoshu/rlhf-data-agent.git
cd rlhf-data-agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For local models, install Ollama separately: https://ollama.ai

## Configuration

Create a `.env` file in the project root (optional for local models):

```env
# Required for cloud models (pick one or both)
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key

# Optional: Ollama configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434
```

### Using Ollama (Free Local Models)

1. Install Ollama: https://ollama.ai
2. Start the server: `ollama serve`
3. Pull a model: `ollama pull llama3.2` or `ollama pull codellama`
4. Select an Ollama model in the app—no API key needed

## Usage

```bash
streamlit run main.py
```

Opens at http://localhost:8501

### Workflow

1. **Configure**:
   - Select Model: Choose local (e.g., Llama 3.2—free) or API (enter key in sidebar)
   - Number of Samples: Slider from 1-1000 (start small for testing)
   - Domains & Task Types: Checkboxes to customize

2. **Generate**:
   - Click "Generate Data"
   - Monitor progress: Bar shows points processed, time elapsed/remaining, cost
   - Stop if needed via "Stop Generation"

3. **Preview & Export**:
   - View current dataset in the Preview tab (prompts, responses, scores, metadata)
   - Export tabs: Download as JSON/CSV/JSONL for immediate use in AI pipelines

## Scoring Methodology

Responses are scored on two dimensions:

### Efficiency (default 60%)
- Line count (fewer = better)
- Cyclomatic complexity (simpler = better)
- Anti-pattern detection (penalizes `iterrows()`, nested loops, etc.)
- Vectorization usage (rewards numpy/pandas operations)

### Clarity (default 40%)
- Docstring presence
- Comment ratio (optimal: 5-20%)
- Variable naming quality
- Type hints
- Modular design

The slider in the sidebar adjusts the weighting:
```
total_score = (efficiency_weight × efficiency) + (clarity_weight × clarity)
```

## Example Data Point

```json
{
  "id": "uuid",
  "prompt": "Optimize this pandas dataframe query for faster execution.",
  "chosen": "df.groupby('category').agg({'value': 'sum'}).reset_index()  # Efficient vectorized operation",
  "rejected": "for category in df['category'].unique(): ...",
  "metadata": {
    "task_type": "optimize",
    "domain": "pandas",
    "complexity": "intermediate",
    "chosen_score": {"efficiency": 0.85, "clarity": 0.70, "total": 0.79},
    "rejected_score": {"efficiency": 0.60, "clarity": 0.80, "total": 0.68},
    "sha256": "abc123..."
  }
}
```

### JSONL Export (HuggingFace compatible)
```json
{"prompt": "...", "chosen": "...", "rejected": "...", "domain": "pandas", "task_type": "optimize"}
```

## Using the Dataset

### With HuggingFace TRL
```python
from datasets import load_dataset
from trl import DPOTrainer

dataset = load_dataset("json", data_files="rlhf_dataset.jsonl")
# Use with DPOTrainer for preference learning
```

### With Custom Training
```python
import json

with open("rlhf_dataset.json") as f:
    data = json.load(f)

for item in data:
    prompt = item["prompt"]
    chosen = item["chosen"]
    rejected = item["rejected"]
    # Use for reward model training or DPO
```

## Project Structure

```
rlhf_data_agent/
├── main.py                 # Streamlit application
├── config.py               # Configuration and model settings
├── requirements.txt        # Dependencies
├── generator/
│   ├── prompts.py          # Prompt templates and generation
│   ├── responses.py        # LLM API integration
│   └── ranker.py           # Heuristic scoring logic
└── data/
    ├── schema.py           # Pydantic data models
    └── exporter.py         # Export utilities
```

## Supported Models

| Provider | Model | Cost |
|----------|-------|------|
| Anthropic | Claude Sonnet 4 | $3/$15 per 1M tokens |
| Anthropic | Claude Haiku 4 | $0.80/$4 per 1M tokens |
| OpenAI | GPT-4o | $2.50/$10 per 1M tokens |
| OpenAI | GPT-4o Mini | $0.15/$0.60 per 1M tokens |
| Ollama | Llama 3.2, CodeLlama, DeepSeek | Free (local) |

### Cost Estimation (250 data points)
- Claude Sonnet: ~$5-7
- Claude Haiku: ~$1-2
- GPT-4o: ~$4-5
- GPT-4o Mini: ~$0.20
- Ollama: **Free**

## Testing & Troubleshooting

**Quick Test**: Generate 10 samples—check exports for diversity and accuracy.

**Common Issues**:
- **API Key Error**: Enter in sidebar for non-local models
- **Slow Generation**: Use local Ollama for free/speed; limit samples
- **Dependencies Missing**: Rerun `pip install -r requirements.txt`
- **Ollama Connection Refused**: Ensure `ollama serve` is running in a terminal

## Limitations

- **Heuristic scoring**: Metrics are proxies, not ground truth preferences
- **Single-model generation**: Both responses come from the same LLM
- **Close margins**: Pairs with small score gaps may have ambiguous preferences
- **Domain-specific**: Optimized for ML/Data science coding tasks

## For REPPO Pod

This agent is designed for REPPO pod hosting—generate custom RLHF/DPO datasets as a decentralized service. SHA-256 hashes on every data point ensure verifiability for blockchain-based AI data markets.

**Use cases**:
- Generate datasets on-demand for fees
- Contribute to decentralized AI training data networks
- Earn yields from human-feedback data generation

## Build Process

Created using Claude Code with iterative development:
- Heuristic scoring system for preference ranking
- Multi-provider support (Anthropic/OpenAI/Ollama)
- Specialized for ML/Data coding domains

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or pull request.

## Contact

[@PreachingApe on X](https://x.com/PreachingApe) | Issues/PRs welcome
=======
# rlhf-data-agent
A Python-based agent for generating synthetic preference-ranked datasets for RLHF or DPO training. Specialized in ML/Data coding tasks using libraries like pandas, numpy, sklearn, and PyTorch. Built with Claude Code for REPPO pod publishing. Verifiable, scalable, and designed to drive real fees in decentralized AI networks.
