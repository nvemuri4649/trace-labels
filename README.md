# Trajectory Labeling Tool

A simple, local-first tool for labeling AI agent trajectories with binary success/failure labels.

## How It Works

1. **Each labeler clones the repo and runs locally**
2. **Labels are saved per-user** to `data/labels_{username}.json`
3. **LLM predictions are pre-computed** and displayed alongside each trajectory
4. **Aggregate labels later** by collecting all `labels_*.json` files

## Quick Start

```bash
# Clone the repo
git clone <your-repo-url>
cd judgment-internal-trace-labeling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Binary Labels

- **0 = Erroneous**: The agent failed to complete the task or made errors
- **1 = Successful**: The agent successfully completed the task

## LLM Judge Script

Run the LLM judge to generate predictions for all trajectories:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your-key-here

# Run the judge (evaluates all trajectories)
python llm_judge.py

# Options:
python llm_judge.py --limit 10          # Only evaluate first 10
python llm_judge.py --concurrency 10    # More parallel API calls
python llm_judge.py --verbose           # Detailed output
```

The judge:
- Uses GPT-5.2 (or your configured model)
- Runs 5 iterations per trajectory
- Takes majority vote for final prediction
- Saves results to `data/llm_predictions.json`

**Important**: Commit and push `data/llm_predictions.json` to the repo so all labelers can see LLM predictions.

## Project Structure

```
judgment-internal-trace-labeling/
├── app.py                      # Streamlit labeling app
├── llm_judge.py                # LLM-as-Judge evaluation script
├── config.py                   # Configuration settings
├── data_store.py               # Data storage module
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── data/
    ├── monaco_traces.jsonl     # Source trajectories (tracked in Git LFS)
    ├── llm_predictions.json    # LLM predictions (commit to repo)
    └── labels_{username}.json  # Per-user labels (each person's local file)
```

## Workflow

### For the project lead:

1. Run the LLM judge to generate predictions:
   ```bash
   python llm_judge.py
   ```

2. Commit and push the predictions:
   ```bash
   git add data/llm_predictions.json
   git commit -m "Add LLM predictions"
   git push
   ```

### For each labeler:

1. Clone the repo (or pull latest):
   ```bash
   git clone <repo-url>
   # or
   git pull origin main
   ```

2. Run the app locally:
   ```bash
   streamlit run app.py
   ```

3. Enter your name and start labeling

4. Labels are automatically saved to `data/labels_{yourname}.json`

5. When done, share your labels file with the project lead
   ```bash
   # Option A: Email/Slack the file
   # Option B: Commit and push (if you have write access)
   git add data/labels_yourname.json
   git commit -m "Add labels from yourname"
   git push
   ```

### Aggregating labels:

Collect all `data/labels_*.json` files and merge them:

```python
import json
from pathlib import Path

all_labels = {}
for f in Path("data").glob("labels_*.json"):
    data = json.loads(f.read_text())
    labeler = data.get("labeler", f.stem.replace("labels_", ""))
    for label in data.get("labels", []):
        tid = label["trajectory_id"]
        if tid not in all_labels:
            all_labels[tid] = {}
        all_labels[tid][labeler] = label["label"]

# all_labels now contains: {trajectory_id: {labeler1: 0, labeler2: 1, ...}}
```

## Configuration

Edit `config.py` to customize:

```python
# LLM Judge settings
LLM_MODEL = "gpt-5.2"           # OpenAI model to use
LLM_JUDGE_ITERATIONS = 5        # Number of iterations for majority vote
LLM_TEMPERATURE = 0.7           # Temperature for non-determinism

# Data paths
DATA_DIR = "data"
TRAJECTORIES_FILE = "monaco_traces.jsonl"
LLM_PREDICTIONS_FILE = "llm_predictions.json"
```

## Requirements

- Python 3.10+
- OpenAI API key (for LLM judge)
- ~100MB disk space for trajectories

## Troubleshooting

**"No trajectories found"**: Make sure `data/monaco_traces.jsonl` exists. If using Git LFS:
```bash
git lfs pull
```

**"OPENAI_API_KEY not set"**: Export your API key:
```bash
export OPENAI_API_KEY=your-key-here
```

**App not loading**: Try a different port:
```bash
streamlit run app.py --server.port 8502
```
