# 🔬 AI Trajectory Labeling Tool

A collaborative Streamlit web app for labeling AI agent trajectories. Multiple labelers can access the same interface via URL and write annotations to a shared backend in real-time.

## Features

- **Pre-assigned labelers**: Data is automatically split among Brian, Nikhil, and Andrew
- **Real-time sync**: All labels are centrally stored and synchronized
- **Progress tracking**: Dashboard shows individual and team progress
- **Simple 6-label system**: Each trajectory gets exactly one label
- **Export capabilities**: Download labels as JSON or CSV

## Labeling Task

Each trajectory represents an AI agent's execution trace (conversation + response). Labelers assign exactly **one** of 7 labels:

| Key | Label | Description |
|-----|-------|-------------|
| **1** | Success | The agent correctly completed the task as requested |
| **2** | Fabricated Information | Agent makes up facts, files, data, or information that doesn't exist |
| **3** | Incorrect Tool Usage | Agent uses the wrong tool, wrong parameters, or misuses available tools |
| **4** | Misunderstood Task | Agent misinterprets what was asked and works on wrong objective |
| **5** | Phantom Progress | Agent claims success or completion without actually finishing the task |
| **6** | Context Confusion | Agent confuses context from different parts of the conversation or mixes up information |
| **7** | Other Error | Any other error or issue not covered by the above categories |

### Quick Labeling

**Press keys 1-7** to instantly label and move to the next trajectory. This enables rapid labeling without clicking.

### Trajectory Ordering

Trajectories are **sorted by length (longest first)** so each labeler tackles the most substantial traces first.

---

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
cd /Users/nikhil/Documents/StudioProjects/judgment-internal-trace-labeling

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### 3. Login and Start Labeling

- Click your name (Brian, Nikhil, or Andrew) to log in
- You'll see only your assigned trajectories
- Label each trajectory and click Submit

---

## Data Assignment

The 641 traces from `data/monaco_traces.jsonl` are automatically split:

| Labeler | Assigned Traces |
|---------|-----------------|
| Brian   | 214 |
| Nikhil  | 214 |
| Andrew  | 213 |

Each labeler only sees their assigned subset.

---

## Deployment Options

### Option 1: Streamlit Community Cloud (Recommended - Free)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/trajectory-labeling.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository, branch (`main`), and file (`app.py`)
   - Click "Deploy"

### Option 2: Self-Hosted (Docker)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t trajectory-labeling .
docker run -p 8501:8501 -v $(pwd)/data:/app/data trajectory-labeling
```

---

## Data Format

### Input: Traces (monaco_traces.jsonl)

Each line is a JSON object with:

```json
{
  "trace_id": "unique-id",
  "input": "[{\"role\": \"system\", \"content\": \"...\"}, {\"role\": \"user\", \"content\": \"...\"}]",
  "output": "Agent response...",
  "span_name": "chat_completions",
  "created_at": "2025-12-12T17:30:00Z",
  "duration": 1234
}
```

### Output: Labels (labels.json)

```json
{
  "labels": [
    {
      "id": "trace-id_labeler-name",
      "trajectory_id": "trace-id",
      "labeled_by": "nikhil",
      "label": "Success",
      "notes": "Optional notes...",
      "labeled_at": "2026-01-12T13:00:00",
      "created_at": "2026-01-12T13:00:00"
    }
  ]
}
```

---

## Project Structure

```
judgment-internal-trace-labeling/
├── app.py                 # Main Streamlit application
├── config.py              # Labels and configuration
├── data_store.py          # Data persistence layer
├── requirements.txt       # Python dependencies
├── data/
│   ├── monaco_traces.jsonl  # 641 AI agent traces (input)
│   ├── labels.json          # User labels (output)
│   └── users.json           # User records
└── README.md
```

---

## Customization

### Modify Labels

Edit `config.py`:

```python
LABELS = [
    "Success",
    "Your Custom Label 1",
    "Your Custom Label 2",
    # ...
]

LABEL_DESCRIPTIONS = {
    "Success": "Description of success...",
    "Your Custom Label 1": "Description...",
    # ...
}
```

### Change Allowed Labelers

Edit `config.py`:

```python
ALLOWED_LABELERS = ["alice", "bob", "charlie"]
```

---

## Tips for Labelers

1. **Read carefully**: Review the full conversation before labeling
2. **Use the correct label**: Choose the most specific failure mode
3. **Add notes**: Document edge cases or reasoning
4. **Review periodically**: Use the Review tab to check your past labels

---

## License

MIT License
