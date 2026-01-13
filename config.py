"""
Configuration for the Trajectory Labeling App

Simple local-first setup: each labeler runs the app locally and 
saves their labels to a personal file that can be aggregated later.
"""

# Allowed labelers - ONLY these users can log in
# Trajectories are evenly distributed among them
ALLOWED_LABELERS = ["andrew", "nikhil", "brian", "rain", "abi"]

# Binary labels only: 0 = erroneous, 1 = successful
LABELS = {
    0: "Erroneous",
    1: "Successful",
}

LABEL_DESCRIPTIONS = {
    0: "The agent failed to complete the task or made errors",
    1: "The agent successfully completed the task as requested",
}

# Data paths
DATA_DIR = "data"
TRAJECTORIES_FILE = "monaco_traces.jsonl"  # Source trajectories
LLM_PREDICTIONS_FILE = "llm_predictions.json"  # LLM judge results (committed to repo)

# LLM Judge configuration
LLM_MODEL = "gpt-5.2"
LLM_JUDGE_ITERATIONS = 5  # Number of LLM calls per trajectory for majority vote
LLM_TEMPERATURE = 0.7  # Higher temp for non-determinism across iterations

# App configuration
APP_TITLE = "🔬 Trajectory Labeling Tool"
APP_ICON = "🔬"
