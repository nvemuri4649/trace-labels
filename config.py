"""
Configuration for the Trajectory Labeling App
"""

# Labels - Success OR one of 6 failure modes (mutually exclusive)
# Press keys 1-7 for quick labeling
LABELS = [
    "Success",  # 1 - Task completed correctly
    "Fabricated Information",  # 2 - Agent makes up facts, files, or data
    "Incorrect Tool Usage",  # 3 - Agent uses wrong tool or wrong parameters
    "Misunderstood Task",  # 4 - Agent misinterprets what was asked
    "Phantom Progress",  # 5 - Agent claims success without actual completion
    "Context Confusion",  # 6 - Agent confuses context from different parts of conversation
    "Other Error",  # 7 - Other error not covered by above categories
]

# Label descriptions for help text
LABEL_DESCRIPTIONS = {
    "Success": "The agent correctly completed the task as requested",
    "Fabricated Information": "Agent makes up facts, files, data, or information that doesn't exist",
    "Incorrect Tool Usage": "Agent uses the wrong tool, wrong parameters, or misuses available tools",
    "Misunderstood Task": "Agent misinterprets what was asked and works on wrong objective",
    "Phantom Progress": "Agent claims success or completion without actually finishing the task",
    "Context Confusion": "Agent confuses context from different parts of the conversation or mixes up information",
    "Other Error": "Any other error or issue not covered by the above categories",
}

# Allowed labelers - traces will be split among these users
ALLOWED_LABELERS = ["brian", "nikhil", "andrew"]

# Data storage backend options
STORAGE_BACKENDS = [
    "local_json",  # Local JSON file (development)
    "firestore",   # Google Cloud Firestore (production)
]

# Default storage backend
# Set to "firestore" for production deployment with persistent storage
# Set to "local_json" for local development
DEFAULT_STORAGE = "firestore"  # Using Firestore for persistent storage

# Local storage paths
LOCAL_DATA_DIR = "data"
TRAJECTORIES_FILE = "monaco_traces.jsonl"  # JSONL format
LABELS_FILE = "labels.json"
USERS_FILE = "users.json"

# App configuration
APP_TITLE = "🔬 Trajectory Labeling Tool"
APP_ICON = "🔬"

# Session timeout (seconds)
SESSION_TIMEOUT = 3600 * 24  # 24 hours
