"""
Data Store Module - Simple local file storage for trajectory labeling.

Each labeler saves their labels to a personal file: data/labels_{username}.json
These files can be aggregated later for analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import config


class LocalDataStore:
    """Local file-based storage for each labeler."""
    
    def __init__(self, username: str):
        self.username = username.lower().strip()
        self.data_dir = Path(config.DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)
        
        # Each user gets their own labels file
        self.labels_file = self.data_dir / f"labels_{self.username}.json"
        self._trajectories_cache = None
        self._llm_predictions_cache = None
        
        self._ensure_labels_file()
    
    def _ensure_labels_file(self):
        """Create labels file if it doesn't exist."""
        if not self.labels_file.exists():
            self._write_json(self.labels_file, {"labels": [], "labeler": self.username})
    
    def _read_json(self, path: Path) -> dict:
        """Read and parse JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _write_json(self, path: Path, data: dict):
        """Write data to JSON file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_all_trajectories(self) -> List[Dict[str, Any]]:
        """Load all trajectories from JSONL file."""
        if self._trajectories_cache is not None:
            return self._trajectories_cache
        
        trajectories = []
        jsonl_path = self.data_dir / config.TRAJECTORIES_FILE
        
        if not jsonl_path.exists():
            return []
        
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                    trace_id = trace.get('trace_id', f"trace_{line_num}")
                    
                    # Parse input to extract messages
                    input_data = trace.get('input', '')
                    messages = []
                    task = ""
                    
                    try:
                        if input_data and input_data.startswith('['):
                            parsed_input = json.loads(input_data)
                            if isinstance(parsed_input, list):
                                messages = parsed_input
                                for msg in parsed_input:
                                    if isinstance(msg, dict):
                                        role = msg.get('role', '')
                                        content = msg.get('content', '')
                                        if role == 'user' and content:
                                            task = content[:500]
                                            break
                    except json.JSONDecodeError:
                        pass
                    
                    # Parse output
                    output_data = trace.get('output', '')
                    
                    # Calculate content length for sorting
                    content_length = len(json.dumps(messages) + str(output_data))
                    
                    trajectory = {
                        "id": trace_id,
                        "task": task or "No task description",
                        "messages": messages,
                        "output": output_data,
                        "span_name": trace.get('span_name', ''),
                        "created_at": trace.get('created_at', ''),
                        "duration": trace.get('duration', 0),
                        "content_length": content_length,
                        "raw_trace": trace,
                    }
                    trajectories.append(trajectory)
                except json.JSONDecodeError:
                    continue
        
        # Sort by content length (longest first)
        trajectories.sort(key=lambda x: x.get('content_length', 0), reverse=True)
        self._trajectories_cache = trajectories
        return trajectories
    
    def get_assigned_trajectories(self) -> List[Dict[str, Any]]:
        """Get trajectories assigned to this user.
        
        Trajectories are evenly distributed among ALLOWED_LABELERS by index.
        Each labeler gets every Nth trajectory based on their position in the list.
        
        Trajectories with LLM predictions are shown FIRST, then those without.
        """
        all_trajectories = self.get_all_trajectories()
        
        # Get user index in allowed labelers
        labelers = [l.lower() for l in config.ALLOWED_LABELERS]
        if self.username not in labelers:
            return []  # User not in allowed list
        
        user_index = labelers.index(self.username)
        num_labelers = len(labelers)
        
        # Assign trajectories by index modulo
        assigned = [t for i, t in enumerate(all_trajectories) if i % num_labelers == user_index]
        
        # Sort: trajectories with LLM predictions come first
        llm_predictions = self.get_llm_predictions()
        
        def sort_key(trajectory):
            has_prediction = trajectory["id"] in llm_predictions
            # Primary sort: has prediction (True=0, False=1, so predicted first)
            # Secondary sort: content length (longest first)
            return (0 if has_prediction else 1, -trajectory.get('content_length', 0))
        
        assigned.sort(key=sort_key)
        return assigned
    
    def get_llm_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Load LLM predictions from the predictions file."""
        if self._llm_predictions_cache is not None:
            return self._llm_predictions_cache
        
        predictions_path = self.data_dir / config.LLM_PREDICTIONS_FILE
        if not predictions_path.exists():
            return {}
        
        data = self._read_json(predictions_path)
        predictions = data.get("predictions", {})
        self._llm_predictions_cache = predictions
        return predictions
    
    def get_llm_prediction_for_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """Get LLM prediction for a specific trajectory."""
        predictions = self.get_llm_predictions()
        return predictions.get(trajectory_id)
    
    def get_all_labels(self) -> List[Dict[str, Any]]:
        """Get all labels from this user's file."""
        data = self._read_json(self.labels_file)
        return data.get("labels", [])
    
    def get_label_for_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """Get label for a specific trajectory."""
        labels = self.get_all_labels()
        for label in labels:
            if label.get("trajectory_id") == trajectory_id:
                return label
        return None
    
    def add_label(self, trajectory_id: str, label: int, notes: str = "") -> str:
        """Add or update a label for a trajectory."""
        data = self._read_json(self.labels_file)
        labels = data.get("labels", [])
        
        # Check if label already exists
        existing_idx = None
        for idx, existing in enumerate(labels):
            if existing.get("trajectory_id") == trajectory_id:
                existing_idx = idx
                break
        
        label_entry = {
            "trajectory_id": trajectory_id,
            "label": label,  # 0 or 1
            "notes": notes,
            "labeled_at": datetime.now().isoformat(),
            "labeler": self.username,
        }
        
        if existing_idx is not None:
            label_entry["created_at"] = labels[existing_idx].get("created_at", label_entry["labeled_at"])
            label_entry["updated_at"] = label_entry["labeled_at"]
            labels[existing_idx] = label_entry
        else:
            label_entry["created_at"] = label_entry["labeled_at"]
            labels.append(label_entry)
        
        data["labels"] = labels
        data["labeler"] = self.username
        data["last_updated"] = datetime.now().isoformat()
        
        self._write_json(self.labels_file, data)
        return trajectory_id
    
    def get_unlabeled_trajectories(self) -> List[Dict[str, Any]]:
        """Get trajectories that haven't been labeled yet."""
        all_trajectories = self.get_all_trajectories()
        labels = self.get_all_labels()
        labeled_ids = {l.get("trajectory_id") for l in labels}
        
        return [t for t in all_trajectories if t.get("id") not in labeled_ids]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get labeling statistics for this user's assigned trajectories."""
        assigned_trajectories = self.get_assigned_trajectories()
        labels = self.get_all_labels()
        
        # Only count labels for assigned trajectories
        assigned_ids = {t.get("id") for t in assigned_trajectories}
        relevant_labels = [l for l in labels if l.get("trajectory_id") in assigned_ids]
        
        label_counts = {0: 0, 1: 0}
        for label in relevant_labels:
            lval = label.get("label")
            if lval in label_counts:
                label_counts[lval] += 1
        
        return {
            "total_trajectories": len(assigned_trajectories),
            "labeled": len(relevant_labels),
            "remaining": len(assigned_trajectories) - len(relevant_labels),
            "erroneous_count": label_counts[0],
            "successful_count": label_counts[1],
            "completion_rate": len(relevant_labels) / len(assigned_trajectories) if assigned_trajectories else 0,
        }


def get_data_store(username: str) -> LocalDataStore:
    """Factory function to get data store for a user."""
    return LocalDataStore(username)
