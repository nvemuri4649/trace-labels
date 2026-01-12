"""
Data Store Module - Handles all data persistence operations
Supports local JSON storage and Google Cloud Firestore
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib

import config


class LocalJSONStore:
    """Local JSON/JSONL file-based storage for development"""
    
    def __init__(self, data_dir: str = config.LOCAL_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self._trajectories_cache = None
        self._assignments_cache = None
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Create default files if they don't exist"""
        labels_path = self.data_dir / config.LABELS_FILE
        users_path = self.data_dir / config.USERS_FILE
        
        if not labels_path.exists():
            self._write_json(labels_path, {"labels": []})
        if not users_path.exists():
            self._write_json(users_path, {"users": {}})
    
    def _read_json(self, path: Path) -> dict:
        """Read and parse JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _write_json(self, path: Path, data: dict):
        """Write data to JSON file"""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_trajectories_from_jsonl(self) -> list:
        """Load trajectories from JSONL file"""
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
                    # Use trace_id as the ID, or generate one
                    trace_id = trace.get('trace_id', f"trace_{line_num}")
                    
                    # Parse input to extract the task/messages
                    input_data = trace.get('input', '')
                    messages = []
                    task = ""
                    
                    try:
                        if input_data and input_data.startswith('['):
                            parsed_input = json.loads(input_data)
                            if isinstance(parsed_input, list):
                                messages = parsed_input
                                # Get task from user messages or system prompt
                                for msg in parsed_input:
                                    if isinstance(msg, dict):
                                        role = msg.get('role', '')
                                        content = msg.get('content', '')
                                        if role == 'user' and content:
                                            task = content[:500]  # First user message as task
                                            break
                                        elif role == 'system' and not task:
                                            task = content[:200]  # System prompt as fallback
                    except json.JSONDecodeError:
                        task = input_data[:500] if input_data else "No task description"
                    
                    # Parse output
                    output_data = trace.get('output', '')
                    
                    # Calculate trajectory length (total content size)
                    content_length = len(input_data) + len(output_data)
                    
                    trajectory = {
                        "id": trace_id,
                        "task": task or "No task description",
                        "messages": messages,
                        "output": output_data,
                        "span_name": trace.get('span_name', ''),
                        "created_at": trace.get('created_at', ''),
                        "duration": trace.get('duration', 0),
                        "error": trace.get('error', ''),
                        "content_length": content_length,  # For sorting by length
                        "raw_trace": trace,  # Keep the full trace for reference
                    }
                    trajectories.append(trajectory)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        
        self._trajectories_cache = trajectories
        return trajectories
    
    def _get_user_assignments(self) -> dict:
        """
        Get the assignment of trajectories to users.
        Splits trajectories evenly among allowed labelers.
        Trajectories are sorted by length (longest first) before assignment.
        """
        if self._assignments_cache is not None:
            return self._assignments_cache
        
        trajectories = self._load_trajectories_from_jsonl()
        labelers = config.ALLOWED_LABELERS
        
        # Sort trajectories by content_length (longest first)
        sorted_trajectories = sorted(
            trajectories, 
            key=lambda t: t.get('content_length', 0), 
            reverse=True
        )
        
        # Split trajectories evenly (round-robin assignment)
        assignments = {labeler: [] for labeler in labelers}
        
        for idx, trajectory in enumerate(sorted_trajectories):
            assigned_labeler = labelers[idx % len(labelers)]
            assignments[assigned_labeler].append(trajectory['id'])
        
        self._assignments_cache = assignments
        return assignments
    
    # ==================== TRAJECTORIES ====================
    
    def get_all_trajectories(self) -> list:
        """Get all trajectories"""
        return self._load_trajectories_from_jsonl()
    
    def get_trajectories_for_user(self, username: str) -> list:
        """Get trajectories assigned to a specific user, sorted by length (longest first)"""
        if username not in config.ALLOWED_LABELERS:
            return []
        
        assignments = self._get_user_assignments()
        assigned_ids = set(assignments.get(username, []))
        
        all_trajectories = self._load_trajectories_from_jsonl()
        user_trajectories = [t for t in all_trajectories if t['id'] in assigned_ids]
        
        # Sort by content_length (longest first)
        return sorted(user_trajectories, key=lambda t: t.get('content_length', 0), reverse=True)
    
    def get_trajectory(self, trajectory_id: str) -> Optional[dict]:
        """Get a single trajectory by ID"""
        trajectories = self._load_trajectories_from_jsonl()
        for t in trajectories:
            if t.get("id") == trajectory_id:
                return t
        return None
    
    # ==================== LABELS ====================
    
    def get_all_labels(self) -> list:
        """Get all labels"""
        data = self._read_json(self.data_dir / config.LABELS_FILE)
        return data.get("labels", [])
    
    def get_labels_for_trajectory(self, trajectory_id: str) -> list:
        """Get all labels for a specific trajectory"""
        labels = self.get_all_labels()
        return [l for l in labels if l.get("trajectory_id") == trajectory_id]
    
    def get_labels_by_user(self, username: str) -> list:
        """Get all labels by a specific user"""
        labels = self.get_all_labels()
        return [l for l in labels if l.get("labeled_by") == username]
    
    def add_label(self, label: dict) -> str:
        """Add or update a label"""
        labels = self.get_all_labels()
        
        # Generate label ID
        label_id = f"{label['trajectory_id']}_{label['labeled_by']}"
        label["id"] = label_id
        label["labeled_at"] = datetime.now().isoformat()
        
        # Check if this user already labeled this trajectory (update)
        existing_idx = None
        for idx, l in enumerate(labels):
            if l.get("id") == label_id:
                existing_idx = idx
                break
        
        if existing_idx is not None:
            label["updated_at"] = datetime.now().isoformat()
            label["created_at"] = labels[existing_idx].get("created_at", label["labeled_at"])
            labels[existing_idx] = label
        else:
            label["created_at"] = label["labeled_at"]
            labels.append(label)
        
        self._write_json(
            self.data_dir / config.LABELS_FILE,
            {"labels": labels}
        )
        return label_id
    
    def delete_label(self, label_id: str) -> bool:
        """Delete a label by ID"""
        labels = self.get_all_labels()
        original_len = len(labels)
        labels = [l for l in labels if l.get("id") != label_id]
        
        if len(labels) < original_len:
            self._write_json(
                self.data_dir / config.LABELS_FILE,
                {"labels": labels}
            )
            return True
        return False
    
    # ==================== USERS ====================
    
    def get_all_users(self) -> dict:
        """Get all users"""
        data = self._read_json(self.data_dir / config.USERS_FILE)
        return data.get("users", {})
    
    def get_user(self, username: str) -> Optional[dict]:
        """Get a user by username"""
        users = self.get_all_users()
        return users.get(username)
    
    def add_user(self, username: str, user_data: dict) -> bool:
        """Add or update a user"""
        users = self.get_all_users()
        
        if username not in users:
            user_data["created_at"] = datetime.now().isoformat()
        else:
            user_data["created_at"] = users[username].get("created_at", datetime.now().isoformat())
        
        user_data["last_active"] = datetime.now().isoformat()
        users[username] = user_data
        
        self._write_json(
            self.data_dir / config.USERS_FILE,
            {"users": users}
        )
        return True
    
    def update_user_activity(self, username: str):
        """Update user's last active timestamp"""
        users = self.get_all_users()
        if username in users:
            users[username]["last_active"] = datetime.now().isoformat()
            self._write_json(
                self.data_dir / config.USERS_FILE,
                {"users": users}
            )
    
    # ==================== STATISTICS ====================
    
    def get_labeling_stats(self) -> dict:
        """Get overall labeling statistics"""
        all_trajectories = self.get_all_trajectories()
        labels = self.get_all_labels()
        users = self.get_all_users()
        
        # Count labels per user
        user_label_counts = {}
        for label in labels:
            user = label.get("labeled_by")
            user_label_counts[user] = user_label_counts.get(user, 0) + 1
        
        # Count label distribution
        label_distribution = {}
        for label in labels:
            label_value = label.get("label", "Unknown")
            label_distribution[label_value] = label_distribution.get(label_value, 0) + 1
        
        # Get per-user stats
        user_stats = {}
        assignments = self._get_user_assignments()
        for labeler in config.ALLOWED_LABELERS:
            assigned_count = len(assignments.get(labeler, []))
            labeled_count = user_label_counts.get(labeler, 0)
            user_stats[labeler] = {
                "assigned": assigned_count,
                "labeled": labeled_count,
                "remaining": assigned_count - labeled_count,
                "completion_rate": labeled_count / assigned_count if assigned_count > 0 else 0
            }
        
        return {
            "total_trajectories": len(all_trajectories),
            "total_labels": len(labels),
            "total_users": len(users),
            "labels_per_user": user_label_counts,
            "label_distribution": label_distribution,
            "user_stats": user_stats,
        }
    
    def get_unlabeled_trajectories_for_user(self, username: str) -> list:
        """Get trajectories assigned to user that they haven't labeled yet"""
        user_trajectories = self.get_trajectories_for_user(username)
        user_labels = self.get_labels_by_user(username)
        labeled_ids = {l.get("trajectory_id") for l in user_labels}
        
        return [t for t in user_trajectories if t.get("id") not in labeled_ids]


def get_data_store(backend: str = None):
    """Factory function to get the appropriate data store"""
    backend = backend or os.environ.get("STORAGE_BACKEND", config.DEFAULT_STORAGE)
    
    # For now, only local JSON store is implemented
    return LocalJSONStore()
