"""
Trajectory Labeling App - Simple local-first labeling tool.

Run locally with: streamlit run app.py
Each labeler gets their own labels file that can be aggregated later.
"""

import json
import re
import time
import streamlit as st
from pathlib import Path

import config
from data_store import get_data_store


def parse_concatenated_json(text: str) -> list:
    """Parse concatenated JSON objects (e.g., {...}{...}) into a list.
    
    Also handles double-encoded JSON strings.
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # Check if the entire thing is a JSON-encoded string (double-encoded)
    if text.startswith('"') and text.endswith('"'):
        try:
            # Decode the outer string
            text = json.loads(text)
            if isinstance(text, str):
                text = text.strip()
        except json.JSONDecodeError:
            pass
    
    results = []
    decoder = json.JSONDecoder()
    idx = 0
    
    while idx < len(text):
        # Skip whitespace
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            results.append(obj)
            idx += end_idx
        except json.JSONDecodeError:
            # If we can't parse more JSON, break
            break
    
    return results

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for cleaner UI
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0f172a;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Label buttons */
    .label-btn-success {
        background: linear-gradient(135deg, #059669, #10b981) !important;
        color: white !important;
        font-weight: 600;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.2rem;
    }
    
    .label-btn-error {
        background: linear-gradient(135deg, #dc2626, #ef4444) !important;
        color: white !important;
        font-weight: 600;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.2rem;
    }
    
    /* Trajectory content */
    .trajectory-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #334155;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #0ea5e9, #06b6d4);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
    }
    
    .assistant-message {
        background: #334155;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .tool-message {
        background: #1e293b;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        font-family: monospace;
        font-size: 0.85rem;
        border-left: 3px solid #f59e0b;
    }
    
    /* LLM prediction panel */
    .llm-panel {
        background: linear-gradient(135deg, #312e81, #4c1d95);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Stats */
    .stat-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Make buttons larger */
    .stButton > button {
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'current_trajectory_idx' not in st.session_state:
        st.session_state.current_trajectory_idx = 0
    if 'pending_advance' not in st.session_state:
        st.session_state.pending_advance = False
    if 'show_all' not in st.session_state:
        st.session_state.show_all = False


def render_login():
    """Render login screen."""
    st.markdown("""
    <div style="text-align: center; padding: 4rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🔬</h1>
        <h1 style="font-size: 2rem; margin-bottom: 2rem;">Trajectory Labeling Tool</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Enter your name to start labeling")
        username = st.text_input(
            "Username",
            placeholder="e.g., brian, nikhil, andrew",
            label_visibility="collapsed"
        )
        
        if st.button("Start Labeling", use_container_width=True, type="primary"):
            if username and username.strip():
                st.session_state.username = username.strip().lower()
                st.rerun()
            else:
                st.error("Please enter a username")
        
        st.markdown("""
        <div style="margin-top: 2rem; color: #64748b; font-size: 0.9rem;">
            <p>Your labels will be saved to: <code>data/labels_{username}.json</code></p>
            <p>Binary labels: <b>0</b> = Erroneous, <b>1</b> = Successful</p>
        </div>
        """, unsafe_allow_html=True)


def render_llm_prediction(prediction: dict):
    """Render LLM prediction panel."""
    if not prediction:
        st.markdown("""
        <div style="background: #1e293b; border-radius: 8px; padding: 1rem; color: #94a3b8;">
            <b>🤖 LLM Judge:</b> No prediction available
        </div>
        """, unsafe_allow_html=True)
        return
    
    final_score = prediction.get("final_score")
    confidence = prediction.get("confidence", 0)
    individual_scores = prediction.get("individual_scores", [])
    
    if final_score == 1:
        bg_color = "rgba(16, 185, 129, 0.2)"
        border_color = "#10b981"
        verdict = "SUCCESSFUL"
    elif final_score == 0:
        bg_color = "rgba(239, 68, 68, 0.2)"
        border_color = "#ef4444"
        verdict = "ERRONEOUS"
    else:
        bg_color = "rgba(148, 163, 184, 0.2)"
        border_color = "#94a3b8"
        verdict = "UNKNOWN"
    
    # Show individual votes
    votes_display = " ".join([
        f"{'✓' if s == 1 else '✗'}" for s in individual_scores
    ])
    
    st.markdown(f"""
    <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 12px; padding: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.2rem; font-weight: 600;">🤖 LLM Judge: {verdict}</span>
                <span style="margin-left: 1rem; color: #94a3b8;">Confidence: {confidence:.0%}</span>
            </div>
            <div style="font-family: monospace; font-size: 1.1rem;">
                Votes: {votes_display}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show reasoning if available
    reasonings = prediction.get("reasonings", [])
    if reasonings:
        with st.expander("📝 LLM Reasoning (first iteration)"):
            st.markdown(reasonings[0])


def render_trajectory(trajectory: dict):
    """Render a single trajectory."""
    st.markdown(f"### Task")
    st.markdown(f"**{trajectory.get('task', 'No task description')}**")
    
    messages = trajectory.get("messages", [])
    
    if not messages:
        st.info("No messages in this trajectory")
        return
    
    st.markdown("---")
    st.markdown("### Conversation")
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        
        if role == "system":
            with st.expander(f"📋 System Prompt", expanded=False):
                st.markdown(content[:2000] if len(content) > 2000 else content)
        
        elif role == "user":
            st.markdown(f"""
            <div class="user-message">
                <b>👤 User:</b><br>
                {content[:1000] if len(content) > 1000 else content}
            </div>
            """, unsafe_allow_html=True)
        
        elif role == "assistant":
            if content:
                st.markdown(f"""
                <div class="assistant-message">
                    <b>🤖 Assistant:</b><br>
                    {content[:1500] if len(content) > 1500 else content}
                </div>
                """, unsafe_allow_html=True)
            
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    if len(args) > 300:
                        args = args[:300] + "..."
                    st.markdown(f"""
                    <div class="tool-message">
                        🔧 <b>{name}</b>: {args}
                    </div>
                    """, unsafe_allow_html=True)
        
        elif role == "tool":
            with st.expander(f"🔧 Tool Response", expanded=False):
                st.code(content[:2000] if len(content) > 2000 else content)
    
    # Output
    output = trajectory.get("output", "")
    if output:
        st.markdown("---")
        st.markdown("### Final Output")
        
        # Try to parse concatenated JSON objects
        parsed_objects = parse_concatenated_json(output)
        
        if parsed_objects:
            st.markdown(f"*Parsed {len(parsed_objects)} JSON object(s)*")
            
            # Display each JSON object in a nice format
            for i, obj in enumerate(parsed_objects):
                if len(parsed_objects) > 1:
                    st.markdown(f"**Response {i+1}:**")
                
                # Check if this is a results-style response
                if isinstance(obj, dict) and "results" in obj:
                    results = obj.get("results", [])
                    if results:
                        st.markdown(f"*Found {len(results)} result(s):*")
                        for j, result in enumerate(results[:5]):  # Limit to 5 results
                            name = result.get("name", result.get("account_name", "Unknown"))
                            stage = result.get("stage", "")
                            summary = str(result.get("summary", ""))[:200]
                            st.markdown(f"""
                            <div style="background: #1e293b; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; border-left: 3px solid #10b981;">
                                <b>{name}</b> <span style="color: #94a3b8;">({stage})</span><br>
                                <span style="font-size: 0.9rem; color: #cbd5e1;">{summary}...</span>
                            </div>
                            """, unsafe_allow_html=True)
                        if len(results) > 5:
                            st.caption(f"...and {len(results) - 5} more results")
                    
                    # Show pagination info if present
                    if "pagination_meta" in obj:
                        meta = obj["pagination_meta"]
                        st.caption(f"Page {meta.get('current_page', 1)} of {meta.get('total_pages', 1)} | {meta.get('total_count', 0)} total")
                
                # Check if this is an events-style response
                elif isinstance(obj, dict) and "events" in obj:
                    events = obj.get("events", [])
                    st.markdown(f"*{len(events)} event(s) in history*")
                    with st.expander("View Events", expanded=False):
                        for event in events[:3]:
                            intent = event.get("intent", "unknown")
                            timestamp = str(event.get("event_timestamp", ""))[:10]
                            st.markdown(f"- **{intent}** ({timestamp})")
                        if len(events) > 3:
                            st.caption(f"...and {len(events) - 3} more events")
                
                # Generic JSON display - use code block instead of st.json to avoid errors
                else:
                    with st.expander(f"View JSON {'#' + str(i+1) if len(parsed_objects) > 1 else ''}", expanded=len(parsed_objects) == 1):
                        try:
                            formatted = json.dumps(obj, indent=2, ensure_ascii=False)
                            st.code(formatted[:10000], language="json")
                        except Exception as e:
                            st.error(f"Error formatting JSON: {e}")
                            st.code(str(obj)[:5000])
        else:
            # Fallback: show raw output
            with st.expander("View Raw Output", expanded=True):
                st.code(output[:5000] if len(output) > 5000 else output)


def render_labeling_interface():
    """Render main labeling interface."""
    username = st.session_state.username
    store = get_data_store(username)
    
    # Get trajectories and labels
    all_trajectories = store.get_all_trajectories()
    labels = store.get_all_labels()
    labeled_ids = {l.get("trajectory_id") for l in labels}
    
    if not all_trajectories:
        st.error("No trajectories found. Make sure data/monaco_traces.jsonl exists.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {username}")
        
        stats = store.get_stats()
        st.markdown(f"""
        **Progress:** {stats['labeled']} / {stats['total_trajectories']} labeled  
        **Remaining:** {stats['remaining']}  
        **Successful:** {stats['successful_count']}  
        **Erroneous:** {stats['erroneous_count']}
        """)
        
        st.progress(stats['completion_rate'])
        
        st.markdown("---")
        
        # Show all toggle
        st.session_state.show_all = st.checkbox(
            "Show already labeled",
            value=st.session_state.show_all
        )
        
        st.markdown("---")
        
        # Navigation
        if st.session_state.show_all:
            display_trajectories = all_trajectories
        else:
            display_trajectories = [t for t in all_trajectories if t["id"] not in labeled_ids]
        
        if not display_trajectories:
            st.success("🎉 All trajectories labeled!")
            if st.button("Show All Trajectories"):
                st.session_state.show_all = True
                st.rerun()
            return
        
        # Ensure index is valid
        if st.session_state.current_trajectory_idx >= len(display_trajectories):
            st.session_state.current_trajectory_idx = 0
        
        st.markdown("### Navigation")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Prev", use_container_width=True):
                if st.session_state.current_trajectory_idx > 0:
                    st.session_state.current_trajectory_idx -= 1
                    st.rerun()
        with col2:
            if st.button("Next →", use_container_width=True):
                if st.session_state.current_trajectory_idx < len(display_trajectories) - 1:
                    st.session_state.current_trajectory_idx += 1
                    st.rerun()
        
        current_idx = st.session_state.current_trajectory_idx
        st.markdown(f"**{current_idx + 1}** of **{len(display_trajectories)}**")
        
        # Jump to trajectory
        jump_idx = st.number_input(
            "Jump to #",
            min_value=1,
            max_value=len(display_trajectories),
            value=current_idx + 1,
            step=1
        )
        if jump_idx - 1 != current_idx:
            st.session_state.current_trajectory_idx = jump_idx - 1
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.username = None
            st.session_state.current_trajectory_idx = 0
            st.rerun()
    
    # Main content
    current_trajectory = display_trajectories[st.session_state.current_trajectory_idx]
    trajectory_id = current_trajectory["id"]
    
    # Check for existing label
    existing_label = store.get_label_for_trajectory(trajectory_id)
    
    # Get LLM prediction
    llm_prediction = store.get_llm_prediction_for_trajectory(trajectory_id)
    
    # Header with trajectory ID
    st.markdown(f"## Trajectory: `{trajectory_id}`")
    
    # Existing label banner
    if existing_label:
        label_val = existing_label.get("label")
        label_name = config.LABELS.get(label_val, "Unknown")
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem;">
            <b>✓ Currently labeled:</b> [{label_val}] {label_name}
            <span style="color: #94a3b8; margin-left: 1rem;">(Click a button below to change)</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Layout: trajectory on left, LLM prediction + labeling on right
    col_main, col_side = st.columns([3, 1])
    
    with col_side:
        # LLM Prediction panel
        st.markdown("### 🤖 LLM Prediction")
        render_llm_prediction(llm_prediction)
        
        st.markdown("---")
        
        # Labeling buttons
        st.markdown("### Your Label")
        st.markdown("*Press `0` for Erroneous, `1` for Successful*")
        
        def apply_label(label_value: int):
            notes = st.session_state.get("label_notes", "")
            store.add_label(trajectory_id, label_value, notes)
            
            # Advance to next
            if st.session_state.current_trajectory_idx < len(display_trajectories) - 1:
                st.session_state.current_trajectory_idx += 1
            else:
                st.session_state.current_trajectory_idx = 0
            
            st.session_state.pending_advance = True
        
        # Handle pending advance (show feedback briefly)
        if st.session_state.pending_advance:
            st.success("✓ Label saved!")
            time.sleep(0.5)
            st.session_state.pending_advance = False
            st.rerun()
        
        # Error button (0)
        is_current_error = existing_label and existing_label.get("label") == 0
        if st.button(
            "[0] ❌ Erroneous",
            use_container_width=True,
            type="primary" if is_current_error else "secondary",
            key="btn_error"
        ):
            apply_label(0)
            st.rerun()
        
        # Success button (1)
        is_current_success = existing_label and existing_label.get("label") == 1
        if st.button(
            "[1] ✅ Successful",
            use_container_width=True,
            type="primary" if is_current_success else "secondary",
            key="btn_success"
        ):
            apply_label(1)
            st.rerun()
        
        st.markdown("---")
        
        # Notes
        notes_value = existing_label.get("notes", "") if existing_label else ""
        st.text_area(
            "Notes (optional)",
            value=notes_value,
            key="label_notes",
            height=100
        )
    
    with col_main:
        render_trajectory(current_trajectory)


def main():
    """Main application entry point."""
    init_session_state()
    
    if st.session_state.username:
        render_labeling_interface()
    else:
        render_login()


if __name__ == "__main__":
    main()
