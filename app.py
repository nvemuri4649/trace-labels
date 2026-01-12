"""
🔬 AI Trajectory Labeling Tool
A collaborative Streamlit app for labeling AI agent trajectories
"""

import streamlit as st
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from data_store import get_data_store

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Trajectory Labeling Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    /* Import unique fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-tertiary: #1f2937;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #f43f5e;
        --accent-violet: #8b5cf6;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border-subtle: #374151;
    }
    
    /* Main app styling */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Code blocks and trajectories */
    code, pre, .trajectory-content {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Main header banner */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid #3730a3;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(99, 102, 241, 0.15);
    }
    
    .main-header h1 {
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-color: #6366f1;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #06b6d4, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Trajectory display box */
    .trajectory-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .trajectory-box::-webkit-scrollbar {
        width: 8px;
    }
    
    .trajectory-box::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 4px;
    }
    
    .trajectory-box::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    /* Message styling */
    .message-container {
        margin: 0.75rem 0;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .message-system {
        background: rgba(139, 92, 246, 0.1);
        border-left: 3px solid #8b5cf6;
    }
    
    .message-user {
        background: rgba(6, 182, 212, 0.1);
        border-left: 3px solid #06b6d4;
    }
    
    .message-assistant {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10b981;
    }
    
    .message-role {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    /* User welcome card */
    .user-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid #4f46e5;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Task info card */
    .task-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .task-title {
        color: #f1f5f9;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .task-description {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Sidebar enhancements */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Label option cards */
    .label-option {
        background: #1e293b;
        border: 2px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .label-option:hover {
        border-color: #6366f1;
        background: #262f40;
    }
    
    .label-option.selected {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.1);
    }
    
    .label-option-title {
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.25rem;
    }
    
    .label-option-desc {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    /* Progress bar */
    .progress-container {
        background: #1e293b;
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #06b6d4, #8b5cf6);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)


# ==================== INITIALIZE SESSION STATE ====================

def init_session_state():
    """Initialize session state variables"""
    if "username" not in st.session_state:
        st.session_state.username = None
    if "current_trajectory_idx" not in st.session_state:
        st.session_state.current_trajectory_idx = 0
    if "data_store" not in st.session_state:
        st.session_state.data_store = get_data_store()
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "label"
    if "quick_label" not in st.session_state:
        st.session_state.quick_label = None


init_session_state()


# ==================== HELPER FUNCTIONS ====================

import re

def escape_html(text: str) -> str:
    """Escape HTML special characters"""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def parse_output(output: str) -> str:
    """
    Parse the output field which is often a JSON-encoded string.
    Returns the decoded content.
    """
    if not output:
        return ""
    
    # Try to decode JSON string (output might be wrapped in quotes)
    try:
        if output.startswith('"'):
            decoded = json.loads(output)
            if isinstance(decoded, str):
                return decoded
    except:
        pass
    
    return output


def parse_concatenated_json(text: str) -> list:
    """
    Parse concatenated JSON objects (no commas/newlines between them).
    Returns a list of parsed objects.
    """
    results = []
    if not text or not text.strip().startswith('{'):
        return results
    
    # Track brace depth to find object boundaries
    depth = 0
    start = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                # Found complete object
                try:
                    obj = json.loads(text[start:i+1])
                    results.append(obj)
                except:
                    pass
    
    return results


def format_markdown_content(text: str) -> str:
    """
    Convert markdown-style content to HTML for display.
    Handles entity links, bold, lists, etc.
    """
    if not text:
        return ""
    
    # Escape HTML first
    text = escape_html(text)
    
    # Convert entity links [Name](entity=type;entityId=id) to styled spans
    def replace_entity_link(match):
        name = match.group(1)
        return f'<span style="color: #60a5fa; font-weight: 500;">{name}</span>'
    
    text = re.sub(r'\[([^\]]+)\]\(entity=[^)]+\)', replace_entity_link, text)
    
    # Convert regular markdown links [text](url) to styled spans
    def replace_link(match):
        name = match.group(1)
        url = match.group(2)
        if url.startswith('http'):
            return f'<a href="{url}" target="_blank" style="color: #60a5fa;">{name}</a>'
        return f'<span style="color: #60a5fa;">{name}</span>'
    
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, text)
    
    # Convert **bold** to <strong>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong style="color: #f1f5f9;">\1</strong>', text)
    
    # Convert *italic* to <em>
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
    
    # Convert `code` to <code>
    text = re.sub(r'`([^`]+)`', r'<code style="background: #374151; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.85em;">\1</code>', text)
    
    # Convert line breaks
    text = text.replace('\n\n', '</p><p style="margin: 0.75rem 0;">')
    text = text.replace('\n- ', '</p><div style="margin: 0.25rem 0 0.25rem 1rem;">• ')
    text = text.replace('\n', '<br>')
    
    # Clean up list items
    text = re.sub(r'</p><div style="margin: 0.25rem 0 0.25rem 1rem;">• ([^<]+)(?=</p>|<br>|$)', 
                  r'</p><div style="margin: 0.25rem 0 0.25rem 1rem;">• \1</div>', text)
    
    return f'<p style="margin: 0.75rem 0;">{text}</p>'


def parse_conversation(messages: list) -> dict:
    """
    Parse messages into structured conversation components.
    Returns dict with: system_prompt, user_queries, conversation_turns, final_answer
    """
    result = {
        "system_prompt": "",
        "user_queries": [],
        "conversation_turns": [],
        "final_answer": "",
        "tool_calls_count": 0,
        "tool_responses_count": 0,
    }
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
            
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id", "")
        
        # System prompt
        if role == "system":
            result["system_prompt"] = content
            continue
        
        # Skip empty user messages
        if role == "user" and not content.strip():
            continue
        
        # User message
        if role == "user" and content.strip():
            result["user_queries"].append(content.strip())
            result["conversation_turns"].append({
                "type": "user",
                "content": content.strip()
            })
        
        # Assistant with tool calls
        elif role == "assistant" and tool_calls:
            tools_called = []
            for tc in tool_calls:
                func = tc.get("function", {})
                tools_called.append({
                    "name": func.get("name", "unknown"),
                    "id": tc.get("id", "")[:20]
                })
            result["tool_calls_count"] += len(tools_called)
            result["conversation_turns"].append({
                "type": "tool_call",
                "tools": tools_called,
                "content": content if content else None
            })
        
        # Assistant response (text)
        elif role == "assistant" and content:
            result["conversation_turns"].append({
                "type": "assistant",
                "content": content
            })
            result["final_answer"] = content  # Last assistant response
        
        # Tool response
        elif role == "tool":
            result["tool_responses_count"] += 1
            result["conversation_turns"].append({
                "type": "tool_response",
                "tool_call_id": tool_call_id[:20] if tool_call_id else "",
                "content": content
            })
    
    return result


def render_trajectory(trajectory: dict):
    """Render a trajectory with improved formatting"""
    messages = trajectory.get("messages", [])
    output = trajectory.get("output", "")
    
    # Parse the conversation
    parsed = parse_conversation(messages)
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💬 User Queries", len(parsed["user_queries"]))
    with col2:
        st.metric("🔧 Tool Calls", parsed["tool_calls_count"])
    with col3:
        st.metric("📊 Tool Responses", parsed["tool_responses_count"])
    with col4:
        content_len = trajectory.get("content_length", 0)
        st.metric("📏 Length", f"{content_len:,} chars")
    
    # System prompt (collapsed by default)
    if parsed["system_prompt"]:
        with st.expander("🤖 System Instructions (click to expand)", expanded=False):
            st.markdown(f"""
            <div style="background: rgba(139, 92, 246, 0.1); border-left: 3px solid #8b5cf6; padding: 1rem; border-radius: 8px; font-size: 0.85rem; color: #cbd5e1; max-height: 300px; overflow-y: auto;">
                {escape_html(parsed["system_prompt"][:2000])}{'...' if len(parsed["system_prompt"]) > 2000 else ''}
            </div>
            """, unsafe_allow_html=True)
    
    # Main conversation flow
    st.markdown("### 💬 Conversation Flow")
    
    # Show the conversation turns
    for i, turn in enumerate(parsed["conversation_turns"]):
        turn_type = turn["type"]
        
        if turn_type == "user":
            # User query - prominent blue box
            user_content = format_markdown_content(turn["content"])
            st.markdown(f"""
            <div style="background: rgba(6, 182, 212, 0.15); border-left: 4px solid #06b6d4; padding: 1rem; border-radius: 8px; margin: 0.75rem 0;">
                <div style="color: #06b6d4; font-weight: 600; font-size: 0.8rem; margin-bottom: 0.5rem;">👤 USER</div>
                <div style="color: #f1f5f9; font-size: 0.95rem; line-height: 1.6;">{user_content}</div>
            </div>
            """, unsafe_allow_html=True)
        
        elif turn_type == "tool_call":
            # Tool calls - compact amber indicator
            tools_str = ", ".join([f"<code>{t['name']}</code>" for t in turn["tools"]])
            st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 0.75rem 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <span style="color: #f59e0b; font-weight: 600; font-size: 0.8rem;">🔧 TOOL CALLS:</span>
                <span style="color: #fcd34d; font-size: 0.85rem; margin-left: 0.5rem;">{tools_str}</span>
            </div>
            """, unsafe_allow_html=True)
        
        elif turn_type == "tool_response":
            # Tool responses - collapsible gray section
            content_preview = turn["content"][:200] if turn["content"] else "(empty)"
            with st.expander(f"📊 Tool Response ({len(turn['content']):,} chars)", expanded=False):
                # Try to parse as JSON for better formatting
                try:
                    if turn["content"].startswith('{') or turn["content"].startswith('['):
                        parsed_json = json.loads(turn["content"])
                        st.json(parsed_json)
                    else:
                        st.code(turn["content"][:3000] + ("..." if len(turn["content"]) > 3000 else ""), language="text")
                except:
                    st.code(turn["content"][:3000] + ("..." if len(turn["content"]) > 3000 else ""), language="text")
        
        elif turn_type == "assistant":
            # Assistant response - green box with formatted markdown
            content = turn["content"]
            is_final = (i == len(parsed["conversation_turns"]) - 1)
            
            if is_final:
                st.markdown("### 🎯 Final Answer")
            
            # Format the content with markdown support
            formatted_content = format_markdown_content(content[:4000])
            if len(content) > 4000:
                formatted_content += '<p style="color: #94a3b8; font-style: italic;">... [truncated]</p>'
            
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.12); border-left: 4px solid #10b981; padding: 1rem; border-radius: 8px; margin: 0.75rem 0; {'border: 2px solid #10b981; box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);' if is_final else ''}">
                <div style="color: #10b981; font-weight: 600; font-size: 0.8rem; margin-bottom: 0.5rem;">🤖 ASSISTANT{' (FINAL)' if is_final else ''}</div>
                <div style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.6;">{formatted_content}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Parse and display the final output properly
    decoded_output = parse_output(output)
    
    # Check if we need to show the output section
    show_output = False
    if decoded_output:
        # Show if the conversation didn't end with an assistant message
        if not parsed["conversation_turns"] or parsed["conversation_turns"][-1]["type"] != "assistant":
            show_output = True
        # Or if output is different from final answer
        elif decoded_output != parsed["final_answer"]:
            show_output = True
    
    if show_output and decoded_output:
        # Determine if it's text or structured data
        is_structured = decoded_output.strip().startswith('{') or decoded_output.strip().startswith('[')
        
        if is_structured:
            parsed_json = None
            
            # First, try standard JSON parsing
            try:
                parsed_json = json.loads(decoded_output)
            except json.JSONDecodeError:
                # Might be concatenated JSON objects - try parsing them
                parsed_objects = parse_concatenated_json(decoded_output)
                if parsed_objects:
                    # Wrap in results format for consistent handling
                    parsed_json = {"results": parsed_objects}
            
            if parsed_json:
                
                # Flatten results - collect all result items from nested structures
                all_results = []
                
                def extract_results(obj, depth=0):
                    """Recursively extract results from nested JSON structures"""
                    if depth > 3:  # Prevent infinite recursion
                        return
                    if isinstance(obj, dict):
                        # If it has a "results" key with a list, extract those
                        if "results" in obj and isinstance(obj["results"], list):
                            for item in obj["results"]:
                                if isinstance(item, dict):
                                    all_results.append(item)
                        # Otherwise, if this object looks like a result item itself
                        elif obj.get("title") or obj.get("name") or obj.get("indexed_text"):
                            all_results.append(obj)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_results(item, depth + 1)
                
                # Handle list of objects (from concatenated JSON)
                if isinstance(parsed_json, dict) and "results" in parsed_json:
                    # Already has results array
                    extract_results(parsed_json)
                elif isinstance(parsed_json, list):
                    for item in parsed_json:
                        extract_results(item)
                else:
                    extract_results(parsed_json)
                
                if all_results:
                    st.markdown("### 📊 Output Results")
                    st.markdown(f"<p style='color: #94a3b8; font-size: 0.85rem;'>{len(all_results)} result(s) returned</p>", unsafe_allow_html=True)
                    
                    # Display each result as a card
                    for i, result in enumerate(all_results[:10]):  # Limit to first 10
                        result_type = result.get("result_type", result.get("type", result.get("document_type", "result"))) or "result"
                        title = result.get("title", result.get("name", result.get("account_name", f"Result {i+1}")))
                        
                        # Build card content based on result type
                        card_items = []
                        
                        # Summary/description text
                        text_content = result.get("indexed_text") or result.get("summary") or result.get("description") or ""
                        if text_content:
                            card_items.append(("📝", text_content[:250] + "..." if len(text_content) > 250 else text_content))
                        
                        # Participants
                        if result.get("participants"):
                            card_items.append(("👥", ", ".join(str(p) for p in result["participants"][:5])))
                        
                        # Date/time
                        if result.get("start_time"):
                            card_items.append(("📅", str(result["start_time"])[:16].replace("T", " ")))
                        
                        # Stage
                        if result.get("stage"):
                            card_items.append(("📊", f"Stage: {result['stage']}"))
                        
                        # Account names
                        if result.get("account_names"):
                            card_items.append(("🏢", ", ".join(str(a) for a in result["account_names"][:3])))
                        elif result.get("account_name"):
                            card_items.append(("🏢", str(result["account_name"])))
                        
                        # Render the card
                        items_html = "".join([
                            f'<div style="margin: 0.4rem 0; color: #cbd5e1; font-size: 0.85rem;"><span style="color: #60a5fa;">{icon}</span> {escape_html(str(text))}</div>'
                            for icon, text in card_items
                        ])
                        
                        # Choose badge color based on result type
                        badge_color = "#4f46e5"  # default indigo
                        if "transcript" in str(result_type).lower():
                            badge_color = "#059669"  # green
                        elif "meeting" in str(result_type).lower():
                            badge_color = "#0891b2"  # cyan
                        elif "account" in str(result_type).lower():
                            badge_color = "#d97706"  # amber
                        
                        st.markdown(f"""
                        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid #4f46e5; border-radius: 10px; padding: 1rem; margin: 0.75rem 0;">
                            <div style="display: flex; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
                                <span style="background: {badge_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem;">{escape_html(str(result_type).upper()[:20])}</span>
                                <span style="color: #f1f5f9; font-weight: 600;">{escape_html(str(title)[:60])}</span>
                            </div>
                            {items_html if items_html else '<div style="color: #64748b; font-size: 0.85rem; font-style: italic;">No additional details</div>'}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(all_results) > 10:
                        st.markdown(f"<p style='color: #94a3b8; font-style: italic;'>... and {len(all_results) - 10} more results</p>", unsafe_allow_html=True)
                    
                    # Also provide raw JSON in expander
                    with st.expander("🔍 View Raw JSON", expanded=False):
                        st.json(parsed_json)
                else:
                    # Single object - show in expander
                    with st.expander("📤 Output Data (JSON)", expanded=False):
                        st.json(parsed_json)
            else:
                # Couldn't parse as JSON - show raw
                with st.expander("📤 Raw Output Data", expanded=False):
                    st.code(decoded_output[:5000], language="json")
        else:
            # It's text - this is likely the actual final answer
            st.markdown("### 🎯 Final Output")
            
            formatted_output = format_markdown_content(decoded_output[:5000])
            if len(decoded_output) > 5000:
                formatted_output += '<p style="color: #94a3b8; font-style: italic;">... [truncated]</p>'
            
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; padding: 1.25rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);">
                <div style="color: #10b981; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.75rem;">🤖 FINAL RESPONSE</div>
                <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.7;">{formatted_output}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Metadata
    with st.expander("📊 Trace Metadata", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text(f"Trace ID: {trajectory.get('id', 'N/A')}")
        with col2:
            st.text(f"Span: {trajectory.get('span_name', 'N/A')}")
        with col3:
            duration = trajectory.get("duration", 0)
            if duration:
                # Convert nanoseconds to seconds
                duration_sec = duration / 1_000_000_000
                st.text(f"Duration: {duration_sec:.2f}s")
        
        if trajectory.get("error"):
            st.error(f"Error: {trajectory.get('error')}")


# ==================== LOGIN PAGE ====================

def render_login():
    """Render the login page for allowed labelers"""
    st.markdown("""
    <div class="main-header">
        <h1>🔬 AI Trajectory Labeling Tool</h1>
        <p>Collaborative annotation platform for AI agent behavior analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 👋 Welcome, Labeler!")
        st.markdown("Select your name to start labeling your assigned trajectories.")
        
        # Show allowed labelers as buttons
        st.markdown("---")
        
        for labeler in config.ALLOWED_LABELERS:
            if st.button(f"🏷️ Log in as **{labeler.title()}**", key=f"login_{labeler}", use_container_width=True):
                st.session_state.username = labeler
                
                # Register/update user
                st.session_state.data_store.add_user(labeler, {
                    "display_name": labeler.title(),
                    "role": "labeler"
                })
                
                st.rerun()
        
        st.markdown("---")
        
        # Show assignment info
        st.markdown("#### 📊 Data Assignment")
        store = st.session_state.data_store
        stats = store.get_labeling_stats()
        
        for labeler in config.ALLOWED_LABELERS:
            user_stats = stats["user_stats"].get(labeler, {})
            assigned = user_stats.get("assigned", 0)
            labeled = user_stats.get("labeled", 0)
            
            progress = labeled / assigned if assigned > 0 else 0
            st.markdown(f"**{labeler.title()}**: {labeled}/{assigned} labeled ({progress*100:.0f}%)")
            st.progress(progress)


# ==================== MAIN LABELING INTERFACE ====================

def render_labeling_interface():
    """Render the main labeling interface"""
    store = st.session_state.data_store
    username = st.session_state.username
    
    # Initialize view mode for showing all vs unlabeled
    if "show_all_trajectories" not in st.session_state:
        st.session_state.show_all_trajectories = False
    
    # Get trajectories for this user
    unlabeled = store.get_unlabeled_trajectories_for_user(username)
    user_trajectories = store.get_trajectories_for_user(username)
    user_labels = store.get_labels_by_user(username)
    
    # Create a lookup for existing labels
    labels_by_trajectory = {l.get("trajectory_id"): l for l in user_labels}
    
    # Header with user info
    st.markdown(f"""
    <div class="user-card">
        <span style="font-size: 1.5rem;">👤</span>
        <span style="color: #e2e8f0; font-weight: 600; margin-left: 0.5rem;">Welcome back, {username.title()}!</span>
        <span style="color: #94a3b8; margin-left: 1rem;">
            {len(user_labels)} labeled | {len(unlabeled)} remaining
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    if user_trajectories:
        progress = len(user_labels) / len(user_trajectories)
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #94a3b8; font-size: 0.9rem;">Your Progress</span>
                <span style="color: #06b6d4; font-weight: 600;">{progress*100:.1f}%</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress*100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Check if there are trajectories to label
    if not user_trajectories:
        st.warning("⚠️ No trajectories assigned to you. Please contact the admin.")
        return
    
    # Toggle to show all trajectories vs just unlabeled
    col_toggle1, col_toggle2 = st.columns([3, 1])
    with col_toggle2:
        show_all = st.checkbox(
            "Show all", 
            value=st.session_state.show_all_trajectories,
            help="Toggle to see all trajectories including already labeled ones"
        )
        if show_all != st.session_state.show_all_trajectories:
            st.session_state.show_all_trajectories = show_all
            st.session_state.current_trajectory_idx = 0  # Reset to start
            st.rerun()
    
    # Determine which list to show
    if st.session_state.show_all_trajectories:
        display_trajectories = user_trajectories
        list_type = "total"
    else:
        display_trajectories = unlabeled
        list_type = "remaining"
    
    if not display_trajectories:
        if st.session_state.show_all_trajectories:
            st.warning("⚠️ No trajectories assigned to you.")
        else:
            st.success("🎉 Amazing! You've labeled all your assigned trajectories!")
            st.info("Toggle 'Show all' above to review and edit your existing labels.")
        return
    
    # Navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    # Ensure index is within bounds
    if st.session_state.current_trajectory_idx >= len(display_trajectories):
        st.session_state.current_trajectory_idx = 0
    
    current_idx = st.session_state.current_trajectory_idx
    current_trajectory = display_trajectories[current_idx]
    
    # Check if this trajectory already has a label
    existing_label = labels_by_trajectory.get(current_trajectory.get("id"))
    
    with col1:
        if st.button("⬅️ Previous", disabled=current_idx == 0):
            st.session_state.current_trajectory_idx -= 1
            st.rerun()
    
    with col2:
        content_len = current_trajectory.get('content_length', 0)
        len_label = "Long" if content_len > 50000 else "Medium" if content_len > 10000 else "Short"
        
        # Show labeled status
        status_badge = ""
        if existing_label:
            status_badge = f'<span style="background: #059669; color: white; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.7rem; margin-left: 0.5rem;">✓ {existing_label.get("label", "labeled")}</span>'
        
        st.markdown(f"""
        <div style="text-align: center; color: #94a3b8;">
            Trajectory {current_idx + 1} of {len(display_trajectories)} {list_type}{status_badge}
            <span style="color: #64748b; font-size: 0.8rem; display: block;">
                ID: {current_trajectory.get('id', 'unknown')[:20]}... | {len_label} ({content_len:,} chars)
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("Next ➡️", disabled=current_idx >= len(display_trajectories) - 1):
            st.session_state.current_trajectory_idx += 1
            st.rerun()
    
    st.markdown("---")
    
    # Display trajectory
    st.markdown("### 📜 Trajectory Content")
    render_trajectory(current_trajectory)
    
    # Labeling form
    st.markdown("---")
    
    # Show existing label if present
    if existing_label:
        st.markdown(f"""
        <div style="background: rgba(5, 150, 105, 0.15); border: 1px solid #059669; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem;">
            <span style="color: #34d399; font-weight: 600;">✓ Already labeled:</span>
            <span style="color: #f1f5f9; margin-left: 0.5rem;">{existing_label.get('label', 'Unknown')}</span>
            <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 0.5rem;">
                (click a button below to change)
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 🔄 Update Label")
    else:
        st.markdown("### 🏷️ Label This Trajectory")
    
    st.markdown("Press **1-7** for quick labeling, or click a button below:")
    
    # Check for quick label from query params or session state
    if "quick_label" in st.session_state and st.session_state.quick_label is not None:
        quick_label_idx = st.session_state.quick_label
        st.session_state.quick_label = None  # Reset
        
        if 0 <= quick_label_idx < len(config.LABELS):
            # Preserve existing notes if updating
            existing_notes = existing_label.get("notes", "") if existing_label else ""
            label_data = {
                "trajectory_id": current_trajectory.get("id"),
                "labeled_by": username,
                "label": config.LABELS[quick_label_idx],
                "notes": existing_notes,
            }
            store.add_label(label_data)
            
            # Move to next trajectory
            if current_idx < len(display_trajectories) - 1:
                st.session_state.current_trajectory_idx += 1
            else:
                st.session_state.current_trajectory_idx = 0
            st.rerun()
    
    # Quick label buttons with keyboard shortcuts
    st.markdown("""
    <style>
        .quick-label-btn {
            display: inline-flex;
            align-items: center;
            padding: 0.75rem 1rem;
            margin: 0.25rem;
            border-radius: 8px;
            background: #1e293b;
            border: 2px solid #334155;
            color: #f1f5f9;
            cursor: pointer;
            transition: all 0.2s;
            font-family: 'Outfit', sans-serif;
        }
        .quick-label-btn:hover {
            border-color: #6366f1;
            background: #262f40;
        }
        .quick-label-key {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            background: #4f46e5;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.9rem;
            margin-right: 0.75rem;
        }
        .quick-label-text {
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create columns for quick label buttons
    cols = st.columns(4)
    
    for idx, label_name in enumerate(config.LABELS):
        col_idx = idx % 4
        with cols[col_idx]:
            key_num = idx + 1
            desc = config.LABEL_DESCRIPTIONS.get(label_name, "")[:50]
            # Highlight if this is the current label
            is_current = existing_label and existing_label.get("label") == label_name
            button_label = f"[{key_num}] {label_name}" + (" ✓" if is_current else "")
            
            if st.button(
                button_label,
                key=f"quick_label_{idx}",
                use_container_width=True,
                help=desc,
                type="primary" if is_current else "secondary"
            ):
                # Preserve existing notes if updating
                existing_notes = existing_label.get("notes", "") if existing_label else ""
                label_data = {
                    "trajectory_id": current_trajectory.get("id"),
                    "labeled_by": username,
                    "label": label_name,
                    "notes": existing_notes,
                }
                store.add_label(label_data)
                
                action = "Updated" if existing_label else "Labeled"
                st.success(f"✅ {action} as '{label_name}'")
                
                # Move to next trajectory
                if current_idx < len(display_trajectories) - 1:
                    st.session_state.current_trajectory_idx += 1
                else:
                    st.session_state.current_trajectory_idx = 0
                st.rerun()
    
    # Label descriptions reference
    with st.expander("📖 Label Descriptions"):
        for idx, label_name in enumerate(config.LABELS):
            st.markdown(f"**[{idx+1}] {label_name}**: {config.LABEL_DESCRIPTIONS.get(label_name, '')}")
    
    # JavaScript for keyboard shortcuts
    st.markdown(f"""
    <script>
        // Keyboard shortcut handler for quick labeling
        document.addEventListener('keydown', function(e) {{
            // Only trigger if not in an input/textarea
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            const key = e.key;
            if (key >= '1' && key <= '7') {{
                const labelIdx = parseInt(key) - 1;
                // Find and click the corresponding button
                const buttons = document.querySelectorAll('button[kind="secondary"]');
                buttons.forEach(btn => {{
                    if (btn.textContent.startsWith('[' + key + ']')) {{
                        btn.click();
                    }}
                }});
            }}
        }});
    </script>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Optional: Full form with notes for detailed labeling
    expander_title = "📝 Edit Label with Notes" if existing_label else "📝 Add Notes with Label"
    with st.expander(expander_title):
        # Get existing values for pre-filling
        existing_label_value = existing_label.get("label", config.LABELS[0]) if existing_label else config.LABELS[0]
        existing_label_idx = config.LABELS.index(existing_label_value) if existing_label_value in config.LABELS else 0
        existing_notes_value = existing_label.get("notes", "") if existing_label else ""
        
        with st.form("labeling_form", clear_on_submit=False):
            label = st.radio(
                "Classification",
                options=config.LABELS,
                index=existing_label_idx,
                help="Choose Success if the task was completed correctly, or select the most appropriate failure mode",
                format_func=lambda x: f"[{config.LABELS.index(x)+1}] {x}"
            )
            
            notes = st.text_area(
                "Notes",
                value=existing_notes_value,
                placeholder="Any observations or reasoning for your label...",
            )
            
            button_text = "💾 Update Label" if existing_label else "✅ Submit with Notes"
            submitted = st.form_submit_button(button_text, use_container_width=True, type="primary")
            
            if submitted:
                label_data = {
                    "trajectory_id": current_trajectory.get("id"),
                    "labeled_by": username,
                    "label": label,
                    "notes": notes,
                }
                
                store.add_label(label_data)
                action = "updated" if existing_label else "saved"
                st.success(f"✅ Label {action} successfully!")
                
                # Move to next trajectory
                if current_idx < len(display_trajectories) - 1:
                    st.session_state.current_trajectory_idx += 1
                else:
                    st.session_state.current_trajectory_idx = 0
                
                st.rerun()
    
    # Skip button
    if st.button("⏭️ Skip (come back later)", help="Skip this trajectory and move to the next one"):
        if current_idx < len(display_trajectories) - 1:
            st.session_state.current_trajectory_idx += 1
        else:
            st.session_state.current_trajectory_idx = 0
        st.rerun()


# ==================== REVIEW INTERFACE ====================

def render_review_interface():
    """Render the label review interface"""
    store = st.session_state.data_store
    username = st.session_state.username
    
    st.markdown("### 📝 Review Your Labels")
    
    if st.button("⬅️ Back to Labeling"):
        st.session_state.view_mode = "label"
        st.rerun()
    
    user_labels = store.get_labels_by_user(username)
    
    if not user_labels:
        st.info("You haven't labeled any trajectories yet.")
        return
    
    # Create dataframe for display
    df = pd.DataFrame(user_labels)
    
    # Display columns we care about
    display_cols = ["trajectory_id", "label", "notes", "labeled_at"]
    df_display = df[[c for c in display_cols if c in df.columns]]
    
    st.dataframe(df_display, use_container_width=True)
    
    # Select label to edit
    st.markdown("---")
    st.markdown("#### Edit a Label")
    
    selected_id = st.selectbox(
        "Select trajectory to edit",
        options=[l.get("trajectory_id") for l in user_labels]
    )
    
    if selected_id:
        trajectory = store.get_trajectory(selected_id)
        existing_label = next((l for l in user_labels if l.get("trajectory_id") == selected_id), None)
        
        if trajectory:
            with st.expander("📜 View Trajectory", expanded=False):
                render_trajectory(trajectory)
        
        if existing_label:
            with st.form("edit_form"):
                new_label = st.radio(
                    "Classification",
                    options=config.LABELS,
                    index=config.LABELS.index(existing_label.get("label", config.LABELS[0])) if existing_label.get("label") in config.LABELS else 0,
                    label_visibility="collapsed"
                )
                
                st.info(f"**{new_label}**: {config.LABEL_DESCRIPTIONS.get(new_label, '')}")
                
                new_notes = st.text_area(
                    "Notes",
                    value=existing_label.get("notes", "")
                )
                
                if st.form_submit_button("💾 Update Label", use_container_width=True):
                    updated_label = {
                        "trajectory_id": selected_id,
                        "labeled_by": username,
                        "label": new_label,
                        "notes": new_notes,
                    }
                    store.add_label(updated_label)
                    st.success("✅ Label updated!")
                    st.rerun()


# ==================== DASHBOARD ====================

def render_dashboard():
    """Render the analytics dashboard"""
    store = st.session_state.data_store
    stats = store.get_labeling_stats()
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 Labeling Dashboard</h1>
        <p>Real-time progress tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_trajectories']}</div>
            <div class="stat-label">Total Trajectories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_labels']}</div>
            <div class="stat-label">Total Labels</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        completion = stats['total_labels'] / stats['total_trajectories'] if stats['total_trajectories'] > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{completion*100:.0f}%</div>
            <div class="stat-label">Overall Completion</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Per-labeler progress
    st.markdown("### 👥 Progress by Labeler")
    
    user_stats = stats.get("user_stats", {})
    
    for labeler in config.ALLOWED_LABELERS:
        labeler_stats = user_stats.get(labeler, {})
        assigned = labeler_stats.get("assigned", 0)
        labeled = labeler_stats.get("labeled", 0)
        remaining = labeler_stats.get("remaining", 0)
        completion_rate = labeler_stats.get("completion_rate", 0)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**{labeler.title()}**")
        with col2:
            st.progress(completion_rate)
            st.caption(f"{labeled}/{assigned} labeled • {remaining} remaining")
    
    # Label distribution
    if stats['total_labels'] > 0:
        st.markdown("### 📊 Label Distribution")
        
        labels = store.get_all_labels()
        df = pd.DataFrame(labels)
        
        if "label" in df.columns:
            # Pie chart
            fig = px.pie(
                df,
                names="label",
                title="Distribution of Labels",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Labels by user
            st.markdown("### 📈 Labels by Labeler")
            
            user_label_counts = df.groupby(['labeled_by', 'label']).size().reset_index(name='count')
            
            fig = px.bar(
                user_label_counts,
                x="labeled_by",
                y="count",
                color="label",
                title="Label Counts by Labeler",
                barmode="stack"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No labels recorded yet. Start labeling to see analytics!")


# ==================== ADMIN PANEL ====================

def render_admin_panel():
    """Render admin panel in sidebar"""
    store = st.session_state.data_store
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Admin Panel")
    
    with st.sidebar.expander("📊 Quick Stats"):
        stats = store.get_labeling_stats()
        
        st.metric("Total Trajectories", stats["total_trajectories"])
        st.metric("Total Labels", stats["total_labels"])
        
        for labeler in config.ALLOWED_LABELERS:
            user_stats = stats["user_stats"].get(labeler, {})
            st.caption(f"{labeler.title()}: {user_stats.get('labeled', 0)}/{user_stats.get('assigned', 0)}")
    
    with st.sidebar.expander("📤 Export Data"):
        labels = store.get_all_labels()
        if labels:
            labels_json = json.dumps(labels, indent=2, default=str)
            st.download_button(
                "📥 Download Labels (JSON)",
                labels_json,
                "labels.json",
                "application/json",
                use_container_width=True
            )
            
            # CSV export
            df = pd.DataFrame(labels)
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Labels (CSV)",
                csv,
                "labels.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No labels to export yet")


# ==================== MAIN APP ====================

def main():
    """Main app entry point"""
    # Sidebar navigation
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">🔬</span>
        <h2 style="margin: 0.5rem 0;">Trajectory Labeling</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Not logged in
    if not st.session_state.username:
        render_login()
        return
    
    # Check if user is allowed
    if st.session_state.username not in config.ALLOWED_LABELERS:
        st.error(f"User '{st.session_state.username}' is not authorized. Please log in as one of: {', '.join(config.ALLOWED_LABELERS)}")
        st.session_state.username = None
        st.rerun()
        return
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏷️ Label Trajectories", "📊 Dashboard", "📝 Review Labels"],
        label_visibility="collapsed"
    )
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.username = None
        st.session_state.current_trajectory_idx = 0
        st.rerun()
    
    # Admin panel
    render_admin_panel()
    
    # Main content
    if page == "🏷️ Label Trajectories":
        if st.session_state.view_mode == "review":
            render_review_interface()
        else:
            render_labeling_interface()
    elif page == "📊 Dashboard":
        render_dashboard()
    elif page == "📝 Review Labels":
        st.session_state.view_mode = "review"
        render_review_interface()


if __name__ == "__main__":
    main()
