#!/usr/bin/env python3
"""
LLM-as-Judge: Evaluate agent trajectories using GPT-5.2.

Runs 5 iterations of the LLM on each trajectory and takes a majority vote.
Results are saved to data/llm_predictions.json.

Usage:
    python llm_judge.py                    # Evaluate all trajectories
    python llm_judge.py --limit 10         # Evaluate first 10 trajectories
    python llm_judge.py --concurrency 10   # Run 10 concurrent evaluations
    python llm_judge.py --verbose          # Show detailed progress
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import openai
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    raise

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import config

# Semaphore for rate limiting
LLM_SEMAPHORE: asyncio.Semaphore

# -----------------------------------------------------------------------------
# Structured Outputs (JSON Schema)
# -----------------------------------------------------------------------------

JUDGE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the judgment. Must appear before score in the JSON text.",
        },
        "score": {
            "type": "integer",
            "enum": [0, 1],
            "description": "Binary success label: 1 if completed successfully, 0 if erroneous.",
        },
        "task_analysis": {
            "type": "string",
            "description": "What the agent was supposed to do.",
        },
        "completion_evidence": {
            "type": "string",
            "description": "Evidence of completion or failure from the trajectory.",
        },
        "errors_found": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of errors found (possibly empty).",
        },
        "error_type": {
            "type": "string",
            "enum": [
                "none",
                "hallucination",
                "underspecified_tool_semantics",
                "incorrect_tool_output",
                "incomplete_response",
                "refusal",
                "other"
            ],
            "description": "Primary error type if score is 0. Use 'none' if score is 1.",
        },
        "error_citation": {
            "type": "string",
            "description": "If erroneous, quote the EXACT text from the trajectory that demonstrates the error. Include message number [XXX] for reference. Empty string if no error.",
        },
        "error_explanation": {
            "type": "string",
            "description": "If erroneous, explain WHY the cited text is wrong, including what the correct information should be if applicable. Empty string if no error.",
        },
    },
    "required": ["reasoning", "score", "task_analysis", "completion_evidence", "errors_found", "error_type", "error_citation", "error_explanation"],
}

JUDGE_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_eval",
        "schema": JUDGE_JSON_SCHEMA,
        "strict": True,
    },
}


@dataclass
class IterationResult:
    """Result of a single LLM evaluation iteration."""
    score: Optional[int] = None
    reasoning: str = ""
    raw_response: str = ""
    error: Optional[str] = None
    time_s: float = 0.0
    # Error details (populated when score=0)
    error_type: str = "none"
    error_citation: str = ""
    error_explanation: str = ""


@dataclass
class TrajectoryResult:
    """Result of evaluating a trajectory with multiple iterations."""
    trajectory_id: str
    iterations: List[IterationResult] = field(default_factory=list)
    final_score: Optional[int] = None
    confidence: float = 0.0
    total_time_s: float = 0.0


# -----------------------------------------------------------------------------
# Trajectory Loading
# -----------------------------------------------------------------------------

def load_trajectories(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load trajectories from JSONL file."""
    trajectories = []
    jsonl_path = Path(config.DATA_DIR) / config.TRAJECTORIES_FILE
    
    if not jsonl_path.exists():
        print(f"Error: Trajectories file not found: {jsonl_path}")
        return []
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f):
            if limit and len(trajectories) >= limit:
                break
                
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
                
                trajectory = {
                    "id": trace_id,
                    "task": task or "No task description",
                    "messages": messages,
                    "output": trace.get('output', ''),
                    "raw_trace": trace,
                }
                trajectories.append(trajectory)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(trajectories)} trajectories")
    return trajectories


# -----------------------------------------------------------------------------
# Trajectory Formatting
# -----------------------------------------------------------------------------

def format_trajectory_as_text(trajectory: Dict[str, Any]) -> str:
    """Format a trajectory as human-readable text for the LLM."""
    lines = []
    task_id = trajectory.get("id", "unknown")
    task = trajectory.get("task", "")
    
    lines.append("=" * 80)
    lines.append(f"TRAJECTORY: {task_id}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Task: {task}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("CONVERSATION:")
    lines.append("-" * 80)
    
    for i, msg in enumerate(trajectory.get("messages", [])):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        
        lines.append(f"\n[{i:03d}] {role}")
        
        if content:
            if len(content) > 2000:
                lines.append(f"     {content[:2000]}...")
                lines.append(f"     [...truncated, {len(content)} total chars]")
            else:
                lines.append(f"     {content}")
        
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                lines.append(f"     -> Tool Call: {func.get('name', 'unknown')}")
                args = func.get("arguments", "{}")
                if len(args) > 500:
                    lines.append(f"        Args: {args[:500]}...")
                else:
                    lines.append(f"        Args: {args}")
    
    # Add output if present
    output = trajectory.get("output", "")
    if output:
        lines.append("")
        lines.append("-" * 80)
        lines.append("FINAL OUTPUT:")
        lines.append("-" * 80)
        if len(output) > 3000:
            lines.append(output[:3000] + "...[truncated]")
        else:
            lines.append(output)
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Score Extraction
# -----------------------------------------------------------------------------

def extract_score_from_response(response: str) -> Optional[int]:
    """Extract a binary score (0 or 1) from LLM response."""
    if not response:
        return None
    
    # Try to find "score" field in JSON
    score_field_match = re.search(r'"score"\s*:\s*"?([01])(?:\.0+)?\"?', response)
    if score_field_match:
        try:
            return int(score_field_match.group(1))
        except ValueError:
            pass
    
    # Try JSON in code blocks
    fenced_json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE)
    if fenced_json_match:
        try:
            data = json.loads(fenced_json_match.group(1))
            score = data.get("score")
            if score is not None:
                s = int(score)
                return s if s in (0, 1) else None
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    # Try to find any JSON object with score
    candidates = re.findall(r"\{[\s\S]*?\}", response)
    for cand in reversed(candidates[-10:]):
        if '"score"' not in cand:
            continue
        try:
            data = json.loads(cand)
            score = data.get("score")
            if score is not None:
                s = int(score)
                return s if s in (0, 1) else None
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    
    return None


# -----------------------------------------------------------------------------
# Judge Prompt
# -----------------------------------------------------------------------------

def build_judge_prompt(trajectory: Dict[str, Any], trajectory_text: str) -> str:
    """Build the judge prompt for trajectory evaluation."""
    task = trajectory.get("task", "")
    
    return f"""You are an expert evaluator judging whether an AI agent successfully completed its task.

## Task Context
The agent was given this task: {task}

## Agent Trajectory
{trajectory_text}

## Error Modes to Check For

When evaluating, look for these specific error types:

### 1. HALLUCINATION
The agent claims something that is NOT supported by the tool outputs or conversation.
- Agent states facts not present in any tool response
- Agent confidently reports information that contradicts tool outputs
- Agent makes up dates, names, numbers, or details not in the data

### 2. UNDERSPECIFIED_TOOL_SEMANTICS  
The agent misuses tools due to unclear tool behavior/parameters.
- Agent assumes ordering (e.g., "recent" vs "oldest") not guaranteed by tool
- Agent expects tool behavior that isn't specified
- Tool parameters are misunderstood or misused

### 3. INCORRECT_TOOL_OUTPUT
The tool itself returned incorrect or inconsistent data.
- Tool response contains factual errors
- Tool response is internally inconsistent
- Tool swapped entities or confused data

### 4. INCOMPLETE_RESPONSE
The agent started work but failed to provide a complete answer.
- Tool calls were made but no final answer given
- Agent's response trails off or is cut short
- Key parts of the answer are missing

### 5. REFUSAL
The agent declined to complete the task.
- Agent says it cannot do something it could do
- Agent refuses without valid reason
- Agent gives up prematurely

### 6. OTHER
Any other error not fitting the above categories.

## Evaluation Instructions

1. Read through the trajectory carefully
2. Check if the agent completed all required actions
3. Look for any of the error modes listed above
4. If score=0, you MUST provide:
   - error_type: One of the categories above
   - error_citation: Quote the EXACT text from the trajectory (with message number [XXX])
   - error_explanation: Explain why this is wrong and what should have been correct

## Required JSON Response

Respond with a JSON object (reasoning FIRST, score AFTER). Put the reasoning BEFORE you decide the score.

```json
{{
  "reasoning": "Brief explanation of your judgment",
  "score": 0,
  "task_analysis": "What the agent was supposed to do",
  "completion_evidence": "Evidence of completion or failure", 
  "errors_found": ["list of errors, or empty"],
  "error_type": "hallucination|underspecified_tool_semantics|incorrect_tool_output|incomplete_response|refusal|other|none",
  "error_citation": "[003] ASSISTANT: The meeting was on December 18... (quote the problematic text with message number)",
  "error_explanation": "The tool output at [002] shows the meeting was on January 10, not December 18. The agent hallucinated the date."
}}
```

CRITICAL:
- reasoning MUST come before score
- score MUST be 0 or 1 (integer)
- score = 1 ONLY if task was clearly completed successfully with no errors
- score = 0 if incomplete, had errors, hallucinations, or failed
- If score = 0, error_type MUST NOT be "none"
- If score = 0, error_citation MUST contain exact quoted text with message number
- If score = 1, error_type = "none" and error_citation/error_explanation are empty strings
"""


# -----------------------------------------------------------------------------
# Single Evaluation
# -----------------------------------------------------------------------------

async def evaluate_single_iteration(
    trajectory: Dict[str, Any],
    iteration: int,
    model: str = "gpt-5.2",
    temperature: float = 0.7,
    max_retries: int = 2,
) -> IterationResult:
    """Evaluate a trajectory once with the LLM."""
    async with LLM_SEMAPHORE:
        start = time.time()
        result = IterationResult()
        
        trajectory_text = format_trajectory_as_text(trajectory)
        
        # Truncate if too long (GPT-5.2 has 400k context but be reasonable)
        max_len = 100000
        if len(trajectory_text) > max_len:
            truncate_msg = f"\n\n[TRUNCATED from {len(trajectory_text)} chars]\n\n"
            head = max_len // 2 - len(truncate_msg) // 2
            tail = max_len - head - len(truncate_msg)
            trajectory_text = trajectory_text[:head] + truncate_msg + trajectory_text[-tail:]
        
        prompt = build_judge_prompt(trajectory, trajectory_text)
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    await asyncio.sleep(2 ** attempt)
                
                client = openai.AsyncOpenAI(timeout=300.0)
                payload: Dict[str, Any] = {
                    "model": model,
                    "max_completion_tokens": 2000,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": JUDGE_RESPONSE_FORMAT,
                }
                
                try:
                    response = await client.chat.completions.create(**payload)
                except Exception as e:
                    msg = str(e)
                    # Handle unsupported response_format
                    if "response_format" in msg or "json_schema" in msg:
                        payload.pop("response_format", None)
                        response = await client.chat.completions.create(**payload)
                    else:
                        raise
                
                response_text = response.choices[0].message.content if response.choices else ""
                result.raw_response = response_text
                result.time_s = time.time() - start
                
                # Parse score and error details
                try:
                    obj = json.loads(response_text) if response_text else None
                    if isinstance(obj, dict) and "score" in obj:
                        result.score = int(obj.get("score"))
                        result.reasoning = obj.get("reasoning", "")
                        # Extract error details
                        result.error_type = obj.get("error_type", "none")
                        result.error_citation = obj.get("error_citation", "")
                        result.error_explanation = obj.get("error_explanation", "")
                except Exception:
                    result.score = extract_score_from_response(response_text)
                
                return result
                
            except asyncio.TimeoutError:
                result.error = f"Timeout (attempt {attempt + 1}/{max_retries + 1})"
            except Exception as e:
                result.error = str(e)
                result.time_s = time.time() - start
                return result
        
        result.time_s = time.time() - start
        return result


async def evaluate_trajectory_with_majority_vote(
    trajectory: Dict[str, Any],
    num_iterations: int = 5,
    model: str = "gpt-5.2",
    temperature: float = 0.7,
) -> TrajectoryResult:
    """Evaluate a trajectory multiple times and take majority vote."""
    result = TrajectoryResult(trajectory_id=trajectory["id"])
    
    # Run all iterations
    tasks = [
        evaluate_single_iteration(trajectory, i, model, temperature)
        for i in range(num_iterations)
    ]
    
    iterations = await asyncio.gather(*tasks)
    result.iterations = list(iterations)
    result.total_time_s = sum(it.time_s for it in iterations)
    
    # Calculate majority vote
    valid_scores = [it.score for it in iterations if it.score is not None]
    if valid_scores:
        counter = Counter(valid_scores)
        result.final_score, count = counter.most_common(1)[0]
        result.confidence = count / len(valid_scores)
    
    return result


# -----------------------------------------------------------------------------
# Batch Evaluation
# -----------------------------------------------------------------------------

async def batch_evaluate(
    trajectories: List[Dict[str, Any]],
    num_iterations: int = 5,
    model: str = "gpt-5.2",
    temperature: float = 0.7,
    concurrency: int = 5,
    verbose: bool = False,
) -> List[TrajectoryResult]:
    """Evaluate all trajectories with majority voting."""
    global LLM_SEMAPHORE
    LLM_SEMAPHORE = asyncio.Semaphore(concurrency)
    
    total = len(trajectories)
    results: List[TrajectoryResult] = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating {total} trajectories with {model}")
    print(f"  - {num_iterations} iterations per trajectory")
    print(f"  - Temperature: {temperature}")
    print(f"  - Concurrency: {concurrency}")
    print(f"{'='*60}\n")
    
    for i, trajectory in enumerate(trajectories):
        tid = trajectory["id"]
        print(f"[{i+1}/{total}] Evaluating {tid}...", end=" ", flush=True)
        
        result = await evaluate_trajectory_with_majority_vote(
            trajectory, num_iterations, model, temperature
        )
        results.append(result)
        
        # Print summary
        scores = [it.score for it in result.iterations if it.score is not None]
        score_str = ",".join(str(s) for s in scores)
        final = result.final_score if result.final_score is not None else "?"
        conf = f"{result.confidence:.0%}" if result.confidence else "N/A"
        
        print(f"Votes=[{score_str}] -> {final} (conf: {conf})")
        
        if verbose:
            for j, it in enumerate(result.iterations):
                status = "✓" if it.score is not None else "✗"
                print(f"    Iter {j+1}: score={it.score} {status}")
                if it.error:
                    print(f"        Error: {it.error}")
    
    return results


# -----------------------------------------------------------------------------
# Save Results
# -----------------------------------------------------------------------------

def save_predictions(results: List[TrajectoryResult], output_path: Path):
    """Save predictions to JSON file."""
    predictions = {}
    
    for result in results:
        scores = [it.score for it in result.iterations if it.score is not None]
        reasonings = [it.reasoning for it in result.iterations if it.reasoning]
        
        # Collect error details from iterations that scored 0
        error_details = []
        for it in result.iterations:
            if it.score == 0 and it.error_type and it.error_type != "none":
                error_details.append({
                    "error_type": it.error_type,
                    "error_citation": it.error_citation,
                    "error_explanation": it.error_explanation,
                })
        
        # Get the most common error type if score is 0
        primary_error = None
        if result.final_score == 0 and error_details:
            error_types = [e["error_type"] for e in error_details if e["error_type"]]
            if error_types:
                primary_error = Counter(error_types).most_common(1)[0][0]
        
        predictions[result.trajectory_id] = {
            "final_score": result.final_score,
            "confidence": result.confidence,
            "individual_scores": scores,
            "reasonings": reasonings[:3],  # Keep first 3 reasonings
            "num_valid_iterations": len(scores),
            "total_time_s": result.total_time_s,
            # Error details for erroneous trajectories
            "primary_error_type": primary_error,
            "error_details": error_details[:3] if error_details else [],  # Keep first 3 error details
        }
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": config.LLM_MODEL,
        "iterations_per_trajectory": config.LLM_JUDGE_ITERATIONS,
        "temperature": config.LLM_TEMPERATURE,
        "total_trajectories": len(results),
        "predictions": predictions,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Predictions saved to {output_path}")


def print_summary(results: List[TrajectoryResult]):
    """Print summary statistics."""
    total = len(results)
    successful = sum(1 for r in results if r.final_score == 1)
    erroneous = sum(1 for r in results if r.final_score == 0)
    unknown = sum(1 for r in results if r.final_score is None)
    
    avg_confidence = sum(r.confidence for r in results) / total if total else 0
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total trajectories: {total}")
    print(f"  - Successful (1): {successful} ({successful/total*100:.1f}%)")
    print(f"  - Erroneous (0):  {erroneous} ({erroneous/total*100:.1f}%)")
    print(f"  - Unknown:        {unknown}")
    print(f"Average confidence: {avg_confidence:.1%}")
    print(f"{'='*60}\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY=your-key-here")
        return 1
    
    # Load trajectories
    trajectories = load_trajectories(limit=args.limit)
    if not trajectories:
        print("No trajectories to evaluate")
        return 1
    
    # Run evaluation
    results = await batch_evaluate(
        trajectories,
        num_iterations=args.iterations,
        model=args.model,
        temperature=args.temperature,
        concurrency=args.concurrency,
        verbose=args.verbose,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(config.DATA_DIR) / config.LLM_PREDICTIONS_FILE
    save_predictions(results, output_path)
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate agent trajectories using LLM-as-Judge with majority voting"
    )
    
    parser.add_argument("--limit", type=int, default=None,
        help="Maximum number of trajectories to evaluate")
    parser.add_argument("--model", type=str, default=config.LLM_MODEL,
        help=f"OpenAI model (default: {config.LLM_MODEL})")
    parser.add_argument("--iterations", type=int, default=config.LLM_JUDGE_ITERATIONS,
        help=f"Number of iterations per trajectory (default: {config.LLM_JUDGE_ITERATIONS})")
    parser.add_argument("--temperature", type=float, default=config.LLM_TEMPERATURE,
        help=f"Temperature for LLM calls (default: {config.LLM_TEMPERATURE})")
    parser.add_argument("--concurrency", type=int, default=5,
        help="Number of concurrent API calls (default: 5)")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Show detailed progress")
    
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
