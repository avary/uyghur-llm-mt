"""
checkpoint.py - Crash-safe, resumable JSONL result storage.

Every completed sentence is appended as one JSON line and flushed immediately.
On restart, `load_completed` reloads everything already on disk (tolerating a
torn final line from a hard kill) so the driver only generates the remainder.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def new_record(
    run_id: int,
    model: str,
    condition: str,
    sentence: str,
    reference: str,
    prompt: str,
    thinking: bool = False,
) -> Dict[str, Any]:
    return {
        "id": run_id,
        "model": model,
        "condition": condition,
        "sentence": sentence,
        "reference": reference,
        "prompt": prompt,
        "thinking": thinking,
        "translation": None,
        "reasoning": None,
        "answered": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "finish_reason": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def finalize_record(record: Dict[str, Any], gen: Dict[str, Any]) -> Dict[str, Any]:
    record["translation"] = gen["text"]
    record["reasoning"] = gen.get("reasoning")
    record["answered"] = gen.get("answered", True)
    record["prompt_tokens"] = gen["prompt_tokens"]
    record["completion_tokens"] = gen["completion_tokens"]
    record["finish_reason"] = gen["finish_reason"]
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    return record


class JSONLCheckpointWriter:
    """Append-only writer with per-record flush (survives SIGKILL)."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]):
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()


def load_completed(path: str) -> Dict[int, Dict[str, Any]]:
    """Load {id: record} for every fully-generated record in `path`.

    Tolerates a truncated/torn last line (killed mid-write). Records with
    `translation is None` are treated as incomplete and skipped.
    """
    completed: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn line -> will be regenerated
            if rec.get("translation") is None:
                continue
            completed[int(rec["id"])] = rec
    return completed


def pending_items(
    items: List[Dict[str, Any]],
    completed: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter an item list ({id, ...}) down to what still needs generation."""
    return [it for it in items if it["id"] not in completed]


def run_status(path: str, total: int) -> str:
    """Human-readable progress line for logging."""
    done = len(load_completed(path))
    return f"{os.path.basename(path)}: {done}/{total} complete"


def is_run_complete(path: str, total: int) -> bool:
    return os.path.exists(path) and len(load_completed(path)) >= total
