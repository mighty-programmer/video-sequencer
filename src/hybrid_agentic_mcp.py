"""Minimal stdio MCP proxy for the hybrid agentic retrieval bridge.

The VideoPrism model and FAISS index stay in the parent benchmark process.
This process translates Codex MCP calls into token-authenticated localhost
requests so the agent never receives benchmark paths or ground truth.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_global_script_context",
        "description": "Return the full ordered script using segment indices and text only.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "get_candidate_catalog",
        "description": "List every anonymous candidate ID in the media library without filenames, timing, or metadata. Available only in full-sequence audit mode.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "get_segment_context",
        "description": "Return one allowed ambiguous segment and its neighboring script segments.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "minimum": 0},
                "radius": {"type": "integer", "minimum": 0, "maximum": 4, "default": 2},
            },
            "required": ["segment_index"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "get_candidate_scores",
        "description": "Return anonymous candidates ranked in the configured baseline-matrix score space.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "minimum": 0},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
            },
            "required": ["segment_index"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "get_initial_assignment",
        "description": "Return the current no-reuse Hungarian assignment with anonymous clip IDs.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "search_video_index",
        "description": "Search VideoPrism/FAISS with the exact custom query, without a baseline prompt template; scores are raw cosine similarities.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "minimum": 0},
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "exclude_assigned": {"type": "boolean", "default": False},
            },
            "required": ["segment_index", "query"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "inspect_clip",
        "description": "Inspect a sampled-frame contact sheet for an anonymous candidate without exposing filenames or duration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "minimum": 0},
                "candidate_index": {"type": "integer", "minimum": 0},
            },
            "required": ["segment_index", "candidate_index"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "inspect_candidate",
        "description": "Inspect a sampled-frame contact sheet for any anonymous library candidate during a full-sequence audit, independent of retrieval rank.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "candidate_index": {"type": "integer", "minimum": 0},
            },
            "required": ["candidate_index"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
]


def _write(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _tool_call(url: str, token: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps({"name": name, "arguments": arguments}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "Hybrid tool request failed")

    result = dict(payload.get("result") or {})
    image_path = result.pop("_contact_sheet_path", None)
    content: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps(result, separators=(",", ":"))}]
    if image_path:
        path = Path(image_path)
        if path.is_file():
            content.append({
                "type": "image",
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                "mimeType": "image/jpeg",
            })
    return {"content": content, "structuredContent": result, "isError": False}


def _serve(url: str, token: str) -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            method = message.get("method")
            request_id = message.get("id")
            if method == "initialize":
                requested = (message.get("params") or {}).get("protocolVersion") or "2024-11-05"
                result = {
                    "protocolVersion": requested,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "video-sequencer-hybrid", "version": "1.1.0"},
                    "instructions": (
                        "Use only these anonymous read-only retrieval tools. Never infer or request filenames, "
                        "ground truth, timestamps, audio, captions, or benchmark metadata. Obey per-segment budgets."
                    ),
                }
            elif method in {"notifications/initialized", "notifications/cancelled"}:
                continue
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = message.get("params") or {}
                result = _tool_call(url, token, str(params.get("name") or ""), params.get("arguments") or {})
            elif method == "resources/list":
                result = {"resources": []}
            elif method == "prompts/list":
                result = {"prompts": []}
            elif method == "logging/setLevel":
                result = {}
            else:
                if request_id is None:
                    continue
                _write({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}})
                continue
            if request_id is not None:
                _write({"jsonrpc": "2.0", "id": request_id, "result": result})
        except (ValueError, RuntimeError, OSError, urllib.error.URLError) as exc:
            request_id = locals().get("request_id")
            if request_id is not None:
                _write({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}})


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid agentic stdio MCP proxy")
    parser.add_argument("--url", default=os.environ.get("HYBRID_AGENT_BRIDGE_URL"))
    parser.add_argument("--token", default=os.environ.get("HYBRID_AGENT_BRIDGE_TOKEN"))
    args = parser.parse_args()
    if not args.url or not args.token:
        parser.error("bridge URL and token must be provided through arguments or HYBRID_AGENT_BRIDGE_* variables")
    _serve(args.url, args.token)


if __name__ == "__main__":
    main()
