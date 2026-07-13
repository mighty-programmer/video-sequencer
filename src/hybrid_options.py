"""Shared option contract for the hybrid agentic pipeline interfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_CODEX_REASONING_EFFORT = "xhigh"

CODEX_REASONING_EFFORTS = [
    {"value": "low", "label": "Low"},
    {"value": "medium", "label": "Medium"},
    {"value": "high", "label": "High"},
    {"value": "xhigh", "label": "Extra high"},
    {"value": "max", "label": "Maximum"},
    {"value": "ultra", "label": "Ultra (automatic delegation)"},
]

FALLBACK_CODEX_MODELS = [
    {
        "value": "gpt-5.6-sol",
        "label": "GPT-5.6-Sol",
        "default_reasoning_effort": "low",
        "supported_reasoning_efforts": ["low", "medium", "high", "xhigh", "max", "ultra"],
    },
    {
        "value": "gpt-5.6-terra",
        "label": "GPT-5.6-Terra",
        "default_reasoning_effort": "medium",
        "supported_reasoning_efforts": ["low", "medium", "high", "xhigh", "max", "ultra"],
    },
    {
        "value": "gpt-5.6-luna",
        "label": "GPT-5.6-Luna",
        "default_reasoning_effort": "medium",
        "supported_reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
    },
    {
        "value": "gpt-5.5",
        "label": "GPT-5.5",
        "default_reasoning_effort": "medium",
        "supported_reasoning_efforts": ["low", "medium", "high", "xhigh"],
    },
    {
        "value": "gpt-5.4",
        "label": "GPT-5.4",
        "default_reasoning_effort": "medium",
        "supported_reasoning_efforts": ["low", "medium", "high", "xhigh"],
    },
    {
        "value": "gpt-5.4-mini",
        "label": "GPT-5.4-Mini",
        "default_reasoning_effort": "medium",
        "supported_reasoning_efforts": ["low", "medium", "high", "xhigh"],
    },
    {
        "value": "gpt-5.3-codex-spark",
        "label": "GPT-5.3-Codex-Spark",
        "default_reasoning_effort": "high",
        "supported_reasoning_efforts": ["low", "medium", "high", "xhigh"],
    },
]


def load_codex_model_catalog(cache_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return visible models from Codex's live cache, with a stable fallback."""
    path = cache_path or (Path.home() / ".codex" / "models_cache.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        catalog = []
        for model in payload.get("models", []):
            if model.get("visibility") == "hide" or not model.get("slug"):
                continue
            efforts = [
                str(item.get("effort"))
                for item in model.get("supported_reasoning_levels", [])
                if item.get("effort")
            ]
            catalog.append({
                "value": str(model["slug"]),
                "label": str(model.get("display_name") or model["slug"]),
                "default_reasoning_effort": str(model.get("default_reasoning_level") or "medium"),
                "supported_reasoning_efforts": efforts or ["low", "medium", "high", "xhigh"],
            })
        if catalog:
            return catalog
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return [dict(model) for model in FALLBACK_CODEX_MODELS]


def codex_reasoning_options(model: str, catalog: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    """Return only the reasoning efforts advertised by a selected Codex model."""
    models = catalog or load_codex_model_catalog()
    supported = next(
        (item.get("supported_reasoning_efforts", []) for item in models if item.get("value") == model),
        [],
    )
    labels = {item["value"]: item["label"] for item in CODEX_REASONING_EFFORTS}
    return [{"value": effort, "label": labels.get(effort, effort)} for effort in supported]

HYBRID_OPTIONS = {
    "models": [
        {"value": "videoprism_lvt_public_v1_base", "label": "VideoPrism Base"},
        {"value": "videoprism_lvt_public_v1_large", "label": "VideoPrism Large"},
    ],
    "frames": [
        {"value": "8", "label": "8 frames"},
        {"value": "16", "label": "16 frames"},
    ],
    "resolutions": [
        {"value": "288", "label": "288p"},
        {"value": "396", "label": "396p"},
    ],
    "dual_softmax": [
        {"value": "false", "label": "No"},
        {"value": "true", "label": "Yes"},
    ],
    "prompt_modes": [
        {"value": "none", "label": "None"},
        {"value": "template:video", "label": "Template: video"},
        {"value": "template:photo", "label": "Template: photo"},
        {"value": "template:scene", "label": "Template: scene"},
        {"value": "template:cooking", "label": "Template: cooking"},
        {"value": "template:clip", "label": "Template: short clip"},
        {"value": "ensemble:template", "label": "Template ensemble"},
    ],
    "decision_modes": [
        {"value": "advisory", "label": "Advisory: soft boosts and penalties"},
        {"value": "hard_lock", "label": "Hard constraints: inspected pairs at 90%+ confidence"},
    ],
    "review_scopes": [
        {"value": "ambiguous", "label": "Uncertainty-focused review"},
        {"value": "all", "label": "Full-sequence visual audit"},
    ],
    "shortlist_sizes": [
        {"value": "3", "label": "3 candidates"},
        {"value": "5", "label": "5 candidates"},
        {"value": "10", "label": "10 candidates"},
    ],
    "ambiguity_margins": [
        {"value": "0.02", "label": "0.02"},
        {"value": "0.05", "label": "0.05"},
        {"value": "0.10", "label": "0.10"},
    ],
    "reviewed_segments": [
        {"value": "3", "label": "3 segments"},
        {"value": "5", "label": "5 segments"},
        {"value": "10", "label": "10 segments"},
    ],
    "search_budgets": [
        {"value": "2", "label": "2 custom searches"},
        {"value": "4", "label": "4 custom searches"},
    ],
    "codex_models": [
        {"value": model["value"], "label": model["label"]}
        for model in FALLBACK_CODEX_MODELS
    ],
    "codex_reasoning_efforts": CODEX_REASONING_EFFORTS,
}

HYBRID_DEFAULT_SELECTIONS = {
    "models": ["videoprism_lvt_public_v1_large"],
    "frames": ["8", "16"],
    "resolutions": ["288", "396"],
    "dual_softmax": ["false", "true"],
    "prompt_modes": ["none", "template:photo", "template:scene"],
    "decision_modes": ["advisory"],
    "review_scopes": ["all"],
    "shortlist_sizes": ["3", "5"],
    "ambiguity_margins": ["0.02", "0.05"],
    "reviewed_segments": ["5", "10"],
    "search_budgets": ["2"],
    "codex_model": [DEFAULT_CODEX_MODEL],
    "codex_reasoning_effort": [DEFAULT_CODEX_REASONING_EFFORT],
}

HYBRID_PROMPT_MODES = tuple(option["value"] for option in HYBRID_OPTIONS["prompt_modes"])
