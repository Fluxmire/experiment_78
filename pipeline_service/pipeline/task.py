from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """One multigen candidate produced by the coder in iteration 0."""

    k: int
    seed: int
    source: str = "coder"
    js_code: str | None = None
    js_valid: bool | None = None
    js_errors: list[str] = field(default_factory=list)
    scene_json: dict | None = None
    rendered_png: bytes | None = None
    render_errors: list[str] = field(default_factory=list)
    judge_white_views: dict[str, bytes] = field(default_factory=dict)
    judge_gray_views: dict[str, bytes] = field(default_factory=dict)
    judge_embeddings: bytes | None = None
    elapsed_s: float = 0.0
    drop_reason: str | None = None


@dataclass
class PipelineTask:
    """Single envelope threaded through every pipeline stage."""

    stem: str
    image_url: str
    seed: int = 42

    image_bytes: bytes | None = None
    image_mime: str = "image/jpeg"

    osd: str | None = None                 
    js_code: str | None = None

    js_valid: bool | None = None
    js_errors: list[str] = field(default_factory=list)
    js_metrics: dict | None = None
    scene_json: dict | None = None
    js_stages_run: list[str] = field(default_factory=list)
    js_module_load_ms: float | None = None
    js_execution_ms: float | None = None
    js_total_ms: float | None = None

    multigen_pngs: list[bytes] = field(default_factory=list) 
    rendered_png: bytes | None = None              
    render_ms: float | None = None
    render_errors: list[str] = field(default_factory=list)
    refinement_rendered_pngs: list[bytes] = field(default_factory=list)  

    iteration: int = 0
    score_history: list[float] = field(default_factory=list)
    best_score: float = -1.0
    best_iter: int = -1
    best_js_code: str | None = None
    best_rendered_png: bytes | None = None

    candidates: list[Candidate] = field(default_factory=list)
    winner_k: int | None = None

    started_at: float = 0.0
    terminal: bool = False
    deadline_task: asyncio.Task | None = field(default=None, repr=False, compare=False)

    failed: bool = False
    failure_reason: str | None = None
    failure_stage: str | None = None
    attempt: int = 0

    meta: dict[str, Any] = field(default_factory=dict)

    oom_retries: int = 0

