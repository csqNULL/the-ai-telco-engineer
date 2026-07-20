# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared data types for the orchestrator."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    """A task with a workspace ID, base query, and assigned approach section."""
    workspace_id: str
    query: str
    assigned_approach_section: str = ""
    prompt_template: str = ""
    timeout: Optional[int] = None
    generation: int = 0


@dataclass
class TaskResult:
    """Result from a completed task."""
    workspace_id: str
    query: str
    response: str
    success: bool
    error: Optional[str] = None
    generation: int = 0


@dataclass
class Idea:
    """An algorithmic approach with optional reference workspaces."""
    cluster_id: int
    description: str
    reference_workspace_ids: list[str] = field(default_factory=list)


@dataclass
class GenerationSummary:
    """Summary of a single candidate's result within a generation."""
    generation: int
    cluster_id: int
    idea_description: str
    summary: str
    metric: float
    complexity: float
    workspace_id: str
    hp_index: int = 0
    info: Optional[str] = None


@dataclass
class JournalAnalysis:
    """Behavioral analysis of a single agent run from its journal."""
    workspace_id: str
    behavioral_summary: str
    num_tool_calls: int = 0
    num_eval_attempts: int = 0
    num_eval_successes: int = 0
    metric_trajectory: list[float] = field(default_factory=list)
    timed_out: bool = False
