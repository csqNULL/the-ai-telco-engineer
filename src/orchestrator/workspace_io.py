# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Workspace filesystem operations: reading code and metrics."""

import json
from pathlib import Path

import printer
from agent import RESULT_FILE, SOLUTION_FILE
from leaderboard import HPResult
from workspace_fs import read_text_under_root


def read_workspace_code(workspace_root: Path, workspace_id: str) -> str:
    """Read the solution code from a workspace.

    Args:
        workspace_root: Root path containing all workspace directories.
        workspace_id: The workspace ID to read from.

    Returns:
        The code as a string, or empty string if not found.
    """
    workspace_path = workspace_root / workspace_id
    try:
        return read_text_under_root(workspace_path, SOLUTION_FILE)
    except FileNotFoundError:
        printer.log(
            f"Warning: {SOLUTION_FILE} not found in {workspace_id}"
        )
        return ""
    except OSError as e:
        printer.log(f"Warning: Failed to read code from {workspace_id}: {e}")
        return ""


def read_results(
    workspace_root: Path, workspace_id: str,
) -> tuple[bool, list[HPResult]]:
    """Read evaluation results from a workspace's ``result.json`` file.

    Expected format::

        {"success": bool,
         "results": [{"hp_index": int, "metric": float, "complexity": float}, ...]}

    The matching hyperparameter values are pulled from ``pareto_params.json``
    in the same workspace and attached to each :class:`HPResult`.

    Args:
        workspace_root: Root path containing all workspace directories.
        workspace_id: The workspace ID to read from.

    Returns:
        ``(success, hp_results)``.  On failure returns ``(False, [])``.
    """
    workspace_path = workspace_root / workspace_id
    try:
        raw = read_text_under_root(workspace_path, RESULT_FILE)
        data = json.loads(raw)
        if not isinstance(data, dict) or not data.get("success", False):
            return False, []
        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            return False, []
        params_by_idx = read_pareto_params(
            workspace_root, workspace_id,
        )
        hp_results = []
        for entry in raw_results:
            if not isinstance(entry, dict):
                return False, []
            hp = HPResult.from_dict(entry)
            hp.params = dict(params_by_idx.get(hp.hp_index, {}))
            hp_results.append(hp)
        return bool(hp_results), hp_results
    except FileNotFoundError:
        return False, []
    except Exception as e:
        printer.log(
            f"Warning: Failed to read {RESULT_FILE} for {workspace_id}: {e}"
        )
        return False, []


def read_pareto_params(
    workspace_root: Path, workspace_id: str,
) -> dict[int, dict]:
    """Read ``pareto_params.json`` from a workspace.

    Returns a mapping from HP index to the parameter dict.
    Returns an empty dict if the file is missing or unreadable.
    """
    workspace_path = workspace_root / workspace_id
    try:
        raw = read_text_under_root(workspace_path, "pareto_params.json")
        data = json.loads(raw)
        return {int(k): v for k, v in data.items()}
    except FileNotFoundError:
        return {}
    except Exception as e:
        printer.log(
            f"Warning: Failed to read pareto_params.json for "
            f"{workspace_id}: {e}"
        )
        return {}
