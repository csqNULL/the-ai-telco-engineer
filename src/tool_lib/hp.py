# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hyperparameter helper for agent code and search-space extraction.

The :class:`HP` class is copied into every workspace so that agent code
can define tunable parameters with ``from hp import HP``.  During the
agent run, ``HP.get()`` simply returns the coded default.  During
post-process hyperparameter tuning, the framework writes candidate
values to ``hyperparams.json`` before each trial evaluation —
``HP.get()`` then returns those values instead of the defaults.  The
``low``/``high``/``choices``/``log`` keyword arguments are metadata
only; the framework's AST parser reads them to build the search space.

:func:`extract_search_space` parses a Python source file's AST and
returns the Optuna-compatible search space implied by ``HP.get()`` calls.
"""

import ast
import json
from pathlib import Path
from typing import Any


class HP:
    """Hyperparameter store backed by ``hyperparams.json``.

    Usage inside agent code::

        from hp import HP

        threshold = HP.get("threshold", 0.05, low=0.01, high=0.1)
        mode = HP.get("mode", "A", choices=["A", "B", "C"])
        lr = HP.get("lr", 1e-4, low=1e-6, high=1e-2, log=True)
    """

    _params: dict[str, Any] = {}
    _loaded = False

    @classmethod
    def _load(cls) -> None:
        if cls._loaded:
            return
        try:
            with open("hyperparams.json") as f:
                cls._params = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cls._params = {}
        cls._loaded = True

    @classmethod
    def get(cls, name: str, default: Any = None, *,
            low: Any = None, high: Any = None,
            choices: Any = None, log: bool = False) -> Any:
        """Return the value for *name*, falling back to *default*.

        The keyword arguments ``low``, ``high``, ``choices``, and ``log``
        are **metadata only** — they are not used at runtime.  The
        framework's AST parser reads them to build the search space for
        automatic hyperparameter tuning.
        """
        cls._load()
        return cls._params.get(name, default)


def extract_search_space(source: str) -> dict[str, list]:
    """Parse ``HP.get()`` calls from Python *source* and return a search space.

    Returns a dict mapping parameter names to Optuna-style specs::

        {"threshold": ["float", 0.01, 0.1],
         "mode":      ["categorical", ["A", "B", "C"]],
         "lr":        ["log_float", 1e-6, 1e-2]}

    Only ``HP.get()`` calls that include ``low``/``high`` or ``choices``
    are included — calls with no range metadata are treated as
    non-tunable constants.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    space: dict[str, list] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_hp_get(node):
            continue

        name = _get_param_name(node)
        if name is None:
            continue

        default = _get_default(node)
        kwargs = _get_kwargs(node)

        choices = kwargs.get("choices")
        low = kwargs.get("low")
        high = kwargs.get("high")
        log = kwargs.get("log", False)

        if choices is not None:
            space[name] = ["categorical", choices]
        elif low is not None and high is not None:
            if log:
                space[name] = ["log_float", low, high]
            elif isinstance(default, int) and isinstance(low, int) and isinstance(high, int):
                space[name] = ["int", low, high]
            else:
                space[name] = ["float", low, high]

    return space


def _is_hp_get(node: ast.Call) -> bool:
    """Return True if *node* is a call to ``HP.get(...)``."""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "get"
        and isinstance(func.value, ast.Name)
        and func.value.id == "HP"
    )


def _get_param_name(node: ast.Call) -> str | None:
    """Extract the first positional argument (parameter name) as a string."""
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    return None


def _get_default(node: ast.Call) -> Any:
    """Extract the second positional argument (default value)."""
    if len(node.args) >= 2:
        return _eval_constant(node.args[1])
    for kw in node.keywords:
        if kw.arg == "default":
            return _eval_constant(kw.value)
    return None


def _get_kwargs(node: ast.Call) -> dict[str, Any]:
    """Extract keyword arguments (low, high, choices, log)."""
    result: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg in ("low", "high", "log"):
            result[kw.arg] = _eval_constant(kw.value)
        elif kw.arg == "choices":
            result["choices"] = _eval_constant(kw.value)
    return result


def bake_hyperparams(source: str, params: dict[str, Any]) -> str:
    """Rewrite ``HP.get()`` defaults in *source* to match *params*.

    For every ``HP.get("name", old_default, ...)`` call whose *name*
    appears in *params*, the *old_default* is replaced with the
    corresponding value from *params*.  All other parts of the call
    (``low``, ``high``, ``choices``, ``log``) are left untouched so that
    future HP tuning can still discover the search space.

    Returns the modified source string.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # (lineno, col_offset, end_col_offset, new_value_str)
    edits: list[tuple[int, int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_hp_get(node):
            continue
        name = _get_param_name(node)
        if name is None or name not in params:
            continue
        default_node = _get_default_node(node)
        if default_node is None:
            continue
        if default_node.lineno != default_node.end_lineno:
            continue
        edits.append((
            default_node.lineno,
            default_node.col_offset,
            default_node.end_col_offset,
            _value_to_source(params[name]),
        ))

    if not edits:
        return source

    # Apply right-to-left within each line to preserve column offsets.
    edits.sort(key=lambda e: (e[0], -e[1]))

    lines = source.splitlines(keepends=True)
    for lineno, col_start, col_end, new_val in edits:
        line = lines[lineno - 1]
        lines[lineno - 1] = line[:col_start] + new_val + line[col_end:]

    return "".join(lines)


def _get_default_node(node: ast.Call) -> ast.expr | None:
    """Return the AST node for the default (second positional arg)."""
    if len(node.args) >= 2:
        return node.args[1]
    for kw in node.keywords:
        if kw.arg == "default":
            return kw.value
    return None


def _value_to_source(value: Any) -> str:
    """Convert a Python value to a source-code literal string."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return "None"
    return repr(value)


def _eval_constant(node: ast.expr) -> Any:
    """Safely evaluate a constant AST node (literals only)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _eval_constant(node.operand)
        if isinstance(val, (int, float)):
            return -val
    if isinstance(node, ast.List):
        return [_eval_constant(elt) for elt in node.elts]
    return None
