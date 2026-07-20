# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tool provider for evaluating candidate OTFS DD-domain detectors.

Wraps the OTFS coded-link evaluation pipeline as a LangChain tool so an
agent can submit its ``draft.py`` and receive a Normalised Validation
Error (NVE) report against a precomputed EP baseline. The agent's file
must define exactly one subclass of
:class:`otfs.equalizers.base.Equalizer`.

The workspace is pre-seeded with a pared-down copy of the ``otfs``
toolkit (channel + IO + grid + equalizer base class) so the agent can
``from otfs.equalizers.base import Equalizer`` without having to ship
its own copy. The existing detector implementations (EP, MP, LMMSE) are
intentionally **not** staged into the workspace.
"""

from __future__ import annotations

from pathlib import Path

from tool_lib.base import EvalToolBase
from langchain_core.tools import tool, BaseTool


_THIS_DIR = Path(__file__).parent
_EVAL_SCRIPT_PATH = _THIS_DIR / "eval/eval.py"
_LINK_CONFIG_PATH = _THIS_DIR / "eval/link_config.py"
_BASELINE_BLER_PATH = _THIS_DIR / "eval/baseline_bler.pkl"
_OTFS_PACKAGE_ROOT = _THIS_DIR / "eval/otfs_workspace"


def _iter_package_files(root: Path):
    """Yield ``(absolute_path, relative_path_in_workspace)`` for every file
    under ``root``, preserving its sub-folder structure.

    Skips Python build artefacts (``__pycache__/*`` and ``*.pyc``); those
    are not part of the task definition, vary by interpreter version, and
    would be opened in text mode by the staging code below.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part == "__pycache__" for part in rel.parts):
            continue
        if path.suffix.lower() == ".pyc":
            continue
        yield path, rel


def _is_binary_suffix(path: Path) -> bool:
    return path.suffix.lower() in {".pkl", ".pt", ".pth", ".npy", ".npz", ".bin"}


class EvalTool(EvalToolBase):
    """Tool provider for evaluating OTFS DD-domain detector implementations.

    Provides a tool that evaluates an agent-supplied detector for a
    coded OTFS uplink driven by the analytical rectangular-CP DD IO
    model. The metric is the Normalised Validation Error (NVE) — the
    mean ratio between the agent's BLER and the EP baseline BLER over
    the evaluation SNR points (closer to 1 / lower is better).
    The complexity is the average total runtime of ``detector.llr``
    alone for one batch of frames, in milliseconds (lower is better).

    Output format (first line only is parsed by the framework):
    ``SUCCESS, <metric>, <complexity>`` or ``FAILURE,`` followed by a
    newline and optional human-readable text.

    The agent must create a file called ``draft.py`` in the workspace
    that defines exactly one subclass of ``otfs.equalizers.base.Equalizer``.
    """

    _TOOL_DESCRIPTION = """\
Evaluate the OTFS DD-domain detector implementation.

Expects `draft.py` in the workspace defining exactly one subclass of
`otfs.equalizers.base.Equalizer` with `channel_form in {"sparse",
"dense"}`, an optional class-level `top_k`, and an `llr(y_dd, channel,
no)` method. See the task prompt for the full signature, scenario
constants, and constraints.
The evaluation harness (eval.py, link_config.py, baseline_bler.pkl, and
the otfs/ package) is injected at evaluation time.

Returns:
    str: First line ``SUCCESS, <metric>, <complexity>`` or ``FAILURE,``.
    <metric> is the Normalised Validation Error vs. the EP baseline
        (closer to 1 / lower is better).
    <complexity> is the average total detector.llr runtime for one
        batch of frames, in milliseconds (lower is better).
    On success, additional `[compile]` / `[compile warnings]` lines report
        whether `llr` compiled as a single graph, hit graph breaks, or fell
        back to eager (with any PyTorch compile warnings), so you can make
        `llr` compile-friendly and lower its complexity.
    Remaining lines are optional human-readable text."""

    def __init__(self, eval_timeout: int, **kwargs):
        super().__init__(eval_timeout, **kwargs)

    def set_workspace(self, workspace) -> None:
        """Bind to a workspace and stage the permanent ``otfs/`` package.

        ``otfs/`` is part of the *task definition*: the agent must be
        able to import it (``from otfs.equalizers.base import
        Equalizer``) and inspect it from the very first turn, so we
        seed it once on workspace creation and leave it in place.
        """
        super().set_workspace(workspace)
        self._stage_otfs_package()

    def _stage_otfs_package(self) -> None:
        if self._workspace is None:
            raise ValueError("Workspace not set")
        # Pre-create the package directories so ``_write_file`` calls
        # don't race on the parent path.
        created_dirs: set[Path] = set()
        for src, rel in _iter_package_files(_OTFS_PACKAGE_ROOT):
            for parent in reversed(rel.parents):
                if parent == Path(".") or parent in created_dirs:
                    continue
                self._workspace._create_dir(parent.as_posix())
                created_dirs.add(parent)
            dst = rel.as_posix()
            if _is_binary_suffix(src):
                with open(src, "rb") as f:
                    self._workspace._write_file_binary(dst, f.read())
            else:
                with open(src, "r") as f:
                    self._workspace._write_file(dst, f.read())

    def setup_workspace(self) -> None:
        if self._workspace is None:
            raise ValueError("Workspace not set")
        with open(_LINK_CONFIG_PATH, "r") as f:
            self._workspace._write_file("link_config.py", f.read())
        with open(_EVAL_SCRIPT_PATH, "r") as f:
            self._workspace._write_file("eval.py", f.read())
        with open(_BASELINE_BLER_PATH, "rb") as f:
            self._workspace._write_file_binary("baseline_bler.pkl", f.read())
        # Re-stage the otfs/ package in case the agent edited it between
        # evaluations. set_workspace() already wrote it on workspace
        # creation so the agent could browse it from turn 1.
        self._stage_otfs_package()

    def cleanup_workspace(self) -> None:
        self._workspace._delete("eval.py")
        self._workspace._delete("link_config.py")
        self._workspace._delete("baseline_bler.pkl")
        # otfs/ is left in place: it is part of the task definition and
        # the agent must be able to import it at any time.

    def _create_tools(self) -> list[BaseTool]:
        def _evaluate() -> str:
            """Evaluate the OTFS detector."""
            return self.run_evaluation(self.default_source_file)

        evaluate_detector = tool(_evaluate)
        evaluate_detector.name = "evaluate_otfs_detector"
        evaluate_detector.description = self._TOOL_DESCRIPTION
        return [evaluate_detector]
