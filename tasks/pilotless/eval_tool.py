# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import shlex
from pathlib import Path

from tool_lib.base import EvalToolBase
from langchain_core.tools import tool, BaseTool


# Paths to evaluation scripts and assets.
_EVAL_SCRIPT_PATH = Path(__file__).parent / "eval/eval.py"
_LINK_CONFIG_PATH = Path(__file__).parent / "eval/link_config.py"
_CONSTELLATION_PATH = Path(__file__).parent / "eval/constellation_points.pkl"
_BASELINE_BER_PATH = Path(__file__).parent / "eval/baseline_bler.pkl"


class EvalTool(EvalToolBase):
    """Tool provider for evaluating pilotless OFDM receiver implementations.

    Provides a tool that evaluates an agent-supplied receiver for a single-
    antenna pilotless OFDM uplink. The metric is the Normalized Validation
    Error (NVE) — the mean ratio between the agent's BER and a reference
    BER across the evaluation SNR points (closer to 1 / lower is better).

    Output format (first line only is parsed by the framework):
    ``SUCCESS, <metric>, <complexity>`` or ``FAILURE,`` followed by a
    newline and optional human-readable text.

    The agent must create a file called 'draft.py' in the workspace that
    implements the receiver(y, no) function.
    """

    _TOOL_DESCRIPTION = """\
Evaluate the pilotless OFDM receiver implementation.

Expects `draft.py` in the workspace defining `receiver(y, no)`.
See the task prompt for signature, link_config symbols, and constraints.
Runs Monte Carlo simulation with ``torch.compile(receiver)``; link_config.py is
injected at evaluation time.

Returns:
    str: First line ``SUCCESS, <metric>, <complexity>`` or ``FAILURE,``.
    <metric> is the normalized validation error (closer to 1 / lower is better).
    <complexity> is the average per-call receiver runtime in seconds (lower is better).
    Remaining lines are optional human-readable text."""

    def __init__(self, eval_timeout: int, **kwargs):
        super().__init__(eval_timeout, **kwargs)

    def set_workspace(self, workspace) -> None:
        """Bind to a workspace and stage permanent task assets.

        ``constellation_points.pkl`` is part of the *task definition*, not
        the evaluation harness: the agent must be able to load and inspect
        it from the very first turn (the prompt invites it to). We push it
        in here so it lives in the workspace for the whole agent run and
        is not removed by ``cleanup_workspace`` between evaluations.
        """
        super().set_workspace(workspace)
        with open(_CONSTELLATION_PATH, "rb") as f:
            workspace._write_file_binary("constellation_points.pkl", f.read())

    def setup_workspace(self) -> None:
        if self._workspace is None:
            raise ValueError("Workspace not set")
        with open(_LINK_CONFIG_PATH, "r") as f:
            self._workspace._write_file("link_config.py", f.read())
        with open(_EVAL_SCRIPT_PATH, "r") as f:
            self._workspace._write_file("eval.py", f.read())
        with open(_BASELINE_BER_PATH, "rb") as f:
            self._workspace._write_file_binary("baseline_bler.pkl", f.read())
        # Re-stage the constellation in case the agent edited it between
        # evaluations. set_workspace() also wrote it on workspace creation
        # so the agent could load/inspect it before the first evaluation.
        with open(_CONSTELLATION_PATH, "rb") as f:
            self._workspace._write_file_binary("constellation_points.pkl", f.read())

    def cleanup_workspace(self) -> None:
        self._workspace._delete("eval.py")
        self._workspace._delete("link_config.py")
        self._workspace._delete("baseline_bler.pkl")
        # constellation_points.pkl is left in place: it is part of the task
        # definition and the agent must be able to reload it at any time.

    def _create_tools(self) -> list[BaseTool]:
        """Return a list of tools this provider offers."""

        def _evaluate() -> str:
            """Evaluate the pilotless OFDM receiver."""
            return self.run_evaluation(self.default_source_file)

        evaluate_receiver = tool(_evaluate)
        evaluate_receiver.name = "evaluate_receiver"
        evaluate_receiver.description = self._TOOL_DESCRIPTION
        return [evaluate_receiver]
