# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Base classes for tool providers.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
import re

from langchain_core.tools import BaseTool

from workspace_fs import atomic_replace_under_root


class ToolProvider(ABC):
    """Base class for objects that provide LangChain tools.

    Subclasses must implement :meth:`get_tools`.  Override :meth:`build`
    when the tool requires expensive one-time setup (e.g. building a
    vector-store index) that should happen before worker processes are
    spawned.
    """

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Return a list of tools this provider offers."""
        ...

    @classmethod
    def build(cls, tools_config) -> None:
        """One-time pre-spawn setup for this tool provider.

        Called by the manager process before worker processes are created.
        Use this for expensive, shared initialisation that should only
        happen once (e.g. building a FAISS index on disk).

        The default implementation does nothing.

        Args:
            tools_config: A :class:`~config.ToolsConfig` instance with
                the current run's tool parameters.
        """
        pass


class EvalToolBase(ToolProvider):
    """Abstract base class for task-specific evaluation tools.

    Every ``eval_tool.py`` in a task folder must define an ``EvalTool``
    class that inherits from this base.  Subclasses must implement:

    * :meth:`get_tools` — LangChain tools exposed to the agent.
    * :meth:`_execute` — run the evaluation command and return the
      raw result string.
    * :meth:`setup_workspace` — copy evaluation harness files into the
      workspace so that evaluation commands can run inside the container.
    * :meth:`cleanup_workspace` — remove those files after evaluation.

    The base class provides the concrete :meth:`run_evaluation` (which
    wraps ``_execute`` with automatic setup/cleanup),
    :meth:`set_workspace`, and :meth:`workspace_ready`.

    Registering the agent-facing tool
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Use ``langchain_core.tools.tool`` to wrap a local function that calls
    :meth:`run_evaluation` with the default source file, then set
    ``name`` and ``description`` explicitly.  The **description is the
    primary documentation the agent sees** — it should explain what the
    tool does, what file the agent must create, the expected function
    signature, and the output format::

        from langchain_core.tools import tool

        # In __init__:
        def _evaluate() -> str:
            ""Evaluate the task.""
            return self.run_evaluation("draft.py")

        self.evaluate = tool(_evaluate)
        self.evaluate.name = "evaluate_my_task"
        self.evaluate.description = (
            "Evaluate the algorithm.  "
            "Expects draft.py defining my_function(x). ..."
        )
    """

    RESULT_RE = re.compile(
        r"^(SUCCESS|FAILURE)\s*,"
        r"(?:\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*,"
        r"\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?))?\s*$",
        re.IGNORECASE,
    )
    """Regex for parsing evaluation output first lines.

    Matches ``SUCCESS, <metric>, <complexity>`` or bare ``FAILURE,``.
    Groups: (1) SUCCESS/FAILURE, (2) metric, (3) complexity.
    Groups 2–3 are either both present or both absent.
    """

    def __init__(self, eval_timeout: int, higher_is_better: bool):
        self._eval_timeout = eval_timeout
        self._higher_is_better = higher_is_better
        self._workspace = None
        self._workspace_ready = False
        self._best_metric: float | None = None
        self._tools: list[BaseTool] = self._create_tools()


    @property
    def default_source_file(self) -> str:
        return "draft.py"

    def get_tools(self) -> list[BaseTool]:
        """Return the list of LangChain tools exposed by this provider."""
        return self._tools

    def set_workspace(self, workspace) -> None:
        """Bind the provider to a workspace instance.

        Resets the best-metric tracker so auto-save starts fresh for
        each new workspace / agent run.
        """
        self._workspace = workspace
        self._best_metric = None

    @abstractmethod
    def setup_workspace(self) -> None:
        """Copy evaluation harness files into the workspace.

        Called before running evaluation commands inside the container.
        Paired with :meth:`cleanup_workspace`.
        """
        ...

    @abstractmethod
    def cleanup_workspace(self) -> None:
        """Remove evaluation harness files from the workspace.

        Called after evaluation commands have finished.
        """

    @contextmanager
    def workspace_ready(self):
        """Context manager that keeps evaluation harness files in place.

        Use this when running many evaluations in a row (e.g. during
        hyperparameter tuning) to avoid copying files for every call::

            with eval_tool.workspace_ready():
                for params in trials:
                    eval_tool.run_evaluation("draft.py")

        Inside the context, :meth:`run_evaluation` skips the per-call
        :meth:`setup_workspace` / :meth:`cleanup_workspace`.
        """
        self.setup_workspace()
        self._workspace_ready = True
        try:
            yield
        finally:
            self._workspace_ready = False
            self.cleanup_workspace()

    def run_evaluation(self, filename: str) -> str:
        """Evaluate a workspace file and return the result string.

        This is the single entry point for evaluation.  It is called
        by the agent-facing tool during the run (with ``"draft.py"``)
        and by the framework after the agent finishes (POST_EVAL, with
        ``"solution.py"``).

        Handles :meth:`setup_workspace` / :meth:`cleanup_workspace`
        automatically unless the :meth:`workspace_ready` context
        manager is active.

        When the evaluated file is ``draft.py``, a successful result
        that beats the previous best metric triggers an automatic copy
        of ``draft.py`` → ``solution.py`` on the host.  Auto-save is
        suppressed inside :meth:`workspace_ready` (used by the
        hyperparameter tuner) to avoid overwriting ``solution.py``
        with trial parameters.

        The return value must follow the standard output format:
        first line ``SUCCESS, <metric>, <complexity>`` or bare
        ``FAILURE,``, optionally followed by additional lines.

        Args:
            filename: Name of the file to evaluate (e.g. ``draft.py``).
        """
        if not self._workspace_ready:
            self.setup_workspace()
        try:
            result = self._execute(filename)
        finally:
            if not self._workspace_ready:
                self.cleanup_workspace()

        if not self._workspace_ready and filename == "draft.py":
            self._maybe_auto_save(result)

        return result

    def _maybe_auto_save(self, eval_output: str) -> None:
        """Copy ``draft.py`` → ``solution.py`` if *eval_output* is a new best."""
        first_line = eval_output.strip().split("\n")[0]
        m = self.RESULT_RE.match(first_line)
        if not m or m.group(1).upper() != "SUCCESS" or m.group(2) is None:
            return
        metric = float(m.group(2))
        is_better = (
            self._best_metric is None
            or (metric > self._best_metric if self._higher_is_better
                else metric < self._best_metric)
        )
        if is_better:
            self._best_metric = metric
            host = self._workspace._host_workspace_path
            try:
                atomic_replace_under_root(host, "draft.py", "solution.py")
            except (FileNotFoundError, OSError):
                pass

    def _execute(self, filename: str) -> str:
        """Run the evaluation command and return the raw result.

        Called by :meth:`run_evaluation` after the workspace has been
        prepared.  Subclasses only need to implement this — setup and
        cleanup are handled by the caller.

        Args:
            filename: Name of the file to evaluate (e.g. ``draft.py``).
        """
        success, output = self._workspace._container.exec(
            ["python", "eval.py", filename], timeout=self._eval_timeout,
        )
        return output if success else f"Error:\n{output}"

    @abstractmethod
    def _create_tools(self) -> list[BaseTool]:
        """Return a list of tools this provider offers."""
        ...
