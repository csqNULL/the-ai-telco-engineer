# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Agent module - A LangGraph/LangChain agent.
"""

from dataclasses import asdict
from datetime import datetime
import json
import logging
import signal
from typing import Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from pathlib import Path

from config import LLMConfig, WorkspaceConfig, ToolsConfig, HyperparameterTunerConfig
from leaderboard import HPResult
from tool_lib.base import EvalToolBase, ToolProvider
from tool_lib.hyperparameter_tuner import HyperparameterTuner
from tool_lib.workspace import Workspace
from workspace_fs import (
    file_exists_under_root,
    open_journal_under_root,
    read_text_under_root,
    write_file_under_root,
)

_HP_MODULE_PATH = Path(__file__).parent / "tool_lib" / "hp.py"
_HP_MODULE_SOURCE = _HP_MODULE_PATH.read_text()


class AgentTimeoutError(BaseException):
    """Raised when agent execution exceeds the timeout.

    Inherits from BaseException (not Exception) so that it bypasses
    LangGraph's ToolNode and run_with_retry handlers, both of which
    catch ``except Exception`` and would silently swallow the timeout.
    The worker retry loop in _worker_fn catches it explicitly before
    the generic ``except Exception`` clause, so retry semantics are
    correct.
    """
    pass


SOLUTION_FILE = "solution.py"
DRAFT_FILE = "draft.py"
RESULT_FILE = "result.json"
JOURNAL_FILE = "journal.log"


# Prompt template to enrich the user query
# Placeholders: original_query, assigned_approach_section, metric_direction
RESULT_PROMPT_TEMPLATE = """
You are an implementing agent in an automated optimization loop. You receive one task and one assigned approach. There is no prior conversation: this message is your only context.

## Context
- You run in an isolated workspace with tools: read/write files, copy files, run code, and an evaluation tool.
- Optimization target: {metric_direction} values are BETTER.
- You write your code in `draft.py`. The evaluation tool tests this file.
- Ignore `solution.py` and `journal.log` — they are managed by the framework.

**Your goal is to achieve the best possible metric for the assigned approach.** When reference code is provided, use it as raw material to edit; do not submit it unchanged. When no reference is provided, implement the approach from scratch.

## Task
{original_query}

## Assigned approach (you MUST follow this)
{assigned_approach_section}

Do not switch to a different strategy. If reference code is shown, it is a starting point you may rewrite freely — the assigned approach takes precedence whenever they conflict, and its headline mechanism must appear in your final `draft.py`. If during implementation you find yourself thinking the assigned approach "doesn't apply here", "reduces to" something simpler, "would need information you don't have", "is essentially equivalent to" a different method, or "isn't really meaningful in this setting" — treat that as a signal to debug your implementation, not as licence to substitute.

## Workflow

### Step 1 — Understand the setup
Read the evaluation tool description to learn the expected function signature, available imports, and constraints.

### Step 2 — Implement your approach
Write your solution into `draft.py`. Use provided configuration objects; do not hardcode values. Keep all logic inline — importing installed libraries is fine, but do not import from other files you create.

**Make parameters tunable**: every numerical constant or mode that affects performance (thresholds, weights, decay factors, model choices, etc.) MUST be defined using the `HP` helper:

    from hp import HP

    threshold = HP.get("threshold", 0.05, low=0.01, high=0.1)
    mode = HP.get("mode", "A", choices=["A", "B", "C"])
    lr = HP.get("lr", 1e-4, low=1e-6, high=1e-2, log=True)

`HP.get(name, default, ...)` always returns `default`. Use `low`/`high` to declare the valid range, `choices` for categorical options, and `log=True` for log-scale parameters. This is mandatory for every tunable constant in your code.

### Step 3 — Evaluate
Call the evaluation tool. It returns a metric ({metric_direction} is better), optional hints, or an error.

### Step 4 — Iterate and optimize
Keep improving: adjust parameters, add algorithmic refinements, re-evaluate. Repeat Steps 2–3.
Stop only when you run out of ideas or time.

## Rules
- Do not call the evaluation tool twice without changing code between calls.
- After the first success, keep iterating — a run with only 1-2 attempts wastes your budget.
- Do not deviate from the assigned approach.
- `draft.py` must be fully self-contained. All logic you write must be defined inline. Importing from installed libraries (e.g. `numpy`, `scipy`, `torch`) and `from hp import HP` is fine, but do NOT import from other files you created (e.g. `from helper import solve`).

=== END INSTRUCTIONS ===
"""


_EVAL_FIRST_LINE = EvalToolBase.RESULT_RE


def parse_eval_output(
    eval_output: str,
) -> tuple[bool, float | None, float | None, str | None]:
    """Parse evaluation tool output.

    Accepted formats:

    * ``SUCCESS, <metric>, <complexity>`` — both values mandatory.
    * ``FAILURE,`` — bare, no values.

    Returns:
        ``(success, metric, complexity, info)``.
        *metric* and *complexity* are ``None`` for bare ``FAILURE,``.
    """
    if not eval_output or not isinstance(eval_output, str):
        return False, None, None, None
    lines = eval_output.strip().split("\n")
    first_line = lines[0].strip()
    m = _EVAL_FIRST_LINE.match(first_line)
    if not m:
        return False, None, None, None
    success = m.group(1).upper() == "SUCCESS"
    info = "\n".join(lines[1:]).strip() or None
    if m.group(2) is None:
        return success, None, None, info
    return success, float(m.group(2)), float(m.group(3)), info


class Agent:
    """
    A LangGraph/LangChain agent that can process queries.
    """

    def __init__(self, llm_config: LLMConfig,
                workspace_config: WorkspaceConfig,
                evaluation_tool_type: type[ToolProvider],
                tool_factory_type: Optional[type[ToolProvider]],
                tools_config: ToolsConfig,
                higher_is_better: bool,
                eval_timeout: int,
                hp_tuner_config: HyperparameterTunerConfig,
                gpu_id: Optional[int] = None):
        """
        Initialize the agent.

        Args:
            llm_config: The configuration for the language model to use.
            workspace_config: Configuration for workspace Docker containers.
            evaluation_tool_type: The type of evaluation tool provider to use.
            tool_factory_type: Optional factory class for creating additional tools.
            tools_config: Configuration for tools.
            higher_is_better: If True, higher metric values are better (e.g., throughput).
                             If False, lower metric values are better (e.g., error rate).
            eval_timeout: Timeout in seconds for each evaluation run.
            hp_tuner_config: Configuration for post-process hyperparameter tuning.
            gpu_id: Pin workspaces created by this agent to the given GPU.
                When None, all GPUs are visible.
        """
        self.workspace_config = workspace_config
        self.higher_is_better = higher_is_better
        self._gpu_id = gpu_id
        self._current_workspace_root = None  # Set during run() for timeout logging

        # Build the LLM
        self.llm = ChatOpenAI(**asdict(llm_config))
        # Silence HTTP-level chatter from the individual HTTP requests
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Evaluation tool
        self.evaluation_tool = evaluation_tool_type(
            eval_timeout=eval_timeout,
            higher_is_better=higher_is_better,
        )

        # Create task-specific tools from factory (if provided)
        self.tool_factory = None
        if tool_factory_type is not None:
            self.tool_factory = tool_factory_type(
                tools_config,
                eval_tool=self.evaluation_tool,
                higher_is_better=higher_is_better,
            )

        # Post-process hyperparameter tuner
        self.hp_tuner = HyperparameterTuner(
            eval_tool=self.evaluation_tool,
            higher_is_better=higher_is_better,
            n_trials=hp_tuner_config.n_trials,
            timeout=hp_tuner_config.timeout,
        )

    def enrich_query(
        self,
        query: str,
        assigned_approach_section: str = "",
        prompt_template: str = "",
    ) -> str:
        """
        Enrich the original query with instructions and the assigned approach section.

        The assigned approach section is built by the orchestrator (idea description
        followed by its reference code). This method injects it into the template.

        Args:
            query: The original user query.
            assigned_approach_section: Pre-built section from the orchestrator (idea + its reference code).
            prompt_template: Optional override for RESULT_PROMPT_TEMPLATE.
                If non-empty, used instead of the default template.

        Returns:
            The enriched query with instructions and assigned approach section.
        """
        if not assigned_approach_section.strip():
            assigned_approach_section = "(No specific approach assigned.)"
        template = (
            prompt_template
            if prompt_template and prompt_template.strip()
            else RESULT_PROMPT_TEMPLATE
        )
        metric_direction = "HIGHER" if self.higher_is_better else "LOWER"

        # Use .replace() instead of .format() so that LLM-refined templates
        # containing literal curly braces don't crash with KeyError/ValueError.
        result = template
        result = result.replace("{original_query}", query)
        result = result.replace("{assigned_approach_section}", assigned_approach_section)
        result = result.replace("{metric_direction}", metric_direction)
        return result

    def _log_message(self, ws_root: Path, msg) -> None:
        """
        Log a message to the journal file.

        Args:
            ws_root: Host path to the workspace directory.
            msg: The message object to log.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []

        if msg.type == "ai":
            # AI message - may contain content and/or tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})
                    tool_id = tc.get("id", "")
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in tool_args.items())
                    lines.append(f"[{timestamp}] [TOOL_CALL] {tool_name}({args_str}) [id: {tool_id}]")

            if hasattr(msg, "content") and msg.content:
                lines.append(f"[{timestamp}] [AI]\n{msg.content}")

        elif msg.type == "tool":
            # Tool response message
            tool_name = getattr(msg, "name", "unknown")
            tool_call_id = getattr(msg, "tool_call_id", "")
            content = getattr(msg, "content", "") or ""
            lines.append(f"[{timestamp}] [TOOL_RESPONSE] {tool_name} [id: {tool_call_id}]\n{content}")

        elif msg.type == "human":
            # Human/user message
            content = getattr(msg, "content", "") or ""
            lines.append(f"[{timestamp}] [USER]\n{content}")

        else:
            # Other message types
            content = getattr(msg, "content", "") or ""
            lines.append(f"[{timestamp}] [{msg.type.upper()}]\n{content}")

        if lines:
            with open_journal_under_root(ws_root, JOURNAL_FILE, "a") as f:
                for line in lines:
                    f.write(line + "\n\n")

    def _timeout_handler(self, signum, frame) -> None:
        """
        Signal handler for timeout. Logs the timeout event and raises an exception.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        if self._current_workspace_root:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open_journal_under_root(
                self._current_workspace_root, JOURNAL_FILE, "a",
            ) as f:
                f.write(f"[{timestamp}] [TIMEOUT] Agent execution timed out\n\n")

        raise AgentTimeoutError("Agent execution timed out")

    def _run_post_processing(self, workspace: Workspace) -> None:
        """Run HP tuning (if applicable) and save result.json.

        1. If ``solution.py`` or ``draft.py`` exists and contains tunable
           ``HP.get()`` calls, run multi-objective Optuna HP tuning.  The
           tuner returns a list of Pareto-optimal :class:`HPResult` entries
           and writes ``pareto_params.json`` to the workspace.
        2. If tuning was skipped (no ``HP.get()`` calls) or failed,
           run a standalone evaluation and produce a single HPResult.
        3. If neither file exists, write a failure ``result.json`` so any
           container-written result file is overwritten.

        The ``result.json`` dict has the shape::

            {"success": True,
             "results": [{"hp_index": 0, "metric": ..., "complexity": ...}, ...]}
        """
        def _save_result(result_dict: dict) -> None:
            try:
                write_file_under_root(
                    ws_path, RESULT_FILE, json.dumps(result_dict),
                )
            except Exception as save_e:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open_journal_under_root(ws_path, JOURNAL_FILE, "a") as f:
                    f.write(f"[{ts}] [POST_EVAL] Save failed: {save_e}\n\n")

        ws_path = workspace._host_workspace_path
        if file_exists_under_root(ws_path, SOLUTION_FILE):
            eval_file = SOLUTION_FILE
        elif file_exists_under_root(ws_path, DRAFT_FILE):
            eval_file = DRAFT_FILE
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open_journal_under_root(ws_path, JOURNAL_FILE, "a") as f:
                f.write(
                    f"[{timestamp}] [POST_EVAL] Neither '{SOLUTION_FILE}' nor "
                    f"'{DRAFT_FILE}' found; saving failure {RESULT_FILE}\n\n"
                )
            _save_result({"success": False, "results": []})
            return

        # --- Phase 1: hyperparameter tuning ---
        hp_results: list[HPResult] = []
        source = read_text_under_root(ws_path, eval_file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open_journal_under_root(ws_path, JOURNAL_FILE, "a") as f:
            f.write(
                f"[{timestamp}] [POST_HP_TUNE] Starting hyperparameter "
                f"tuning on {eval_file}\n\n"
            )
        try:
            self.hp_tuner.set_workspace(workspace)
            message, hp_results = self.hp_tuner.tune(
                source=source, filename=eval_file,
            )
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open_journal_under_root(ws_path, JOURNAL_FILE, "a") as f:
                f.write(f"[{timestamp}] [POST_HP_TUNE] Result:\n{message}\n\n")
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open_journal_under_root(ws_path, JOURNAL_FILE, "a") as f:
                f.write(f"[{timestamp}] [POST_HP_TUNE] Failed: {e}\n\n")

        if hp_results:
            _save_result({
                "success": True,
                "results": [hp.to_dict() for hp in hp_results],
            })
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open_journal_under_root(ws_path, JOURNAL_FILE, "a") as f:
                f.write(
                    f"[{timestamp}] [POST_EVAL] Saved {len(hp_results)} "
                    f"Pareto-optimal results to {RESULT_FILE} (from HP tuning)\n\n"
                )
            return

        # --- Phase 2: standalone evaluation (tuning skipped or failed) ---
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            eval_output = self.evaluation_tool.run_evaluation(eval_file)
            if not isinstance(eval_output, str):
                eval_output = str(eval_output)
        except Exception as e:
            error_msg = str(e)
            with open_journal_under_root(ws_path, JOURNAL_FILE, "a") as f:
                f.write(
                    f"[{timestamp}] [POST_EVAL] Evaluation tool failed: "
                    f"{error_msg}\n\n"
                )
            _save_result({"success": False, "results": []})
            return

        with open_journal_under_root(ws_path, JOURNAL_FILE, "a") as f:
            f.write(
                f"[{timestamp}] [POST_EVAL] Evaluation output "
                f"({eval_file}):\n{eval_output}\n\n"
            )
        success, metric, complexity, info = parse_eval_output(eval_output)
        if not success or metric is None:
            with open_journal_under_root(ws_path, JOURNAL_FILE, "a") as f:
                f.write(
                    f"[{timestamp}] [POST_EVAL] Evaluation did not succeed; "
                    f"saving failure {RESULT_FILE}\n\n"
                )
            _save_result({"success": False, "results": []})
            return

        _save_result({
            "success": True,
            "results": [{"hp_index": 0, "metric": metric, "complexity": complexity}],
        })
        with open_journal_under_root(ws_path, JOURNAL_FILE, "a") as f:
            f.write(
                f"[{timestamp}] [POST_EVAL] Saved metric={metric}, "
                f"complexity={complexity} to {RESULT_FILE}\n\n"
            )

    def run(
        self,
        workspace_id: str,
        query: str,
        timeout: Optional[int] = None,
        assigned_approach_section: str = "",
        prompt_template: str = "",
    ) -> str:
        """
        Run the agent on a query and return the final response.

        Args:
            workspace_id: The workspace ID to use for this run.
            query: The user query to process.
            timeout: Optional timeout in seconds for the agent run.
                     If None, no timeout is applied.
            assigned_approach_section: Pre-built section from the orchestrator
                (idea followed by its reference code).
            prompt_template: Optional override for the default prompt template.
                If non-empty, used instead of RESULT_PROMPT_TEMPLATE.

        Returns:
            The final AI response as a string.

        Raises:
            AgentTimeoutError: If the agent execution exceeds the timeout.
        """
        effective_query = self.enrich_query(
            query, assigned_approach_section, prompt_template,
        )

        ws_root = Path(self.workspace_config.base_path) / workspace_id
        self._current_workspace_root = ws_root

        # Set up timeout if specified
        if timeout is not None:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(timeout)

        try:
            # Use context manager for automatic workspace cleanup after run
            # No workspace inheritance - each candidate starts fresh
            with Workspace(workspace_id=workspace_id,
                           gpu_id=self._gpu_id,
                           cfg=self.workspace_config) as workspace:
                # Write the query as the first entry in the journal
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open_journal_under_root(ws_root, JOURNAL_FILE, "w") as f:
                    f.write(f"[{timestamp}] [QUERY]\n{effective_query}\n\n")

                # Set workspace on tool factory (if provided)
                if self.tool_factory is not None:
                    self.tool_factory.set_workspace(workspace)

                # Set workspace on evaluation tool
                self.evaluation_tool.set_workspace(workspace)

                # Copy the HP helper module into the workspace
                workspace._write_file("hp.py", _HP_MODULE_SOURCE)

                # Collect all tools
                tools = (workspace.get_tools()
                         + self.evaluation_tool.get_tools())
                if self.tool_factory is not None:
                    tools += self.tool_factory.get_tools()
                # Build the agent
                agent = create_agent(self.llm, tools)

                final_response = ""
                tool_call_count = 0
                last_ai_has_tool_calls = None
                last_ai_content_preview = ""
                last_ai_content_len = 0
                try:
                    for event in agent.stream({"messages": [("user", effective_query)]}):
                        for _, node_output in event.items():
                            if "messages" in node_output:
                                for msg in node_output["messages"]:
                                    # Log all messages to journal
                                    self._log_message(ws_root, msg)
                                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                                        tool_call_count += len(msg.tool_calls)
                                    if msg.type == "ai":
                                        if msg.content:
                                            final_response = msg.content
                                        last_ai_has_tool_calls = bool(
                                            getattr(msg, "tool_calls", None)
                                        )
                                        content = (msg.content or "").strip()
                                        last_ai_content_len = len(content)
                                        last_ai_content_preview = (
                                            content[:200] + "..." if len(content) > 200
                                            else content
                                        )
                except AgentTimeoutError:
                    if timeout is not None:
                        signal.alarm(0)
                    self._run_post_processing(workspace)
                    raise

                # Normal completion: run POST_EVAL (cancel alarm so it is not interrupted).
                if timeout is not None:
                    signal.alarm(0)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open_journal_under_root(ws_root, JOURNAL_FILE, "a") as f:
                    f.write(
                        f"[{timestamp}] [RUN_END] tool_calls={tool_call_count} | "
                        f"last_ai_has_tool_calls={last_ai_has_tool_calls} | "
                        f"last_ai_content_len={last_ai_content_len}\n"
                    )
                    if last_ai_content_preview:
                        f.write(f"[{timestamp}] [RUN_END] last_ai_content_preview: {last_ai_content_preview!r}\n")
                    f.write("\n")
                self._run_post_processing(workspace)
            return final_response

        finally:
            if timeout is not None:
                signal.alarm(0)
