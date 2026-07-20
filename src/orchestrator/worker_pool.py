# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multiprocessing worker pool for running agents in parallel."""

import multiprocessing as mp
import queue
import random
import signal
import time
from typing import Optional

import printer
from config import LLMConfig, WorkspaceConfig, ToolsConfig, HyperparameterTunerConfig, Config, LogConfig
from agent import Agent, AgentTimeoutError
from tool_lib.base import ToolProvider
from utils import (
    INITIAL_BACKOFF_SECONDS,
    MAX_BACKOFF_SECONDS,
    MAX_RATE_LIMIT_RETRIES,
    is_rate_limit_error,
)

from .models import Task, TaskResult


def _worker_fn(
    worker_id: int,
    logging_config: LogConfig,
    print_lock: mp.Lock,
    agent_llm: LLMConfig,
    workspace_config: WorkspaceConfig,
    evaluation_tool_type: type[ToolProvider],
    tool_factory_type: Optional[type[ToolProvider]],
    tools_config: ToolsConfig,
    higher_is_better: bool,
    eval_timeout: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    num_gpus: int,
    hp_tuner_config: HyperparameterTunerConfig,
):
    """Worker subprocess entry point.

    Creates a single :class:`Agent` and processes tasks from *task_queue*
    until a ``None`` sentinel is received.  Rate-limit errors are retried
    with exponential backoff.
    """
    def _sigterm_handler(signum, frame):
        raise SystemExit("Worker terminated")
    signal.signal(signal.SIGTERM, _sigterm_handler)

    printer.init(logging_config, print_lock, f"WORKER-{worker_id}")

    # Assign a specific GPU to this worker.
    gpu_id = worker_id % num_gpus if num_gpus > 1 else None

    agent = Agent(
        agent_llm, workspace_config, evaluation_tool_type,
        tool_factory_type, tools_config, higher_is_better,
        eval_timeout=eval_timeout,
        hp_tuner_config=hp_tuner_config,
        gpu_id=gpu_id,
    )

    while True:
        task = task_queue.get()
        if task is None:
            break

        workspace_id = task.workspace_id
        query = task.query
        printer.set_header(f"AGENT-{workspace_id}")

        last_error = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = agent.run(
                    workspace_id,
                    query,
                    timeout=task.timeout,
                    assigned_approach_section=task.assigned_approach_section,
                    prompt_template=task.prompt_template,
                )
                result = TaskResult(
                    workspace_id=workspace_id,
                    query=query,
                    response=response,
                    success=True,
                    generation=task.generation,
                )
                break

            except AgentTimeoutError as e:
                result = TaskResult(
                    workspace_id=workspace_id,
                    query=query,
                    response="",
                    success=False,
                    error=f"Timeout: {str(e)}",
                    generation=task.generation,
                )
                break

            except Exception as e:
                last_error = e
                if is_rate_limit_error(e) and attempt < MAX_RATE_LIMIT_RETRIES:
                    backoff = min(
                        INITIAL_BACKOFF_SECONDS * (2 ** attempt),
                        MAX_BACKOFF_SECONDS,
                    )
                    jitter = random.uniform(0, backoff * 0.25)
                    wait_time = backoff + jitter
                    printer.log(
                        f"Rate limit hit, attempt {attempt + 1}/"
                        f"{MAX_RATE_LIMIT_RETRIES}. "
                        f"Waiting {wait_time:.1f}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue

                printer.log(
                    f"ERROR: {workspace_id} failed: "
                    f"{type(e).__name__}: {e}"
                )
                result = TaskResult(
                    workspace_id=workspace_id,
                    query=query,
                    response="",
                    success=False,
                    error=str(e),
                    generation=task.generation,
                )
                break

        result_queue.put(result)


class WorkerPool:
    """Manages a pool of agent worker processes.

    Use as a context manager or call :meth:`start` / :meth:`stop` manually.
    """

    def __init__(
        self,
        config: Config,
        num_workers: int,
        agent_llm: LLMConfig,
        workspace_config: WorkspaceConfig,
        evaluation_tool_type: type[ToolProvider],
        tool_factory_type: Optional[type[ToolProvider]],
        tools_config: ToolsConfig,
        higher_is_better: bool,
        eval_timeout: int,
        num_gpus: int,
        hp_tuner_config: HyperparameterTunerConfig,
    ):
        mp.set_start_method("spawn", force=True)
        self.num_workers = num_workers
        self._config = config
        self._agent_llm = agent_llm
        self._workspace_config = workspace_config
        self._evaluation_tool_type = evaluation_tool_type
        self._tool_factory_type = tool_factory_type
        self._tools_config = tools_config
        self._higher_is_better = higher_is_better
        self._eval_timeout = eval_timeout
        self._num_gpus = num_gpus
        self._hp_tuner_config = hp_tuner_config

        self._task_queue: mp.Queue = mp.Queue()
        self._result_queue: mp.Queue = mp.Queue()
        self._print_lock: mp.Lock = mp.Lock()
        self._workers: list[mp.Process] = []

    @property
    def print_lock(self) -> mp.Lock:
        """Shared lock used for process-safe printing."""
        return self._print_lock

    def start(self):
        """Start all worker processes, running one-time tool setup first."""
        printer.log(f"Starting {self.num_workers} agent workers...")

        self._evaluation_tool_type.build(self._tools_config)
        if self._tool_factory_type is not None:
            if not hasattr(self._tool_factory_type, "TOOL_TYPES"):
                raise AttributeError(
                    f"{self._tool_factory_type.__name__} must define a "
                    "TOOL_TYPES class attribute listing its ToolProvider types."
                )
            for tool_type in self._tool_factory_type.TOOL_TYPES:
                tool_type.build(self._tools_config)

        for i in range(self.num_workers):
            p = mp.Process(
                target=_worker_fn,
                args=(
                    i,
                    self._config.logging_config,
                    self._print_lock,
                    self._agent_llm,
                    self._workspace_config,
                    self._evaluation_tool_type,
                    self._tool_factory_type,
                    self._tools_config,
                    self._higher_is_better,
                    self._eval_timeout,
                    self._task_queue,
                    self._result_queue,
                    self._num_gpus,
                    self._hp_tuner_config,
                ),
            )
            gpu_label = f", GPU {i % self._num_gpus}" if self._num_gpus > 1 else ""
            p.start()
            self._workers.append(p)
            printer.log(f"Worker {i} started (PID: {p.pid}{gpu_label})")

        printer.log(f"All {self.num_workers} workers ready.")

    def stop(self):
        """Stop all worker processes."""
        printer.log("Stopping workers...")
        for _ in self._workers:
            self._task_queue.put(None)
        for i, p in enumerate(self._workers):
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                printer.log(f"Worker {i} terminated")
            else:
                printer.log(f"Worker {i} stopped")
        self._workers.clear()
        printer.log("All workers stopped.")

    def submit(self, task: Task) -> None:
        """Submit a task to the worker pool."""
        self._task_queue.put(task)

    def get_result(self, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get a result from the result queue."""
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            # Expected when callers poll with a short timeout
            return None
        except Exception as e:
            printer.log(f"ERROR: Failed to get result from queue: {type(e).__name__}: {e}")
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
