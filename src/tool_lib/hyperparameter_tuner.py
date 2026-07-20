# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Post-process hyperparameter tuning using Optuna (multi-objective).

Runs Bayesian multi-objective optimization on the host after the agent
finishes.  The agent's ``solution.py`` must define tunable parameters
via ``HP.get("name", default, low=..., high=...)`` (see :mod:`hp`).
The search space is extracted automatically by parsing the source AST.

Each trial returns ``(metric, complexity)`` and Optuna optimises both
objectives simultaneously.  After tuning, the 2-D Pareto front over
all completed trials (plus the baseline) is saved as
``pareto_params.json`` in the workspace.
"""

import json
from typing import Optional

import optuna

from leaderboard import HPResult
from .base import EvalToolBase
from .hp import extract_search_space
from .workspace import Workspace

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _compute_pareto_2d(
    points: list[tuple[float, float, dict]],
    higher_is_better: bool,
) -> list[tuple[float, float, dict]]:
    """Return the 2-D Pareto front from *(metric, complexity, payload)* triples.

    Complexity is always minimised.  *higher_is_better* controls metric
    direction.  Uses the sweep-line algorithm: sort by metric (best
    first), sweep while tracking the lowest complexity seen.
    """
    points.sort(key=lambda p: -p[0] if higher_is_better else p[0])
    front: list[tuple[float, float, dict]] = []
    best_complexity = float("inf")
    for m, c, payload in points:
        if c < best_complexity:
            front.append((m, c, payload))
            best_complexity = c
    return front


class HyperparameterTuner:
    """Runs multi-objective Optuna on the host, evaluating via the task's eval tool."""

    def __init__(self, eval_tool: EvalToolBase, higher_is_better: bool,
                 n_trials: int, timeout: int):
        self._eval_tool = eval_tool
        self._higher_is_better = higher_is_better
        self._n_trials = n_trials
        self._timeout = timeout
        self._workspace: Optional[Workspace] = None

    def set_workspace(self, workspace: Workspace) -> None:
        self._workspace = workspace

    def tune(self, source: str, filename: str) -> tuple[str, list[HPResult]]:
        """Run multi-objective Bayesian hyperparameter tuning.

        Returns ``(message, hp_results)`` where *hp_results* is a list
        of :class:`HPResult` entries forming the Pareto front (index 0
        is the baseline).  An empty list means tuning failed entirely.
        """
        if self._workspace is None:
            return "Error: workspace not set.", []

        space = extract_search_space(source)
        if not space:
            return "SKIP: No tunable HP.get() calls found in source.", []

        for name, spec in list(space.items()):
            ptype = spec[0]
            if ptype == "log_float" and spec[1] <= 0:
                space[name] = ["float", spec[1], spec[2]]

        result_re = EvalToolBase.RESULT_RE

        def _parse(output: str) -> tuple[float, float] | None:
            first_line = output.strip().split("\n")[0]
            m = result_re.match(first_line)
            if not m or m.group(1).upper() == "FAILURE" or m.group(2) is None:
                return None
            return float(m.group(2)), float(m.group(3))

        # Evaluate baseline (original defaults, no HP overrides)
        baseline_hp = self._workspace._read_file("hyperparams.json")
        if baseline_hp.startswith("Error"):
            baseline_hp = None

        with self._eval_tool.workspace_ready():
            baseline_output = self._eval_tool.run_evaluation(filename)
            baseline = _parse(baseline_output)

            if baseline is None:
                return (
                    "FAILURE: Baseline evaluation did not produce "
                    "metric and complexity.",
                    [],
                )
            baseline_metric, baseline_complexity = baseline

            metric_dir = "maximize" if self._higher_is_better else "minimize"
            study = optuna.create_study(directions=[metric_dir, "minimize"])

            def objective(trial: optuna.Trial) -> tuple[float, float]:
                params: dict = {}
                for name, spec in space.items():
                    ptype = spec[0]
                    if ptype == "log_float":
                        params[name] = trial.suggest_float(
                            name, spec[1], spec[2], log=True,
                        )
                    elif ptype == "float":
                        params[name] = trial.suggest_float(
                            name, spec[1], spec[2],
                        )
                    elif ptype == "int":
                        params[name] = trial.suggest_int(
                            name, int(spec[1]), int(spec[2]),
                        )
                    elif ptype == "categorical":
                        params[name] = trial.suggest_categorical(
                            name, tuple(spec[1]),
                        )

                self._workspace._write_file(
                    "hyperparams.json", json.dumps(params),
                )
                result = self._eval_tool.run_evaluation(filename)

                parsed = _parse(result)
                if parsed is None:
                    raise optuna.TrialPruned()
                return parsed

            study.optimize(
                objective, n_trials=self._n_trials, timeout=self._timeout,
            )

        # Collect all completed trial points + baseline
        all_points: list[tuple[float, float, dict]] = [
            (baseline_metric, baseline_complexity, {}),
        ]
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                all_points.append((t.values[0], t.values[1], dict(t.params)))

        completed_count = sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )

        # Compute 2-D Pareto front
        pareto = _compute_pareto_2d(all_points, self._higher_is_better)

        # Build indexed results and params mapping
        hp_results: list[HPResult] = []
        pareto_params: dict[str, dict] = {}
        for idx, (m, c, params) in enumerate(pareto):
            hp_results.append(HPResult(hp_index=idx, metric=m, complexity=c))
            pareto_params[str(idx)] = params

        # Save pareto_params.json so the orchestrator can bake lazily
        self._workspace._write_file(
            "pareto_params.json", json.dumps(pareto_params, indent=2),
        )

        # Restore original hyperparams.json (or remove it)
        if baseline_hp is not None:
            self._workspace._write_file("hyperparams.json", baseline_hp)

        msg = (
            f"SUCCESS: {len(pareto)} Pareto-optimal configs "
            f"(baseline + {completed_count} trials)\n"
            f"Pareto front: "
            + ", ".join(
                f"[{hp.hp_index}] metric={hp.metric:.6f} "
                f"complexity={hp.complexity:.6f}"
                for hp in hp_results
            )
        )
        return msg, hp_results
