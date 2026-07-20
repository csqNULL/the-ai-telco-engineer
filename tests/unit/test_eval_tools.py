# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for all EvalTool implementations.

Uses a mock workspace and stub eval-harness files so no Docker, GPU, or
heavyweight simulation is needed.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tool_lib.base import EvalToolBase
from tool_lib.workspace import Workspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_class(task: str) -> type:
    """Import tasks/<task>/eval_tool.py and return its EvalTool class."""
    mod_name = f"{task}_eval_tool"
    path = _REPO / "tasks" / task / "eval_tool.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.EvalTool


TASKS = ("otfs_detector", "pilotless")
CLS: dict[str, type] = {t: _load_class(t) for t in TASKS}

# Some tasks leave some files in the workspace on purpose. An entry may be a
# file or a directory; a directory entry also whitelists everything under it.
TASKS_LEFTOVERS_WHITELIST = {
    "pilotless": ["constellation_points.pkl"],
    # otfs/ is part of the task definition and is intentionally left in place.
    "otfs_detector": ["otfs"],
}


def _mock_ws(host: Path) -> MagicMock:
    """Build a mock Workspace with the methods used by EvalToolBase."""
    ws = MagicMock()
    ws._host_workspace_path = host
    ws._container = MagicMock()
    ws._container.exec = MagicMock(return_value=(True, "SUCCESS, 1.2345"))
    return ws


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=TASKS)
def task_name(request):
    return request.param


@pytest.fixture
def eval_cls(task_name):
    return CLS[task_name]


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return _mock_ws(tmp_path)


@pytest.fixture
def eval_tool(eval_cls, workspace, tmp_path):
    """An EvalTool bound to a mock workspace with a dummy draft.py on the host."""
    et = eval_cls(eval_timeout=30, higher_is_better=False)
    et.set_workspace(workspace)
    (tmp_path / "draft.py").write_text("print('SUCCESS, 1.2345')")
    return et


@pytest.fixture
def stub_files(tmp_path, task_name, monkeypatch):
    """Stub eval-harness file I/O so setup_workspace never touches real disk."""
    # Create a fake data directory BEFORE patching open
    data_stub = tmp_path / "_data_stub"
    data_stub.mkdir()
    (data_stub / "scenario.csv").write_text("x\n1\n")

    _real_open = open

    def _open(path, mode="r", *args, **kwargs):
        # Allow real I/O for files inside the test's tmp_path
        if str(tmp_path) in str(path):
            return _real_open(path, mode, *args, **kwargs)
        # Return a context-manager-compatible mock for everything else
        m = MagicMock()
        m.read.return_value = b"\x80\x04" if "b" in mode else "# stub"
        return m

    monkeypatch.setattr("builtins.open", _open)

    # Redirect the task's _DATA_DIR (if any) to the stub directory
    mod = sys.modules.get(f"{task_name}_eval_tool")
    if mod and hasattr(mod, "_DATA_DIR"):
        monkeypatch.setattr(mod, "_DATA_DIR", data_stub)

    return task_name


# ---------------------------------------------------------------------------
# Tests: Instantiation & tool registration
# ---------------------------------------------------------------------------

def test_default_attrs(eval_cls):
    et = eval_cls(eval_timeout=60, higher_is_better=True)
    assert et._eval_timeout == 60
    assert et._higher_is_better is True
    assert et._workspace is None
    assert et._best_metric is None
    assert et.default_source_file == "draft.py"


@pytest.mark.parametrize(
    "task,expected",
    [
        ("otfs_detector", "evaluate_otfs_detector"),
        ("pilotless", "evaluate_receiver"),
    ],
)
def test_get_tools(task, expected):
    tools = CLS[task](eval_timeout=30, higher_is_better=False).get_tools()
    assert isinstance(tools, list) and len(tools) >= 1
    for t in tools:
        assert t.name == expected
        assert len(t.description) > 10


# ---------------------------------------------------------------------------
# Tests: Workspace binding
# ---------------------------------------------------------------------------

def test_set_workspace_binds(eval_cls, workspace, tmp_path):
    et = eval_cls(eval_timeout=30, higher_is_better=False)
    et._best_metric = 99.0
    et.set_workspace(workspace)
    assert et._workspace is workspace
    assert et._best_metric is None, "Best metric should be reset when setting workspace"

    ws2 = _mock_ws(tmp_path / "ws2")
    et.set_workspace(ws2)
    assert et._workspace is ws2, "Should be able to set another workspace"


# ---------------------------------------------------------------------------
# Tests: setup_workspace / cleanup_workspace
# ---------------------------------------------------------------------------

def test_setup_raises_without_workspace(eval_cls):
    et = eval_cls(eval_timeout=30, higher_is_better=False)
    with pytest.raises(ValueError, match="[Ww]orkspace"):
        et.setup_workspace()


def test_setup_and_cleanup_files(eval_tool, workspace, stub_files):
    """setup_workspace writes eval.py; cleanup deletes it; every written file is cleaned up."""
    eval_tool.setup_workspace()
    written = {c.args[0] for c in workspace._write_file.call_args_list}
    written |= {c.args[0] for c in workspace._write_file_binary.call_args_list}
    dirs = {c.args[0] for c in workspace._create_dir.call_args_list}
    assert "eval.py" in written

    current_task_name = stub_files

    workspace.reset_mock()
    eval_tool.cleanup_workspace()
    deleted = {c.args[0] for c in workspace._delete.call_args_list}
    assert "eval.py" in deleted

    whitelist = TASKS_LEFTOVERS_WHITELIST.get(current_task_name, [])
    for item in written | dirs:
        if any(item == w or item.startswith(w + "/") for w in whitelist):
            continue

        in_deleted = item in deleted or any(
            item.startswith(d + "/") for d in deleted
        )
        assert in_deleted, f"{item!r} written by setup but never cleaned up"


# ---------------------------------------------------------------------------
# Tests: run_evaluation
# ---------------------------------------------------------------------------

def test_run_evaluation(eval_tool, workspace, stub_files):
    result = eval_tool.run_evaluation("draft.py")
    assert isinstance(result, str)
    workspace._container.exec.assert_called_once()
    cmd = workspace._container.exec.call_args[0][0]
    assert cmd == ["python", "eval.py", "draft.py"]
    workspace._write_file.assert_called()
    workspace._delete.assert_called()


def test_run_evaluation_error_on_exec_failure(eval_tool, workspace, stub_files):
    workspace._container.exec.return_value = (False, "segfault")
    assert "Error" in eval_tool.run_evaluation("draft.py")


def test_run_evaluation_cleanup_on_exec_exception(eval_tool, workspace, stub_files):
    workspace._container.exec.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        eval_tool.run_evaluation("draft.py")
    workspace._delete.assert_called()


# ---------------------------------------------------------------------------
# Tests: workspace_ready context manager
# ---------------------------------------------------------------------------

def test_workspace_ready_lifecycle(eval_tool, workspace, stub_files):
    """Setup on enter, no per-call setup/cleanup, cleanup on exit."""
    with eval_tool.workspace_ready():
        assert workspace._write_file.call_count > 0
        workspace._write_file.reset_mock()
        eval_tool.run_evaluation("draft.py")
        workspace._write_file.assert_not_called()
        workspace._delete.assert_not_called()
    workspace._delete.assert_called()


def test_workspace_ready_cleanup_on_exception(eval_tool, workspace, stub_files):
    with pytest.raises(ValueError):
        with eval_tool.workspace_ready():
            raise ValueError("test")
    workspace._delete.assert_called()


# ---------------------------------------------------------------------------
# Tests: auto-save (draft.py -> solution.py)
# ---------------------------------------------------------------------------

def test_auto_save_success_and_failure(eval_tool, workspace, tmp_path, stub_files):
    """Saves on SUCCESS, skips on FAILURE, skips for non-draft files."""
    workspace._container.exec.return_value = (True, "FAILURE,\nERROR, 0.25")
    eval_tool.run_evaluation("draft.py")
    assert not (tmp_path / "solution.py").exists()

    workspace._container.exec.return_value = (True, "SUCCESS, 0.5, 0.25")
    eval_tool.run_evaluation("solution.py")
    assert not (tmp_path / "solution.py").exists()

    (tmp_path / "draft.py").write_text("# good")
    workspace._container.exec.return_value = (True, "SUCCESS, 0.5, 0.25")
    eval_tool.run_evaluation("draft.py")
    assert (tmp_path / "solution.py").exists()


def test_auto_save_metric_tracking(eval_tool, workspace, tmp_path, stub_files):
    """Saves when metric improves (lower-is-better), keeps best on regression."""
    (tmp_path / "draft.py").write_text("v1")
    workspace._container.exec.return_value = (True, "SUCCESS, 2.0, 0.25")
    eval_tool.run_evaluation("draft.py")
    assert (tmp_path / "solution.py").read_text() == "v1"

    (tmp_path / "draft.py").write_text("v2")
    workspace._container.exec.return_value = (True, "SUCCESS, 1.0, 0.25")
    eval_tool.run_evaluation("draft.py")
    assert (tmp_path / "solution.py").read_text() == "v2"

    (tmp_path / "draft.py").write_text("v3-worse")
    workspace._container.exec.return_value = (True, "SUCCESS, 1.5, 0.25")
    eval_tool.run_evaluation("draft.py")
    assert (tmp_path / "solution.py").read_text() == "v2"


def test_auto_save_higher_is_better(eval_cls, workspace, tmp_path, stub_files):
    et = eval_cls(eval_timeout=30, higher_is_better=True)
    et.set_workspace(workspace)

    (tmp_path / "draft.py").write_text("v1")
    workspace._container.exec.return_value = (True, "SUCCESS, 1.0, 0.25")
    et.run_evaluation("draft.py")
    assert (tmp_path / "solution.py").read_text() == "v1"

    (tmp_path / "draft.py").write_text("v2")
    workspace._container.exec.return_value = (True, "SUCCESS, 2.0, 0.25")
    et.run_evaluation("draft.py")
    assert (tmp_path / "solution.py").read_text() == "v2"

    (tmp_path / "draft.py").write_text("v3-worse")
    workspace._container.exec.return_value = (True, "SUCCESS, 1.5, 0.25")
    et.run_evaluation("draft.py")
    assert (tmp_path / "solution.py").read_text() == "v2"


def test_auto_save_suppressed_inside_workspace_ready(
    eval_tool, workspace, tmp_path, stub_files
):
    workspace._container.exec.return_value = (True, "SUCCESS, 0.5, 0.25")
    (tmp_path / "draft.py").write_text("trial")
    with eval_tool.workspace_ready():
        eval_tool.run_evaluation("draft.py")
    assert not (tmp_path / "solution.py").exists()


# ---------------------------------------------------------------------------
# Tests: RESULT_RE parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "line,status,metric,complexity",
    [
        ("SUCCESS, 1.234, 0.25", "SUCCESS", "1.234", "0.25"),
        ("FAILURE,", "FAILURE", None, None),
        ("success, 0.5, 0.44", "success", "0.5", "0.44"),
        ("SUCCESS, -3.5e-2, 0.25", "SUCCESS", "-3.5e-2", "0.25"),
        ("SUCCESS,  42, 0.25", "SUCCESS", "42", "0.25"),
        ("FAILURE, 0.0, 0.0", "FAILURE", "0.0", "0.0"),
    ],
)
def test_result_re_matches_valid(line, status, metric, complexity):
    m = EvalToolBase.RESULT_RE.match(line)
    assert m is not None
    assert m.group(1) == status
    assert m.group(2) == metric
    assert m.group(3) == complexity


@pytest.mark.parametrize(
    "line",
    ["SUCCESS", "MAYBE, 1.0", "", "SUCCESS 1.0", "SUCCESS, abc"],
)
def test_result_re_rejects_invalid(line):
    assert EvalToolBase.RESULT_RE.match(line) is None


# ---------------------------------------------------------------------------
# Tests: LangChain tool invocation
# ---------------------------------------------------------------------------

def test_tool_invoke_returns_eval_result(eval_tool, stub_files):
    tool = eval_tool.get_tools()[0]
    result = tool.invoke({})
    assert isinstance(result, str)
    assert "SUCCESS" in result or "FAILURE" in result or "Error" in result
