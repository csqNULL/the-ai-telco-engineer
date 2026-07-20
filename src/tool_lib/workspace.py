# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from pathlib import Path
import shutil
from typing import Optional

from langchain_core.tools import tool, BaseTool

from config import WorkspaceConfig # TODO: proper repository layout
from .base import ToolProvider
from .container_interface import ContainerInterface


########################################################
# Tools
########################################################

class Workspace(ToolProvider):
    """A tool provider for running isolated workspace.

    Each workspace is a separate Docker container, with tools for file management and Python code execution.

    Can be used as a context manager for automatic cleanup:

        with Workspace("ws_id", host_workspace_path="workspaces",
                        docker_image="img", memory_limit="16g",
                        pids_limit=2048, use_gpu=True) as ws:
            ws._container.run_python_code("print('Hello')")

    Note that even though file management operations could in principle be performed fully locally,
    we prefer to forward them to `ContainerInterface` so that the agent's code always sees the
    latest version of the files, even in case of e.g. caching / sync delays in the network-mounted
    workspace directory.
    """

    def __init__(self,
                 cfg: WorkspaceConfig,
                 workspace_id: str,
                 parent_workspace_id: Optional[str] = None,
                 gpu_id: Optional[int] = None):
        """
        Initialize an isolated workspace for running tools in a Docker container.

        Args:
            workspace_id: Unique identifier for this workspace.
            gpu_id: Pin the container to a specific GPU via CUDA_VISIBLE_DEVICES.
                When None, all GPUs are visible inside the container.
            parent_workspace_id: If provided, the new workspace is initialized by copying files from the parent workspace.
            cfg: workspace configuration options.
        """

        self._workspace_id = workspace_id
        self._host_workspace_path = Path(cfg.base_path) / workspace_id
        self._host_parent_workspace_path = (Path(cfg.base_path) / parent_workspace_id) if (parent_workspace_id is not None) else None
        self._cfg = cfg

        self._prepare_workspace()

        container_name = f"workspace_{self._workspace_id}"
        self._container = ContainerInterface(
            name=container_name,
            cfg=cfg.container,
            host_workspace_path=self._host_workspace_path.resolve(),
            gpu_id=gpu_id,
        )

        self._create_tools()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops and cleans up the workspace."""
        self.stop_workspace(remove_host_workspace=False)
        return False  # Don't suppress exceptions

    def _prepare_workspace(self):
        """Prepare the workspace directory on the host.

        If a parent workspace ID is provided, copies the parent's directory as the initial workspace state.
        If no parent is provided, creates an empty workspace directory.

        Raises:
            ValueError: If the parent workspace does not exist, or if the target workspace already exists when inheriting.
        """

        # If the workspace is inherited from a parent workspace, copy the parent workspace directory to the new workspace path
        if self._host_parent_workspace_path is not None:
            if not self._host_parent_workspace_path.exists():
                raise ValueError(f"Parent workspace {self._host_parent_workspace_path} does not exist")
            if self._host_workspace_path.exists():
                raise ValueError(f"Workspace {self._workspace_id} already exists")

            # Recursively copy the parent workspace directory to the new workspace path
            shutil.copytree(self._host_parent_workspace_path, self._host_workspace_path)
        #
        else:
            os.makedirs(self._host_workspace_path, exist_ok=True)

    # ---------- Code execution tools

    def _install_package(self, package: str) -> str:
        # Note: docstring automatically copied from container interface.
        return self._container.install_package(package)

    def _run_python_code(self, code: str, timeout: int | None = None) -> str:
        # Note: docstring automatically copied from container interface.
        return self._container.run_python_code(code, timeout)

    def _run_python_code_in_home(self, code: str, timeout: int | None = None) -> str:
        # Internal-only: not registered as a tool. See ContainerInterface.run_python_code_in_home.
        return self._container.run_python_code_in_home(code, timeout)

    def _run_python_script(self, script_path: str, timeout: int | None = None) -> str:
        # Note: docstring automatically copied from container interface.
        return self._container.run_python_script(script_path, timeout)

    # ---------- File management tools

    def _read_file(self, path: str) -> str:
        return self._container.read_file(path)

    def _write_file(self, path: str, content: str) -> str:
        return self._container.write_file(path, content)

    def _write_file_binary(self, path: str, content: bytes) -> str:
        return self._container.write_file_binary(path, content)

    def _list_dir(self, path: str = ".") -> str:
        return self._container.list_dir(path)

    def _create_dir(self, path: str) -> str:
        return self._container.create_dir(path)

    def _edit_file(self, path: str, old_string: str, new_string: str) -> str:
        return self._container.edit_file(path, old_string, new_string)

    def _copy_file(self, source: str, destination: str) -> str:
        return self._container.copy_file(source, destination)

    def _delete(self, path: str) -> str:
        return self._container.delete_file_or_dir(path)


    def _create_tools(self):
        """Create tools from bound methods."""
        self.install_package = tool(self._install_package)
        self.run_python_code = tool(self._run_python_code)
        self.run_python_script = tool(self._run_python_script)
        self.read_file = tool(self._read_file)
        self.write_file = tool(self._write_file)
        self.edit_file = tool(self._edit_file)
        self.copy_file = tool(self._copy_file)
        self.list_dir = tool(self._list_dir)
        self.create_dir = tool(self._create_dir)
        self.delete = tool(self._delete)

    def get_tools(self) -> list[BaseTool]:
        """Get the tools for the workspace."""
        return [
            self.install_package,
            self.run_python_code,
            self.run_python_script,
            self.read_file,
            self.write_file,
            self.edit_file,
            self.copy_file,
            self.list_dir,
            self.create_dir,
            self.delete
        ]

    def stop_workspace(self, remove_host_workspace: bool = False):
        """Stop the workspace.

        Args:
            remove_host_workspace: Whether to remove the host workspace directory.
        """
        self._container.stop()
        if remove_host_workspace:
            shutil.rmtree(self._host_workspace_path)


# Automatically copy docstrings to avoid duplication
Workspace._install_package.__doc__ = ContainerInterface.install_package.__doc__
Workspace._run_python_code.__doc__ = ContainerInterface.run_python_code.__doc__
Workspace._run_python_script.__doc__ = ContainerInterface.run_python_script.__doc__
Workspace._read_file.__doc__ = ContainerInterface.read_file.__doc__
Workspace._write_file.__doc__ = ContainerInterface.write_file.__doc__
Workspace._write_file_binary.__doc__ = ContainerInterface.write_file_binary.__doc__
Workspace._list_dir.__doc__ = ContainerInterface.list_dir.__doc__
Workspace._create_dir.__doc__ = ContainerInterface.create_dir.__doc__
Workspace._edit_file.__doc__ = ContainerInterface.edit_file.__doc__
Workspace._copy_file.__doc__ = ContainerInterface.copy_file.__doc__
Workspace._delete.__doc__ = ContainerInterface.delete_file_or_dir.__doc__
