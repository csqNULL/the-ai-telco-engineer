# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from pathlib import Path
import re
import subprocess

# TODO: proper repository layout
from config import ContainerConfig
import printer
from workspace_fs import (
    copy_file_under_root,
    delete_under_root,
    listdir_under_root,
    mkdir_under_root,
    read_text_under_root,
    write_file_under_root,
)


AGENT_HOME = Path("/agent_home")


class ContainerInterface:
    """
    Holder class to manage a Docker container, intended to expose tools to an Agent.

    Note that the owner of this object is responsible for stopping the container when it is no longer needed.
    """

    # Default timeout values (in seconds)
    DEFAULT_EXEC_TIMEOUT = 60
    DEFAULT_PIP_TIMEOUT = 300
    DEFAULT_SCRIPT_TIMEOUT = 300
    # Bounded waits for Docker lifecycle calls. A wedged container (stuck syscall,
    # bad GPU state, hung filesystem...) must NEVER be allowed to block the
    # orchestrator: we kill, then bounded-rm, then move on and let the daemon
    # reap the rest asynchronously.
    DOCKER_KILL_TIMEOUT = 30
    DOCKER_RM_TIMEOUT = 60
    # Default Docker bridge used for short-lived outbound access (e.g. pip install).
    BRIDGE_NETWORK = "bridge"
    # Strict PyPI requirement specifier: a normal package name with optional
    # extras and version constraints. Deliberately rejects paths, URLs, VCS
    # specs (``git+...``), leading ``-`` (pip flag injection), and ``@`` so an
    # agent cannot coerce pip into building/executing arbitrary code (e.g. via a
    # local ``setup.py``/PEP-517 hook) inside the network-enabled install
    # container.
    PACKAGE_SPEC_RE = re.compile(
        r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?"
        r"(\[[A-Za-z0-9,._-]+\])?"
        r"([<>=!~]=?[A-Za-z0-9.*+!-]+)?$"
    )

    def __init__(
        self,
        name: str,
        cfg: ContainerConfig,
        host_workspace_path: Path,
        gpu_id: int | None = None,
    ):
        """
        Automatically builds the Docker image (if missing) and starts the container.

        Args:
            name: Name of the container.
            cfg: Container configuration.
            gpu_id: Pin the container to a specific GPU via CUDA_VISIBLE_DEVICES.
                When None, all GPUs are visible inside the container.
        """
        self.cfg = cfg
        # Workspace directory path, as seen by the agent inside the container.
        self.workspace_root = Path(cfg.workspace_mount_point)

        self._name = name
        self._use_gpu = cfg.use_gpu and self.nvidia_runtime_available()

        # Resolve visible GPUs.
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_gpus is not None:
            visible_gpus = set(int(v) for v in visible_gpus.split(","))

        if gpu_id is None:
            self._visible_gpus = visible_gpus
        else:
            if (visible_gpus is not None) and (gpu_id not in visible_gpus):
                raise ValueError(
                    f"Container {name} was supposed to use GPU {gpu_id},"
                    " but no GPU with that ID is visible!"
                    f" Visible GPUs: {visible_gpus} ('None' indicates all GPUs are visible.)")
            self._visible_gpus = {gpu_id}

        printer.set_header(f"CONTAINER-{name}")

        # Source of the workspace directory on the host.
        self._host_workspace_path = host_workspace_path
        # Local path to the workspace directory (outside of the Docker container).
        self._workspace_bind_source = self._host_workspace_path
        if not self._host_workspace_path.exists():
            raise ValueError(
                f"Host workspace path {self._host_workspace_path} does not exist!"
            )

        # Use the workspace directory as the home directory for the agent
        # within the Docker container.
        self._local_agent_home = self._host_workspace_path
        self._local_agent_home.mkdir(parents=True, exist_ok=True)
        printer.log(f"Docker container {self._name} will use $HOME directory: {self._local_agent_home}")

        self._ensure_image_ready()
        self.start_container()
        # TODO(!): automatically stop the container when this object is garbage collected

    def stop(self):
        # TODO: ensure this is called upon destruction
        self.stop_container()

    @property
    def name(self) -> str:
        return self._name

    def nvidia_runtime_available(self) -> bool:
        """Check if NVIDIA Docker runtime is available."""
        try:
            result = subprocess.check_output(
                ["docker", "info", "--format", "{{.Runtimes}}"],
                timeout=10,
            )
            return "nvidia" in result.decode("utf-8").lower()
        except Exception:
            return False

    def exec(self, argv: list[str], timeout: int | None = None) -> tuple[bool, str]:
        """Execute a command in the container without invoking a shell.

        Args:
            argv: Command and arguments (e.g. ``["python", "eval.py", "draft.py"]``).
            timeout: The timeout in seconds (default: DEFAULT_EXEC_TIMEOUT).

        Returns:
            A tuple containing a boolean indicating success and the output of the command.
        """
        if not argv:
            return False, "Error: empty command"
        if timeout is None:
            timeout = self.DEFAULT_EXEC_TIMEOUT

        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    str(self.workspace_root),
                    self._name,
                    *argv,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr

            printer.debug(
                f"Executed command:\n"
                "```\n"
                f"{' '.join(argv)}\n"
                "```\n"
                f"Result (return code: {result.returncode}, success: {success}):\n"
                "```\n"
                f"{output}\n"
                "```\n"
            )

            return success, output
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"

    def _pip_install_argv(self, package: str) -> list[str]:
        """Build ``docker run --rm`` argv for a networked pip install.

        The long-lived workspace container runs with ``--network=none``; Docker
        cannot attach a second network to it.  Instead, run pip in a one-off
        container with the same volume mounts so packages land in ``$HOME/.local``.
        """
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            self.BRIDGE_NETWORK,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "USER=agent",
            "-e",
            "LOGNAME=agent",
            "-v",
            f"{self._workspace_bind_source.resolve()}:{self.workspace_root}:rw",
            "-v",
            f"{self._local_agent_home.resolve()}:{AGENT_HOME}:rw",
            "-e",
            f"HOME={AGENT_HOME}",
            "-e",
            "PYTHONPYCACHEPREFIX=/tmp/__pycache__",
            "--security-opt=no-new-privileges",
            # Mirror the workspace container's resource caps so a malicious or
            # broken package build cannot exhaust host memory or fork-bomb it.
            "--memory",
            self.cfg.memory_limit,
            "--pids-limit",
            str(self.cfg.pids_limit),
            self.cfg.docker_image,
            "pip",
            "install",
            # Never build from source: wheel-only installs don't execute
            # setup.py / PEP-517 build hooks, closing the arbitrary-code-execution
            # path through this network-enabled container.
            "--only-binary=:all:",
            "--user",
            # Terminate pip option parsing so the (validated) package specifier
            # can never be interpreted as a flag (e.g. -r, -e, --index-url).
            "--",
            package,
        ]

    def install_package(self, package: str) -> str:
        """Installs a package using pip in the workspace docker container.
        If the package installation fails, this tools return the error message so you can fix it.

        Network access is enabled only for the duration of the install (via a
        one-off ``docker run`` on the bridge network); agent code runs with
        ``--network=none``.

        Args:
            package: The package to install.

        Returns:
            The result of the package installation.

        Examples:
            install("numpy")  # Installs the numpy package
            install("pandas")  # Installs the pandas package
        """
        if not self.PACKAGE_SPEC_RE.match(package):
            return (
                f"Error installing package: invalid package specifier '{package}'. "
                "Provide a plain PyPI package name with optional extras and version "
                "constraints (e.g. 'numpy', 'pandas>=2.0', 'requests[socks]'). "
                "Paths, URLs, VCS specifiers, and pip flags are not allowed."
            )
        try:
            result = subprocess.run(
                self._pip_install_argv(package),
                check=False,
                capture_output=True,
                text=True,
                timeout=self.DEFAULT_PIP_TIMEOUT,
            )
            if result.returncode != 0:
                err = (result.stderr or result.stdout).strip()
                return f"Error installing package: {err}"

            printer.debug(f"Installed package {package} in container '{self._name}'")

            return f"Package {package} installed successfully"
        except subprocess.TimeoutExpired:
            return f"Error installing package: installation timed out after {self.DEFAULT_PIP_TIMEOUT} seconds"
        except Exception as e:
            return f"Error installing package: {str(e)}"

    def run_python_code(self, code: str, timeout: int | None = None) -> str:
        """Runs Python code in the workspace docker container.

        If the code execution is successful, returns the output of the code execution.
        If the code execution fails, returns the error message.

        Args:
            code: The Python code to run.
            timeout: The timeout in seconds (default: DEFAULT_SCRIPT_TIMEOUT).

        Returns:
            The result of the code execution.

        Examples:
            run_python_code("print('Hello, world!')")  # Runs the code and returns the output
        """
        return self._run_python_in(code, self.workspace_root, timeout)

    def run_python_code_in_home(self, code: str, timeout: int | None = None) -> str:
        """Runs Python code in the container with cwd set to the agent home.

        Intended for internal callers (e.g. doc-introspection tools) that don't
        need access to the workspace.

        The agent's home directory is the workspace directory itself, so this is
        functionally equivalent to running from the workspace mount.
        """
        return self._run_python_in(code, AGENT_HOME, timeout)

    def _run_python_in(
        self,
        code: str,
        workdir: str | Path,
        timeout: int | None,
    ) -> str:
        if timeout is None:
            timeout = self.DEFAULT_SCRIPT_TIMEOUT
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    str(workdir),
                    self._name,
                    "python",
                    "-c",
                    code,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                return f"Error running Python code:\n{result.stderr}"

            printer.debug(
                f"Ran Python code:\n"
                "```\n"
                f"{code}\n"
                "```\n"
                "Result:\n"
                "```\n"
                f"{result.stdout}\n"
                "```\n"
            )

            return result.stdout

        except subprocess.TimeoutExpired:
            return f"Error running Python code: execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Error running Python code: {str(e)}"

    def run_python_script(self, script_path: str, timeout: int | None = None) -> str:
        """Runs a Python script from the workspace in the workspace docker container.

        If the script execution is successful, returns the output of the script execution.
        If the script execution fails, returns the error message.

        Args:
            script_path: The path to the Python script to run (relative to workspace).
            timeout: The timeout in seconds (default: DEFAULT_SCRIPT_TIMEOUT).
        Returns:
            The result of the script execution, or the error message if it fails.
        """
        if timeout is None:
            timeout = self.DEFAULT_SCRIPT_TIMEOUT

        try:
            script_full_path = self.workspace_root / script_path
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    str(self.workspace_root),
                    self._name,
                    "python",
                    str(script_full_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                return f"Error running Python script:\n{result.stderr}"

            printer.debug(
                f"Ran Python script '{script_path}' with result:\n"
                "```\n"
                f"{result.stdout}\n"
                "```\n"
            )
            return result.stdout

        except subprocess.TimeoutExpired:
            return f"Error running Python script: execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Error running Python script: {str(e)}"


    # ---------- File operations -----------

    def read_file(self, path: str) -> str:
        """Read the contents of a file.

        Args:
            path: Path to the file, relative to the workspace directory.

        Returns:
            The file contents, or an error message if reading fails.

        Examples:
            read_file("main.py")
            read_file("src/utils.py")
        """
        try:
            return read_text_under_root(self._workspace_bind_source, path)

        except FileNotFoundError:
            return f"Error reading file: file '{path}' does not exist in workspace"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file. Creates parent directories if needed.

        Args:
            path: Path to the file, relative to the workspace directory.
            content: The content to write to the file.

        Returns:
            Success message or error message if writing fails.

        Examples:
            write_file("main.py", "print('Hello, World!')")
        """
        return self._write_file_generic(path, content, binary=False)

    def write_file_binary(self, path: str, content: bytes) -> str:
        """Write binary content to a file. Creates parent directories if needed.

        Args:
            path: Path to the file, relative to the workspace directory.
            content: The binary content to write to the file.

        Returns:
            Success message or error message if writing fails.

        Examples:
            write_file_binary("data.bin", binary_bytes)
        """
        return self._write_file_generic(path, content, binary=True)

    def _write_file_generic(self, path: str, content: str | bytes, binary: bool) -> str:
        try:
            write_file_under_root(self._workspace_bind_source, path, content)
            return f"Successfully wrote to {path}"

        except Exception as e:
            return f"Error writing file: {str(e)}"


    def list_dir(self, path: str = ".") -> str:
        """List contents of a directory.

        Args:
            path: Path to the directory, relative to workspace. Defaults to workspace root.

        Returns:
            List of files and directories, or error message.

        Examples:
            list_dir()
            list_dir("src")
        """
        try:
            return os.linesep.join(
                listdir_under_root(self._workspace_bind_source, path)
            )

        except FileNotFoundError:
            return f"Error listing directory: {path} is not a directory"
        except NotADirectoryError:
            return f"Error listing directory: {path} is not a directory"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def create_dir(self, path: str) -> str:
        """Create a directory (and parent directories if needed).

        Args:
            path: Path to the directory, relative to workspace.

        Returns:
            Success message or error message.

        Examples:
            create_dir("src")
            create_dir("src/utils")
        """
        try:
            mkdir_under_root(self._workspace_bind_source, path)
            return f"Successfully created directory {path}"
        except Exception as e:
            return f"Error creating directory: {str(e)}"

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        """Edit a file by replacing a specific string with a new string.

        PREFERRED over write_file when modifying existing files — you only specify the
        changed portion instead of regenerating the entire file content, which is
        significantly more efficient.
        The old_string must appear exactly once in the file to avoid ambiguous edits.

        Args:
            path: Path to the file, relative to the workspace directory.
            old_string: The exact text to find in the file (must be unique).
            new_string: The text to replace it with.

        Returns:
            Success message or error message if the edit fails.

        Examples:
            edit_file("main.py", "x = 1", "x = 2")
            edit_file("src/model.py", "learning_rate = 0.01", "learning_rate = 0.001")
        """
        try:
            content = self.read_file(path)
            if content.startswith("Error"):
                return content

            count = content.count(old_string)
            if count == 0:
                return (
                    f"Error editing file: old_string not found in {path}. "
                    "Make sure the string matches exactly, including whitespace and indentation."
                )
            if count > 1:
                return (
                    f"Error editing file: old_string appears {count} times in {path}. "
                    "Include more surrounding context to make the match unique."
                )

            new_content = content.replace(old_string, new_string, 1)
            return self.write_file(path, new_content)

        except Exception as e:
            return f"Error editing file: {str(e)}"

    def copy_file(self, source: str, destination: str) -> str:
        """Copy a file within the workspace.

        Args:
            source: Source file path, relative to the workspace directory.
            destination: Destination file path, relative to the workspace directory.

        Returns:
            Success message or error message if copying fails.

        Examples:
            copy_file("draft.py", "solution.py")
            copy_file("config.json", "config_backup.json")
        """
        try:
            copy_file_under_root(
                self._workspace_bind_source,
                source,
                destination,
            )
            return f"Successfully copied {source} to {destination}"

        except FileNotFoundError:
            return f"Error copying file: file '{source}' does not exist in workspace"
        except IsADirectoryError:
            return f"Error copying file: file '{source}' does not exist in workspace"
        except Exception as e:
            return f"Error copying file: {str(e)}"

    def delete_file_or_dir(self, path: str) -> str:
        """Delete a file or directory (recursively).

        Args:
            path: Path to the file or directory, relative to workspace.

        Returns:
            Success message or error message.

        Examples:
            delete("temp.py")
            delete("old_src")
        """
        try:
            delete_under_root(self._workspace_bind_source, path)
            return f"Successfully deleted {path}"

        except FileNotFoundError:
            return f"Error deleting: file or directory '{path}' does not exist in workspace"
        except Exception as e:
            return f"Error deleting: {str(e)}"


    # ---------- Container management -----------

    def start_container(self):
        """Start the Docker container for the workspace."""

        # Remove any accidentally leftover container with the same name. Bounded
        # by a timeout so a wedged predecessor cannot hang the new agent at
        # startup; if it does, the docker run below will fail with a clear
        # "name already in use" error which is a much better failure mode than
        # an indefinite hang.
        try:
            subprocess.run(
                ["docker", "rm", "-f", self._name],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.DOCKER_RM_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            printer.warning(
                f"Pre-start cleanup of stale container '{self._name}' timed out after "
                f"{self.DOCKER_RM_TIMEOUT}s; will let `docker run` surface the conflict."
            )

        # Build docker run command
        cmd = ["docker", "run", "-d"]

        # Add GPU support if enabled and available
        visible_gpus = "none"
        if self._use_gpu:
            visible_gpus = \
                "all" \
                if (self._visible_gpus is None) \
                else ",".join(str(g) for g in self._visible_gpus)
            cmd.extend(
                [
                    "--runtime=nvidia",
                    "-e",
                    f"NVIDIA_VISIBLE_DEVICES={visible_gpus}"
                ]
            )

        cmd.extend(
            [
                "--name",
                self._name,
                # Run as current user to avoid permission issues
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                # Provide USER/LOGNAME so getpass.getuser() and similar tools
                # don't fall through to pwd.getpwuid(), which fails because the
                # --user UID is not present in the image's /etc/passwd.
                "-e",
                "USER=agent",
                "-e",
                "LOGNAME=agent",
                # Mount the workspace directory
                "-v",
                f"{self._workspace_bind_source.resolve()}:{self.workspace_root}:rw",
                # Mount and set home directory (shared among agents working on that machine)
                "-v",
                f"{self._local_agent_home.resolve()}:{AGENT_HOME}:rw",
                "-e",
                f"HOME={AGENT_HOME}",
                # Keep Python bytecode cache out of the workspace directory.
                "-e",
                "PYTHONPYCACHEPREFIX=/tmp/__pycache__",
                #
                "--security-opt=no-new-privileges",
                # No outbound network during agent code execution; pip install
                # uses a one-off container on the bridge network.
                "--network",
                "none",
                # Set memory limit
                "--memory",
                self.cfg.memory_limit,
                # Set pids limit
                "--pids-limit",
                str(self.cfg.pids_limit),
                self.cfg.docker_image,
                "sleep",
                "infinity",
            ]
        )

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start container '{self._name}': {result.stderr.strip()}"
            )
        printer.info(f"Started container '{self._name}' with GPUs {visible_gpus}")

    def stop_container(self):
        """Stop and remove the Docker container for the workspace.

        Robust to wedged containers: every Docker call is bounded by a timeout
        and known non-fatal error strings ('No such container',
        'did not receive an exit event') are downgraded to warnings. A single
        stuck container must never block the orchestrator for hours.
        """
        # 1) SIGKILL up front. Bypasses the 10s SIGTERM grace period of
        #    `docker stop` and is fast for healthy containers. Failures (e.g.
        #    container already gone) are not fatal here.
        try:
            subprocess.run(
                ["docker", "kill", "--signal=KILL", self._name],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.DOCKER_KILL_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            printer.warning(
                f"docker kill timed out for '{self._name}' after "
                f"{self.DOCKER_KILL_TIMEOUT}s; will still attempt to remove it."
            )

        # 2) Remove with a bounded wait. If `docker rm` itself hangs (daemon
        #    stuck waiting for an exit event that will never arrive), give up
        #    and let the daemon reap the leftover asynchronously.
        try:
            result = subprocess.run(
                ["docker", "rm", "-f", self._name],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.DOCKER_RM_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            printer.warning(
                f"docker rm timed out for '{self._name}' after "
                f"{self.DOCKER_RM_TIMEOUT}s; daemon will reap it asynchronously."
            )
            return

        if result.returncode == 0:
            printer.info(f"Stopped container '{self._name}'")
            return

        msg = result.stderr.strip()
        if "No such container" in msg or "did not receive an exit event" in msg:
            printer.warning(
                f"Container '{self._name}' could not be cleanly removed: {msg}"
            )
            return
        raise RuntimeError(f"Failed to stop container '{self._name}': {msg}")

    def _ensure_image_ready(self):
        """Ensure that the Docker image is available locally, building it from
        the Dockerfile if necessary.
        """
        if self.cfg.dockerfile_path is None:
            return

        output = subprocess.run(
            ["docker", "images", "-q", self.cfg.docker_image],
            check=False,
            capture_output=True,
            text=True,
        )
        if output.stdout.strip():
            return

        printer.info(
            f"Building Docker image {self.cfg.docker_image} from Dockerfile {self.cfg.dockerfile_path}"
        )
        build = subprocess.run(
            [
                "docker",
                "build",
                "-t",
                self.cfg.docker_image,
                "-f",
                self.cfg.dockerfile_path,
                ".",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            raise RuntimeError(
                f"Failed to build Docker image {self.cfg.docker_image}.\n"
                f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
            )
