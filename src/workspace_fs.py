# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Safe host-side filesystem access under a workspace root.

Paths are opened with ``O_NOFOLLOW`` and walked via ``dir_fd`` so bind-mounted
workspace directories cannot escape the root through symlinks or TOCTOU races.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat


def _relative_parts(subpath: str | Path) -> list[str]:
    rel = Path(subpath)
    if rel.is_absolute():
        raise ValueError(
            f"Path '{subpath!r}' must be relative to the workspace directory."
        )
    parts: list[str] = []
    for part in rel.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(
                f"Path '{subpath!r}' must stay inside the workspace directory."
            )
        parts.append(part)
    return parts


def _open_or_create_dir(parent_fd: int, name: str) -> int:
    try:
        return os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        os.mkdir(name, 0o755, dir_fd=parent_fd)
        return os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )


def _open_dir_under_root(
    root: Path,
    parts: list[str],
    *,
    create_intermediate: bool = False,
) -> int:
    root = root.resolve()
    dir_fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts:
            if create_intermediate:
                next_fd = _open_or_create_dir(dir_fd, part)
            else:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=dir_fd,
                )
            os.close(dir_fd)
            dir_fd = next_fd
        return dir_fd
    except Exception:
        os.close(dir_fd)
        raise


def open_file_under_root(
    root: Path,
    subpath: str | Path,
    flags: int,
    mode: int = 0o644,
) -> int:
    """Open a file under *root* using ``O_NOFOLLOW`` (no symlink following)."""
    parts = _relative_parts(subpath)
    if not parts:
        raise ValueError("subpath must name a file inside the workspace")
    root = root.resolve()
    dir_fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
    try:
        create_parents = bool(flags & os.O_CREAT)
        for part in parts[:-1]:
            if create_parents:
                next_fd = _open_or_create_dir(dir_fd, part)
            else:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=dir_fd,
                )
            os.close(dir_fd)
            dir_fd = next_fd
        return os.open(parts[-1], flags, mode, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def file_exists_under_root(root: Path, subpath: str | Path) -> bool:
    """Return True if a regular file exists under *root* (symlinks not followed)."""
    try:
        fd = open_file_under_root(root, subpath, os.O_RDONLY | os.O_NOFOLLOW)
    except (FileNotFoundError, OSError):
        return False
    try:
        return stat.S_ISREG(os.fstat(fd).st_mode)
    finally:
        os.close(fd)


def read_text_under_root(
    root: Path,
    subpath: str | Path,
    *,
    errors: str = "strict",
) -> str:
    fd = open_file_under_root(root, subpath, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8", errors=errors) as handle:
        return handle.read()


def open_journal_under_root(
    root: Path,
    subpath: str | Path = "journal.log",
    mode: str = "a",
):
    """Open a log file under *root* for append/truncate without following symlinks."""
    if mode not in ("a", "w"):
        raise ValueError(f"unsupported journal open mode: {mode!r}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    if mode == "a":
        flags |= os.O_APPEND
    else:
        flags |= os.O_TRUNC
    fd = open_file_under_root(root, subpath, flags)
    return os.fdopen(fd, mode, encoding="utf-8")


def read_bytes_under_root(root: Path, subpath: str | Path) -> bytes:
    fd = open_file_under_root(root, subpath, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def write_file_under_root(
    root: Path,
    subpath: str | Path,
    content: str | bytes,
) -> None:
    binary = isinstance(content, (bytes, bytearray))
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    fd = open_file_under_root(root, subpath, flags)
    with os.fdopen(fd, "wb" if binary else "w", encoding=None if binary else "utf-8") as handle:
        handle.write(content)


def listdir_under_root(root: Path, subpath: str | Path = ".") -> list[str]:
    parts = _relative_parts(subpath)
    root = root.resolve()
    if not parts:
        return sorted(os.listdir(root))
    dir_fd = _open_dir_under_root(root, parts)
    try:
        return sorted(os.listdir(dir_fd))
    finally:
        os.close(dir_fd)


def mkdir_under_root(root: Path, subpath: str | Path) -> None:
    parts = _relative_parts(subpath)
    if not parts:
        return
    dir_fd = _open_dir_under_root(root, parts, create_intermediate=True)
    os.close(dir_fd)


def _rmtree_fd(dir_fd: int) -> None:
    for entry in os.listdir(dir_fd):
        try:
            child_fd = os.open(
                entry,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=dir_fd,
            )
        except OSError:
            os.unlink(entry, dir_fd=dir_fd)
            continue
        try:
            _rmtree_fd(child_fd)
        finally:
            os.close(child_fd)
        os.rmdir(entry, dir_fd=dir_fd)


def delete_under_root(root: Path, subpath: str | Path) -> None:
    parts = _relative_parts(subpath)
    if not parts:
        raise ValueError("cannot delete workspace root")
    root = root.resolve()
    dir_fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=dir_fd,
            )
            os.close(dir_fd)
            dir_fd = next_fd
        name = parts[-1]
        try:
            target_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=dir_fd,
            )
        except OSError:
            os.unlink(name, dir_fd=dir_fd)
            return
        try:
            _rmtree_fd(target_fd)
        finally:
            os.close(target_fd)
        os.rmdir(name, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def atomic_replace_under_root(
    root: Path,
    source: str | Path,
    destination: str | Path,
) -> None:
    """Copy *source* to *destination* atomically without following symlinks."""
    dest_parts = _relative_parts(destination)

    fd = open_file_under_root(root, source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise IsADirectoryError(str(source))
        with os.fdopen(fd, "rb") as handle:
            data = handle.read()
        fd = None
    finally:
        if fd is not None:
            os.close(fd)

    tmp_parts = dest_parts[:-1] + [f".{dest_parts[-1]}.replace.{os.getpid()}.tmp"]
    tmp_subpath = str(Path(*tmp_parts))
    tmp_fd = open_file_under_root(
        root,
        tmp_subpath,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    )
    with os.fdopen(tmp_fd, "wb") as handle:
        handle.write(data)

    root = root.resolve()
    root_fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
    dir_fd = root_fd
    try:
        for part in dest_parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=dir_fd,
            )
            if dir_fd != root_fd:
                os.close(dir_fd)
            dir_fd = next_fd

        dest_name = dest_parts[-1]
        tmp_name = Path(tmp_subpath).name
        try:
            dest_stat = os.lstat(dest_name, dir_fd=dir_fd)
            if stat.S_ISLNK(dest_stat.st_mode):
                raise OSError("destination is a symlink")
        except FileNotFoundError:
            pass

        try:
            os.rename(tmp_name, dest_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except OSError:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except OSError:
                pass
            raise
    finally:
        if dir_fd != root_fd:
            os.close(dir_fd)
        os.close(root_fd)


def copy_file_under_root(
    root: Path,
    source: str | Path,
    destination: str | Path,
) -> None:
    atomic_replace_under_root(root, source, destination)
