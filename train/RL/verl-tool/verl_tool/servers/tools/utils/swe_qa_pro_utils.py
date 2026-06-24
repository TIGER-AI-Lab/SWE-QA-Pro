import os
import sys
import ast
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

BLOCKED_BASH_COMMANDS = [
    "git", "ipython", "jupyter", "nohup",
    "python", "python3", "pip", "pip3",
    "npm", "node", "yarn", "make", "cmake",
    "docker", "sudo", "su", "chmod", "chown",
    "rm", "rmdir", "mv", "cp", "mkdir", "touch",
    "echo", "printf", "tee", "dd",
    "vi", "vim", "nano", "emacs",
    "wget", "curl", "scp", "rsync",
    "kill", "killall", "pkill",
    "systemctl", "service",
    "export", "unset", "alias", "unalias","cd"
]

ALLOWED_BASH_COMMANDS = [
    "ls", "tree", "find",
    "basename", "dirname", "realpath", "pwd",
    "cat", "head", "tail", "less", "more",
    "grep", "egrep", "fgrep", "rg", "ag",
    "wc", "sort", "uniq", "cut", "awk", "sed",
    "file", "stat", "du", "df",
]

MAX_RESPONSE_LEN = 6000

TRUNCATED_MESSAGE = (
    "\n<response clipped>\n"
    "<NOTE>Only part of the content is shown to limit context size. "
    "Use search tools (e.g., grep) and view_range to inspect specific sections.</NOTE>"
)

def run_command(cmd, cwd: Optional[Path] = None):
    try:
        return subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
        )
    except TypeError:
        # Compatibility for Python 3.5 / 3.6
        return subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )


def is_command_allowed(command: str):
    command = command.strip()
    if not command:
        return False, "Empty command"

    first_token = command.split()[0]

    # Check for dangerous redirections or write-related operations
    dangerous_patterns = [
        ">", ">>", "tee", "xargs", "exec"
    ]
    for pattern in dangerous_patterns:
        if pattern in command:
            return False, f"Command contains a forbidden operator or command: {pattern}"

    # Check if the command is explicitly blocked
    if first_token in BLOCKED_BASH_COMMANDS:
        return False, f"Command '{first_token}' is forbidden in read-only mode"

    # Check if the command is explicitly allowed
    if first_token in ALLOWED_BASH_COMMANDS:
        return True, ""

    # Deny all other commands by default
    return False, (
        f"Command '{first_token}' is not in the allowlist. "
        "Only read-only inspection commands are permitted."
    )

def execute_readonly_command(command: str, cwd: Optional[Path] = None) -> str:
    """
    Validate and execute a read-only bash command.
    Args:
        command (str): The bash command to execute (read-only).
        The command MUST start with one of the following allowed commands:
        ls, tree, find, basename, dirname, realpath, pwd,
        cat, head, tail, less, more,
        grep, egrep, fgrep, rg, ag,
        wc, sort, uniq, cut, awk, sed,
        file, stat, du, df.
    Returns:
        str: formatted command output or error message
    """
    allowed, error_msg = is_command_allowed(command)
    if not allowed:
        return (
            f"ERROR: {error_msg}\n"
            "In read-only mode, only inspection commands are allowed, "
            "such as: ls, cat, grep, find, tree, etc."
        )

    result = run_command(command, cwd=cwd)

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    if result.returncode != 0:
        return (
            "ERROR: Command execution failed\n"
            f"STDOUT:\n{stdout if stdout else '(no output)'}\n\n"
            f"STDERR:\n{stderr if stderr else '(no error message)'}"
        )

    if stderr:
        return (
            f"STDOUT:\n{stdout if stdout else '(no output)'}\n\n"
            f"STDERR:\n{stderr}"
        )

    return f"STDOUT:\n{stdout if stdout else '(no output)'}"

def truncate_text(text: str, limit: int = MAX_RESPONSE_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATED_MESSAGE


class CodebaseViewer:
    """
    Read-only viewer for inspecting files and directory structure.
    Designed for agent usage. No write operations are supported.
    """

    def view(
        self,
        path: Path,
        *,
        view_range: Optional[List[int]] = None,
        concise: bool = False,
        python_only: bool = False,
    ) -> str:
        if not path.exists():
            return f"ERROR: Path does not exist: {path}"

        if path.is_dir():
            return self._view_directory(path, python_only)
        else:
            return self._view_file(path, view_range, concise, python_only)

    def _view_directory(self, path: Path, python_only: bool) -> str:
        if python_only:
            cmd = [
                "find", str(path), "-maxdepth", "2",
                "-not", "-path", "*/.*",
                "(", "-type", "d", "-o", "-name", "*.py", ")"
            ]
        else:
            cmd = [
                "find", str(path), "-maxdepth", "2",
                "-not", "-path", "*/.*"
            ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stderr.strip():
            return f"ERROR: {proc.stderr.strip()}"

        output = (
            f"Directory listing for {path} (depth ≤ 2, hidden paths excluded):\n"
            f"{proc.stdout}"
        )
        return truncate_text(output)

    def _view_file(
        self,
        path: Path,
        view_range: Optional[List[int]],
        concise: bool,
        python_only: bool,
    ) -> str:
        if python_only and path.suffix != ".py":
            return (
                f"ERROR: python_only=True but file is not a Python file: {path.name}"
            )

        text = self._read_file(path)

        if concise and path.suffix == ".py":
            lines = self._python_outline(text, path)
        else:
            lines = list(enumerate(text.splitlines(), start=1))

        if view_range:
            start, end = view_range
            if start < 1 or (end != -1 and end < start):
                return f"ERROR: Invalid view_range: {view_range}"
            lines = [
                (i, line)
                for (i, line) in lines
                if i >= start and (end == -1 or i <= end)
            ]

        output_lines = [f"{i:6d} {line}" for (i, line) in lines]
        output = f"File view: {path}\n" + "\n".join(output_lines)
        return truncate_text(output)

    def _read_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")

    def _python_outline(
        self, file_text: str, path: Path
    ) -> List[Tuple[int, str]]:
        try:
            tree = ast.parse(file_text)
        except SyntaxError as e:
            return [(1, f"ERROR: Failed to parse Python file: {e}")]

        elide_ranges = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                start = node.body[0].lineno
                end = getattr(node.body[-1], "end_lineno", start)
                if end - start >= 3:
                    elide_ranges.append((start, end))

        lines = file_text.splitlines()
        elided = set(
            line for (s, e) in elide_ranges for line in range(s, e + 1)
        )

        result = []
        for i, line in enumerate(lines, start=1):
            if i in elided:
                continue
            result.append((i, line))

        for (s, e) in elide_ranges:
            result.append((s, f"... elided lines {s}-{e} ..."))

        result.sort(key=lambda x: x[0])
        return result

def view_codebase(
    path: str,
    view_range: Optional[List[int]] = None,
    concise: bool = False,
    python_only: bool = False,
) -> str:
    """
    Agent tool: view files or directories in a read-only manner.

    Args:
        path: File or directory path.
        view_range: Optional [start, end] line range (1-based, end=-1 allowed). If omitted, the entire file will be read.
        concise: If True, return a structural outline for Python files.
        python_only: If True, restrict viewing to .py files only.

    Returns:
        A formatted string suitable for agent consumption.
    """
    viewer = CodebaseViewer()
    return viewer.view(
        Path(path),
        view_range=view_range,
        concise=concise,
        python_only=python_only,
    )

def semantic_search(
    term: str,
    path: str = ".",
    *,
    python_only: bool = False,
    max_files: int = 100,
    include_lines: bool = False,
) -> Dict:
    """
    High-level semantic search over the codebase. Use this tool FIRST to identify relevant files.\n\nThis tool does NOT return full file contents.

    Args:
        term (str):
            The search term (plain substring match).

        path (str, default="."):
            Path to a directory or a single file.

        python_only (bool, default=False):
            If True, only search files ending with `.py`.

        max_files (int, default=100):
            Maximum number of files allowed to match.
            If exceeded, the search stops early.

        include_lines (bool, default=False):
            If True, include matching line numbers and content
            (intended for single-file or small searches).

    Returns:
        dict with keys:
            - success (bool):
                Whether the search completed successfully.
            - scope (str):
                "file" or "directory".
            - term (str):
                The search term.
            - root (str):
                Absolute path of the search root.
            - files (list):
                List of matched files. Each entry contains:
                    - path (str): relative path
                    - count (int): number of matching lines
                    - matches (optional): list of {line, text}
            - truncated (bool):
                Whether results were truncated due to max_files.
            - error (str | None):
                Error message if success == False.
    """
    root = os.path.realpath(path)

    if not os.path.exists(root):
        return {
            "success": False,
            "error": f"Path does not exist: {root}",
        }

    results: List[Dict] = []
    truncated = False

    def should_skip_file(filename: str) -> bool:
        if filename.startswith("."):
            return True
        if python_only and not filename.endswith(".py"):
            return True
        return False

    # Single file search
    if os.path.isfile(root):
        try:
            matches = []
            count = 0
            with open(root, "r", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    if term in line:
                        count += 1
                        if include_lines:
                            matches.append(
                                {
                                    "line": lineno,
                                    "text": line.rstrip(),
                                }
                            )

            if count > 0:
                results.append(
                    {
                        "path": os.path.relpath(root, os.getcwd()),
                        "count": count,
                        "matches": matches if include_lines else None,
                    }
                )

            return {
                "success": True,
                "scope": "file",
                "term": term,
                "root": root,
                "files": results,
                "truncated": False,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # Directory search
    for current_root, dirs, files in os.walk(root):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for filename in files:
            if should_skip_file(filename):
                continue

            filepath = os.path.join(current_root, filename)

            try:
                count = 0
                with open(filepath, "r", errors="ignore") as f:
                    for line in f:
                        if term in line:
                            count += 1

                if count > 0:
                    results.append(
                        {
                            "path": os.path.relpath(filepath, os.getcwd()),
                            "count": count,
                        }
                    )

                    if len(results) >= max_files:
                        truncated = True
                        break

            except (UnicodeDecodeError, PermissionError):
                continue

        if truncated:
            break

    return {
        "success": True,
        "scope": "directory",
        "term": term,
        "root": root,
        "files": results,
        "truncated": truncated,
        "error": None,
    }