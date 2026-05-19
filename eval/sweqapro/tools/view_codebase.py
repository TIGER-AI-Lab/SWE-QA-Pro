import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import ast

MAX_RESPONSE_LEN = 6000

TRUNCATED_MESSAGE = (
    "\n<response clipped>\n"
    "<NOTE>Only part of the content is shown to limit context size. "
    "Use search tools (e.g., grep) and view_range to inspect specific sections.</NOTE>"
)


def _truncate(text: str, limit: int = MAX_RESPONSE_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATED_MESSAGE


class _CodebaseViewer:
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
        return self._view_file(path, view_range, concise, python_only)

    def _view_directory(self, path: Path, python_only: bool) -> str:
        if python_only:
            cmd = [
                "find", str(path), "-maxdepth", "2",
                "-not", "-path", "*/.*",
                "(", "-type", "d", "-o", "-name", "*.py", ")",
            ]
        else:
            cmd = ["find", str(path), "-maxdepth", "2", "-not", "-path", "*/.*"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stderr.strip():
            return f"ERROR: {proc.stderr.strip()}"
        output = (
            f"Directory listing for {path} (depth <= 2, hidden paths excluded):\n"
            f"{proc.stdout}"
        )
        return _truncate(output)

    def _view_file(
        self,
        path: Path,
        view_range: Optional[List[int]],
        concise: bool,
        python_only: bool,
    ) -> str:
        if python_only and path.suffix != ".py":
            return f"ERROR: python_only=True but file is not a Python file: {path.name}"
        text = self._read_file(path)
        if concise and path.suffix == ".py":
            lines = self._python_outline(text)
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
        return _truncate(output)

    def _read_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")

    def _python_outline(self, file_text: str) -> List[Tuple[int, str]]:
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
        elided = set(line for (s, e) in elide_ranges for line in range(s, e + 1))

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
    """Agent tool: view files or directories in a read-only manner."""
    return _CodebaseViewer().view(
        Path(path),
        view_range=view_range,
        concise=concise,
        python_only=python_only,
    )
