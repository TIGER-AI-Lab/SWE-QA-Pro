import os
from typing import Dict, List


def semantic_search(
    term: str,
    path: str = ".",
    *,
    python_only: bool = False,
    max_files: int = 100,
    include_lines: bool = False,
) -> Dict:
    """High-level substring search over a file or directory tree.

    Returns a dict with keys: success, scope, term, root, files, truncated, error.
    """
    root = os.path.realpath(path)

    if not os.path.exists(root):
        return {"success": False, "error": f"Path does not exist: {root}"}

    results: List[Dict] = []
    truncated = False

    def should_skip_file(filename: str) -> bool:
        if filename.startswith("."):
            return True
        if python_only and not filename.endswith(".py"):
            return True
        return False

    if os.path.isfile(root):
        try:
            matches = []
            count = 0
            with open(root, "r", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    if term in line:
                        count += 1
                        if include_lines:
                            matches.append({"line": lineno, "text": line.rstrip()})
            if count > 0:
                results.append({
                    "path": os.path.relpath(root, os.getcwd()),
                    "count": count,
                    "matches": matches if include_lines else None,
                })
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
            return {"success": False, "error": str(e)}

    for current_root, dirs, files in os.walk(root):
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
                    results.append({
                        "path": os.path.relpath(filepath, os.getcwd()),
                        "count": count,
                    })
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
