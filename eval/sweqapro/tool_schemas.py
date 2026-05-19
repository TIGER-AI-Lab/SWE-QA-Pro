"""OpenAI-style function schemas for the three read-only tools.

Shared between `registry.build_llm` (passed to `.bind_tools`) and any prompt
template that wants to embed the tool list verbatim.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "High-level semantic search over the codebase.\n"
                "Use this tool FIRST to identify relevant files, symbols, or modules related to the question.\n"
                "The tool returns a summarized list of file paths with match counts.\n"
                "Set include_lines=True ONLY when exact line-level matches are required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "The search term (plain substring match)."},
                    "path": {"type": "string", "description": "File or directory path."},
                    "python_only": {"type": "boolean", "description": "If True, restrict to .py files only.", "default": False},
                    "max_files": {"type": "integer", "description": "Maximum number of files allowed to match."},
                    "include_lines": {"type": "boolean", "description": "If True, include matching line numbers and content."},
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_codebase",
            "description": (
                "Read-only viewer for files and directories in the codebase.\n"
                "Use this tool to READ file contents or INSPECT directory structure.\n"
                "For Python files, set concise=True to obtain a structural outline (imports, class/function signatures, major blocks).\n"
                "If the output is truncated or insufficient:\n"
                "1) Use semantic_search or grep to identify relevant line numbers.\n"
                "2) Call this tool again with view_range=[start, end]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File or directory path."},
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Line range [start, end], 1-based. Use -1 as end to read until end of file.",
                    },
                    "concise": {"type": "boolean", "description": "If True, return a structural outline for Python files."},
                    "python_only": {"type": "boolean", "description": "If True, restrict viewing to .py files only."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_readonly_command",
            "description": (
                "Execute read-only shell commands (tree, ls, grep, cat, etc.) for repository inspection. "
                "Must start with an allowed inspection command."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute (must be read-only).",
                    },
                },
                "required": ["command"],
            },
        },
    },
]
