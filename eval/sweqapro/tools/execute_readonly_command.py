import subprocess

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
    "export", "unset", "alias", "unalias", "cd",
]

ALLOWED_BASH_COMMANDS = [
    "ls", "tree", "find",
    "basename", "dirname", "realpath", "pwd",
    "cat", "head", "tail", "less", "more",
    "grep", "egrep", "fgrep", "rg", "ag",
    "wc", "sort", "uniq", "cut", "awk", "sed",
    "file", "stat", "du", "df",
]


def _is_allowed(command: str):
    command = command.strip()
    if not command:
        return False, "Empty command"

    first_token = command.split()[0]

    dangerous_patterns = [">", ">>", "tee", "xargs", "exec"]
    for pattern in dangerous_patterns:
        if pattern in command:
            return False, f"Command contains a forbidden operator or command: {pattern}"

    if first_token in BLOCKED_BASH_COMMANDS:
        return False, f"Command '{first_token}' is forbidden in read-only mode"
    if first_token in ALLOWED_BASH_COMMANDS:
        return True, ""
    return False, (
        f"Command '{first_token}' is not in the allowlist. "
        "Only read-only inspection commands are permitted."
    )


def execute_readonly_command(command: str) -> str:
    """Validate and execute a read-only bash command."""
    allowed, error_msg = _is_allowed(command)
    if not allowed:
        return (
            f"ERROR: {error_msg}\n"
            "In read-only mode, only inspection commands are allowed."
        )

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
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
