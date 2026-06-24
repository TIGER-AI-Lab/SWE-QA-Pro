from .base import BaseTool, register_tool
import json
import os
import re
import asyncio
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Any
from .utils.swe_qa_pro_utils import view_codebase, semantic_search, execute_readonly_command
# Timeout for command execution in seconds
TIMEOUT = 10

def dump_obs(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

def run(cmd, cwd: Path | None = None):
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {cmd}\n"
            f"cwd={cwd}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )
    return p.stdout


def safe_rmtree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def clone_and_checkout(repo_name: str, commit_id: str, repo_base_dir: Path) -> Path:
    repo_base_dir = Path(repo_base_dir).resolve()
    repo_base_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = repo_base_dir / repo_name.split("/")[-1]

    repo_url = f"https://github.com/{repo_name}.git"

    if not repo_dir.exists():
        run(["git", "clone", repo_url, repo_dir.name], cwd=repo_base_dir)
        run(["git", "checkout", "--force", commit_id], cwd=repo_dir)

    return repo_dir

@register_tool
class SWEQAProTool(BaseTool):
    tool_type = "swe_qa_pro"
    timeout = TIMEOUT
    stop_tokens = ["</tool_call>"]
    valid_func_names = ["view_codebase","semantic_search", "execute_readonly_command"]
    
    def __init__(self, num_workers=1, repo_root: str = "repos"):
        super().__init__(num_workers)
        self.repo_root = Path(repo_root).resolve()
        
    def get_usage_inst(self):
        return (
            "You have access to read-only tools for inspecting a Git repository:\n"
            "- Use semantic_search to find relevant files.\n"
            "- Use view_codebase to read code.\n"
            "- Use execute_readonly_command for safe inspection commands."
        )
    
    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        try:
            import regex as re
            m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", action, re.DOTALL)
            if not m:
                return {}, False
            call = json.loads(m.group(1))
            if call.get("name") not in self.valid_func_names:
                return {}, False
            if not isinstance(call.get("arguments", {}), dict):
                return {}, False
            return call, True
        except Exception:
            return {}, False
        
    def load_env(self, trajectory_id: str) -> Dict[str, Any]:
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "repo_dir": None,
                "repo_name": None,
                "commit_id": None,
                "metadata": {"turns": 0},
                "previous_obs": [],
            }
        return env
    
    def _ensure_repo(self, env: Dict[str, Any], extra_field: Dict[str, Any]):
        if env["repo_dir"] is not None:
            return

        repo_name = extra_field.get("repo_name")
        commit_id = extra_field.get("commit_id")

        if not repo_name or not commit_id:
            raise ValueError("repo_name and commit_id must be provided in extra_fields")

        repo_dir = clone_and_checkout(repo_name, commit_id, self.repo_root)

        env["repo_name"] = repo_name
        env["commit_id"] = commit_id
        env["repo_dir"] = repo_dir
    
    def conduct_action(self, trajectory_id, action, extra_field):
        parsed, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        print(f"env:{env}")
        print(f"is_valid:{is_valid}")
        if not is_valid:
            observation = {
                "error": "Invalid tool_call format or function name"
            }
            done = False
            valid = False
        else:
            try:
                self._ensure_repo(env, extra_field)
                repo_dir: Path = env["repo_dir"]

                name = parsed["name"]
                args = parsed.get("arguments", {})

                if name == "view_codebase":
                    raw = view_codebase(
                        path=args.get("path", env["repo_dir"]),
                        view_range=args.get("view_range"),
                        concise=args.get("concise", False),
                        python_only=args.get("python_only", False),
                    )
                    observation = {"obs": dump_obs({"view_content": raw})}

                elif name == "semantic_search":
                    raw = semantic_search(
                        term=args["term"],
                        path=args.get("path", env["repo_dir"]),
                        python_only=args.get("python_only", False),
                        max_files=args.get("max_files", 100),
                        include_lines=args.get("include_lines", False),
                    )
                    observation = {"obs": dump_obs(raw)}

                elif name == "execute_readonly_command":
                    raw = execute_readonly_command(
                        command=args["command"],
                        cwd=os.getcwd()
                    )
                    observation = {"obs": dump_obs({"command_output": raw})}

                else:
                    raise RuntimeError(f"Unknown tool name: {name}")

                done = False
                valid = True

            except Exception as e:
                name = parsed["name"]
                args = parsed.get("arguments", {})
                observation = {
                    "obs": dump_obs({
                        "error": str(e),
                        "tool": name,
                        "args": args
                    })
                }
                done = False
                valid = False

        self.update_env(
            trajectory_id,
            env,
            parsed,
            is_valid,
            extra_field,
            observation,
        )
        self.save_env(trajectory_id, env)

        if extra_field.get("is_last_step"):
            self.delete_env(trajectory_id)

        return observation, done, valid
