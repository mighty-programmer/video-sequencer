from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_PORT = 8000
STATE_ROOT = PROJECT_ROOT / "webapp" / "state"
SERVER_STATE_FILE = STATE_ROOT / "server_state.json"
SERVER_LOG_FILE = STATE_ROOT / "server.log"

STATE_ROOT.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_project_root() -> Path:
    """Prefer the caller's current worktree path over any resolved symlink path."""
    return Path(os.environ.get("PWD", str(PROJECT_ROOT)))


def get_server_log_path(project_root: Optional[Path] = None) -> Path:
    override = os.environ.get("VIDEO_SEQUENCER_SERVER_LOG")
    if override:
        return Path(override)
    base = project_root or get_project_root()
    return base / "webapp" / "state" / "server.log"


def get_server_state_path(project_root: Optional[Path] = None) -> Path:
    base = project_root or get_project_root()
    return base / "webapp" / "state" / "server_state.json"


def server_command(python_executable: Optional[str] = None) -> List[str]:
    return [python_executable or sys.executable, "src/webapp.py"]


def restart_command(project_root: Optional[Path] = None, python_executable: Optional[str] = None) -> str:
    root = project_root or get_project_root()
    command = " ".join(server_command(python_executable))
    return f"cd {root} && {command}"


def process_exists(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_server_state(project_root: Optional[Path] = None) -> Dict[str, Any]:
    path = get_server_state_path(project_root)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def write_server_state(
    *,
    pid: int,
    project_root: Optional[Path] = None,
    python_executable: Optional[str] = None,
    log_file: Optional[Path] = None,
    port: int = SERVER_PORT,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = project_root or get_project_root()
    state_path = get_server_state_path(root)
    payload: Dict[str, Any] = {
        "pid": pid,
        "port": port,
        "python_executable": python_executable or sys.executable,
        "project_root": str(root),
        "log_file": str(log_file or get_server_log_path(root)),
        "command": " ".join(server_command(python_executable or sys.executable)),
        "updated_at": now_iso(),
    }
    if metadata:
        payload.update(metadata)

    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temp_path, state_path)
    return payload


def clear_server_state(project_root: Optional[Path] = None, expected_pid: Optional[int] = None) -> bool:
    state_path = get_server_state_path(project_root)
    if not state_path.exists():
        return False
    if expected_pid is not None:
        state = read_server_state(project_root)
        if state.get("pid") != expected_pid:
            return False
    try:
        state_path.unlink()
    except OSError:
        return False
    return True


def discover_server_processes(python_executable: Optional[str] = None, project_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = project_root or get_project_root()
    python_value = re.escape(python_executable or sys.executable)
    pattern = rf"{python_value} (?:.*/)?src/webapp\.py"
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        capture_output=True,
        text=True,
        cwd=str(root),
        check=False,
    )
    processes: List[Dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        processes.append({"pid": pid, "command": command.strip()})
    return processes


def get_server_status(
    *,
    hostname: str = "localhost",
    python_executable: Optional[str] = None,
    project_root: Optional[Path] = None,
    log_file: Optional[Path] = None,
    port: int = SERVER_PORT,
) -> Dict[str, Any]:
    root = project_root or get_project_root()
    resolved_log = Path(log_file) if log_file else get_server_log_path(root)
    state = read_server_state(root)
    pid = state.get("pid")
    state_source = "state-file" if state else "none"
    processes: List[Dict[str, Any]] = []

    if process_exists(pid):
        processes.append({"pid": pid, "command": state.get("command", "").strip()})
    else:
        if pid:
            clear_server_state(root, expected_pid=pid)
        processes = discover_server_processes(python_executable=python_executable, project_root=root)
        if processes:
            pid = processes[0]["pid"]
            state_source = "process-scan"
            write_server_state(
                pid=pid,
                project_root=root,
                python_executable=python_executable,
                log_file=resolved_log,
                port=port,
                metadata={"command": processes[0]["command"], "discovered_at": now_iso()},
            )
        else:
            pid = None

    return {
        "running": bool(pid),
        "pid": pid,
        "processes": [proc for proc in processes if proc.get("pid")],
        "port": port,
        "hostname": hostname,
        "url": f"http://{hostname}:{port}/",
        "project_root": str(root),
        "log_file": str(resolved_log),
        "python_executable": python_executable or state.get("python_executable") or sys.executable,
        "restart_command": restart_command(root, python_executable or state.get("python_executable") or sys.executable),
        "state_file": str(get_server_state_path(root)),
        "state_source": state_source,
    }
