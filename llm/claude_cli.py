"""Call Claude through the Claude Code CLI in headless mode, with a clean context.

The CLI is used (rather than the API SDK) because it is how the experiments were
authorised to run: on the author's Claude plan, with no API key on the machine.
Each call runs in an empty working directory with user settings, plugins, MCP
servers, skills and built-in tools disabled, and a replaced system prompt, so the
model sees only the experiment's prompt. Wall-clock latency therefore includes
CLI start-up, which a deployed CLI-based agent also pays; the API-side duration
reported by the CLI is recorded separately.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

NEUTRAL_CWD = Path(tempfile.gettempdir()) / "stability-lab-llm-cwd"
STRIP_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_HOST_SESSION_ID",
             "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
             "CLAUDE_PID", "CLAUDE_AGENT_SDK_VERSION")


def call(system: str, prompt: str, schema: dict, model: str, effort: str | None = None,
         timeout: float = 180.0) -> dict:
    NEUTRAL_CWD.mkdir(parents=True, exist_ok=True)
    cmd = ["claude", "-p", prompt,
           "--model", model,
           "--system-prompt", system,
           "--tools", "",
           "--output-format", "json",
           "--no-session-persistence",
           "--setting-sources", "project,local",
           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--disable-slash-commands",
           "--json-schema", json.dumps(schema)]
    if effort:
        cmd += ["--effort", effort]
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=NEUTRAL_CWD, env=env, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "wall_s": time.time() - t0, "t_start": t0}
    wall = time.time() - t0
    out = {"ok": False, "wall_s": wall, "t_start": t0, "model": model, "effort": effort}
    try:
        res = json.loads(proc.stdout.decode())
    except json.JSONDecodeError:
        out["error"] = "bad-json"
        out["stderr"] = proc.stderr.decode()[-2000:]
        out["stdout"] = proc.stdout.decode()[-2000:]
        return out
    out["api_ms"] = res.get("duration_api_ms")
    out["duration_ms"] = res.get("duration_ms")
    out["usage"] = res.get("usage")
    out["cost_usd"] = res.get("total_cost_usd")
    out["model_usage"] = res.get("modelUsage")
    if res.get("is_error"):
        out["error"] = res.get("result")
        out["api_error_status"] = res.get("api_error_status")
        return out
    structured = res.get("structured_output")
    if structured is None:
        try:
            structured = json.loads(res.get("result") or "")
        except (json.JSONDecodeError, TypeError):
            structured = None
    out["structured"] = structured
    out["raw_result"] = res.get("result")
    out["ok"] = structured is not None
    return out


def auth_ok(model: str = "claude-haiku-4-5") -> tuple[bool, str]:
    r = call("Reply with the requested JSON only.", "Return ok=true.",
             {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
              "additionalProperties": False}, model=model, timeout=90)
    return bool(r.get("ok")), str(r.get("error") or "")
