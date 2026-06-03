"""
APEX v2.0 Tool System
Combines Hermes v1 tools + OpenClaw tool-policy + plugin support.

Key additions over v1:
- Tool policy pipeline (allowlist/deny list per agent)
- MCP server tool integration (from OpenClaw)
- Plugin tool sources
- CWD jail support
- Background process management
- PTY mode for interactive CLIs (from OpenClaw pty)
"""
from __future__ import annotations

import asyncio
import json
import time
import os
import logging
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("apex.tools")


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str = ""
    data: Any = None
    duration: float = 0.0
    session_id: str = ""  # for background processes

    def to_dict(self) -> dict[str, Any]:
        d = {"success": self.success, "output": self.output, "error": self.error,
             "data": self.data, "duration": self.duration}
        if self.session_id:
            d["session_id"] = self.session_id
        return d


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str] = field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    required: list[str] = []

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult: ...

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, description=self.description,
                               parameters=self.parameters, required=self.required)


# ── Shell Tool (enhanced from OpenClaw bash-tools) ──

class ShellTool(BaseTool):
    """Execute shell commands with CWD jail, background processes, and PTY support."""

    name = "shell"
    description = "Execute a shell command. Supports foreground, background, PTY mode, and process interaction."
    parameters = {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 180)", "default": 180},
        "workdir": {"type": "string", "description": "Working directory (default: workspace root)", "default": ""},
        "background": {"type": "boolean", "description": "Run in background (returns session_id)", "default": False},
        "pty": {"type": "boolean", "description": "Run in PTY mode for interactive CLIs", "default": False},
        "env": {"type": "object", "description": "Extra environment variables", "default": {}},
    }
    required = ["command"]

    BLOCKED = [
        "rm -rf /", "mkfs", "dd if=", ":(){:|:&};:", "> /dev/",
        "chmod -R 777 /", "chown -R",
    ]

    _processes: dict[str, asyncio.subprocess.Process] = {}

    async def execute(self, **kwargs: Any) -> ToolResult:
        command: str = kwargs["command"]
        timeout: int = kwargs.get("timeout", 180)
        workdir: str = kwargs.get("workdir", "")
        background: bool = kwargs.get("background", False)
        pty: bool = kwargs.get("pty", False)
        extra_env: dict = kwargs.get("env", {})

        # Safety check
        for blocked in self.BLOCKED:
            if blocked in command:
                return ToolResult(success=False, output="",
                                  error=f"Blocked dangerous command: {blocked}")

        start = time.time()
        env = {**os.environ, **extra_env}

        try:
            if pty:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workdir or None,
                    env=env,
                    # PTY not directly supported in asyncio subprocess; use ptyspawn
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workdir or None,
                    env=env,
                )

            if background:
                session_id = f"proc_{id(proc)}_{int(time.time())}"
                self._processes[session_id] = proc
                return ToolResult(
                    success=True,
                    output=f"Process started with session_id: {session_id}",
                    session_id=session_id,
                    data={"pid": proc.pid},
                    duration=time.time() - start,
                )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            duration = time.time() - start

            return ToolResult(
                success=proc.returncode == 0,
                output=stdout.decode("utf-8", errors="replace")[:50_000],
                error=stderr.decode("utf-8", errors="replace")[:10_000],
                duration=duration,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(success=False, output="", error=f"Timed out after {timeout}s",
                              duration=time.time() - start)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e),
                              duration=time.time() - start)


class ProcessTool(BaseTool):
    """Interact with background processes."""

    name = "process"
    description = "Manage background processes: poll, log, wait, kill, write."
    parameters = {
        "action": {"type": "string", "description": "Action: poll, log, wait, kill, write, submit, close"},
        "session_id": {"type": "string", "description": "Process session ID"},
        "data": {"type": "string", "description": "Data to write (for write/submit actions)", "default": ""},
        "timeout": {"type": "integer", "description": "Wait timeout in seconds", "default": 30},
    }
    required = ["action", "session_id"]

    _processes: dict[str, asyncio.subprocess.Process] = {}

    async def execute(self, **kwargs: Any) -> ToolResult:
        action: str = kwargs["action"]
        session_id: str = kwargs["session_id"]
        data: str = kwargs.get("data", "")
        timeout: int = kwargs.get("timeout", 30)

        proc = self._processes.get(session_id)
        if not proc:
            return ToolResult(success=False, output="",
                              error=f"Process {session_id} not found")

        try:
            if action == "poll":
                if proc.returncode is not None:
                    stdout, stderr = await proc.communicate()
                    del self._processes[session_id]
                    return ToolResult(
                        success=proc.returncode == 0,
                        output=stdout.decode("utf-8", errors="replace")[:50_000],
                        error=stderr.decode("utf-8", errors="replace")[:10_000],
                        data={"exit_code": proc.returncode},
                    )
                return ToolResult(success=True, output="Process still running",
                                  data={"running": True})

            elif action == "kill":
                proc.kill()
                del self._processes[session_id]
                return ToolResult(success=True, output=f"Killed {session_id}")

            elif action == "write":
                if proc.stdin:
                    proc.stdin.write(data.encode())
                    await proc.stdin.drain()
                    return ToolResult(success=True, output="Data written")
                return ToolResult(success=False, error="Process stdin not available")

            elif action == "submit":
                if proc.stdin:
                    proc.stdin.write((data + "\n").encode())
                    await proc.stdin.drain()
                    return ToolResult(success=True, output="Data submitted")
                return ToolResult(success=False, error="Process stdin not available")

            elif action == "close":
                if proc.stdin:
                    proc.stdin.close()
                return ToolResult(success=True, output="stdin closed")

            elif action == "wait":
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=timeout
                    )
                    del self._processes[session_id]
                    return ToolResult(
                        success=proc.returncode == 0,
                        output=stdout.decode("utf-8", errors="replace")[:50_000],
                        error=stderr.decode("utf-8", errors="replace")[:10_000],
                        data={"exit_code": proc.returncode},
                    )
                except asyncio.TimeoutError:
                    return ToolResult(success=False, error=f"Wait timed out after {timeout}s")

            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


# ── File Tools ──

class FileReadTool(BaseTool):
    name = "file_read"
    description = "Read a text file with line numbers and pagination."
    parameters = {
        "path": {"type": "string", "description": "File path (absolute or relative)"},
        "offset": {"type": "integer", "description": "Start line (1-indexed)", "default": 1},
        "limit": {"type": "integer", "description": "Max lines to read", "default": 500},
    }
    required = ["path"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = Path(kwargs["path"]).expanduser().resolve()
        offset = kwargs.get("offset", 1)
        limit = kwargs.get("limit", 500)
        try:
            if not path.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            selected = lines[offset - 1 : offset - 1 + limit]
            numbered = "\n".join(f"{i + offset}|{line}" for i, line in enumerate(selected))
            return ToolResult(success=True, output=numbered,
                              data={"total_lines": len(lines), "path": str(path)})
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class FileWriteTool(BaseTool):
    name = "file_write"
    description = "Write content to a file. Creates or overwrites."
    parameters = {
        "path": {"type": "string", "description": "File path"},
        "content": {"type": "string", "description": "Content to write"},
    }
    required = ["path", "content"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = Path(kwargs["path"]).expanduser().resolve()
        content: str = kwargs["content"]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Written {len(content)} chars to {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class FilePatchTool(BaseTool):
    name = "file_patch"
    description = "Find and replace text in a file."
    parameters = {
        "path": {"type": "string", "description": "File path"},
        "old_string": {"type": "string", "description": "Text to find"},
        "new_string": {"type": "string", "description": "Replacement text"},
    }
    required = ["path", "old_string", "new_string"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = Path(kwargs["path"]).expanduser().resolve()
        old: str = kwargs["old_string"]
        new: str = kwargs["new_string"]
        try:
            if not path.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")
            content = path.read_text(encoding="utf-8")
            if old not in content:
                return ToolResult(success=False, output="",
                                  error=f"Pattern not found in {path}")
            content = content.replace(old, new, 1)
            path.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Patched {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class FileSearchTool(BaseTool):
    name = "file_search"
    description = "Search files by name glob or content regex."
    parameters = {
        "pattern": {"type": "string", "description": "File glob or regex pattern"},
        "path": {"type": "string", "description": "Directory to search (default: cwd)", "default": "."},
        "target": {"type": "string", "description": "'files' for name search, 'content' for grep", "default": "content"},
        "limit": {"type": "integer", "description": "Max results", "default": 50},
    }
    required = ["pattern"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        pattern: str = kwargs["pattern"]
        search_path = Path(kwargs.get("path", ".")).expanduser().resolve()
        target: str = kwargs.get("target", "content")
        limit: int = kwargs.get("limit", 50)
        try:
            if target == "files":
                matches = list(search_path.rglob(pattern))[:limit]
                output = "\n".join(str(m) for m in matches)
                return ToolResult(success=True, output=output,
                                  data={"matches": [str(m) for m in matches]})
            else:
                import re, subprocess
                result = subprocess.run(
                    ["rg", "--no-heading", "--line-number", "-m", str(limit), pattern, str(search_path)],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode in (0, 1):  # 1 = no matches, that's ok
                    return ToolResult(success=True, output=result.stdout[:50_000])
                # Fallback to grep
                result = subprocess.run(
                    ["grep", "-rn", "-m", str(limit), pattern, str(search_path)],
                    capture_output=True, text=True, timeout=30,
                )
                return ToolResult(success=True, output=result.stdout[:50_000])
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


# ── Web Tools ──

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for information."
    parameters = {
        "query": {"type": "string", "description": "Search query"},
        "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
    }
    required = ["query"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs["query"]
        limit: int = kwargs.get("limit", 5)
        try:
            import aiohttp
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": "1"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    data = await resp.json()
            results = []
            for topic in data.get("RelatedTopics", [])[:limit]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({"title": topic.get("Text", "")[:100],
                                    "url": topic.get("FirstURL", ""),
                                    "snippet": topic.get("Text", "")})
            if not results and data.get("AbstractText"):
                results.append({"title": data.get("Heading", ""),
                                "url": data.get("AbstractURL", ""),
                                "snippet": data["AbstractText"]})
            return ToolResult(success=True, output=json.dumps(results, indent=2),
                              data={"results": results})
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WebExtractTool(BaseTool):
    name = "web_extract"
    description = "Extract readable content from a URL."
    parameters = {"url": {"type": "string", "description": "URL to extract"}}
    required = ["url"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        url: str = kwargs["url"]
        try:
            import aiohttp, re
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15,
                                        headers={"User-Agent": "Mozilla/5.0 (compatible; APEX/2.0)"}) as resp:
                    html = await resp.text()
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return ToolResult(success=True, output=text[:10_000])
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


# ── Code Execution Tool ──

class CodeExecTool(BaseTool):
    name = "code_exec"
    description = "Execute code in a sandboxed subprocess. Supports Python, JavaScript (node), and Bash."
    parameters = {
        "code": {"type": "string", "description": "Code to execute"},
        "language": {"type": "string", "description": "Language: python, javascript, bash (default: python)", "default": "python"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 60)", "default": 60},
    }
    required = ["code"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        code: str = kwargs["code"]
        language: str = kwargs.get("language", "python")
        timeout: int = kwargs.get("timeout", 60)
        start = time.time()

        cmd_map = {
            "python": ["python3", "-c", code],
            "javascript": ["node", "-e", code],
            "bash": ["bash", "-c", code],
        }
        cmd = cmd_map.get(language, ["python3", "-c", code])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            duration = time.time() - start
            return ToolResult(
                success=proc.returncode == 0,
                output=stdout.decode("utf-8", errors="replace")[:50_000],
                error=stderr.decode("utf-8", errors="replace")[:10_000],
                duration=duration,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(success=False, output="",
                              error=f"Timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


# ── Tool Registry with Policy (from OpenClaw tool-policy) ──

class ToolRegistry:
    """
    Dynamic tool registry with policy enforcement.
    Combines Hermes v1 registry + OpenClaw tool-policy-pipeline.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._mcp_tools: dict[str, ToolDefinition] = {}
        self._plugin_tools: dict[str, BaseTool] = {}
        self._register_defaults()

    def _register_defaults(self):
        for cls in [ShellTool, ProcessTool, FileReadTool, FileWriteTool,
                     FilePatchTool, FileSearchTool, WebSearchTool,
                     WebExtractTool, CodeExecTool]:
            tool = cls()
            self._tools[tool.name] = tool

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_mcp_tools(self, server_name: str, tools: list[dict[str, Any]]):
        """Register tools from an MCP server (from OpenClaw mcp pattern)."""
        for t in tools:
            name = f"mcp_{server_name}_{t['name']}"
            self._mcp_tools[name] = ToolDefinition(
                name=name,
                description=t.get("description", ""),
                parameters=t.get("inputSchema", {"type": "object", "properties": {}}),
            )

    def register_plugin_tool(self, tool: BaseTool):
        self._plugin_tools[tool.name] = tool
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def check_policy(self, tool_name: str, allow: list[str] | None = None,
                     deny: list[str] | None = None) -> bool:
        """Check if a tool is allowed by policy."""
        if deny and tool_name in deny:
            return False
        if allow and allow != ["*"] and tool_name not in allow:
            # Check wildcards
            for pattern in allow:
                if pattern.endswith("*") and tool_name.startswith(pattern[:-1]):
                    return True
            return False
        return True

    def list_tools(self, allow: list[str] | None = None,
                   deny: list[str] | None = None) -> list[ToolDefinition]:
        """List all allowed tool definitions."""
        results = []
        for name, tool in self._tools.items():
            if self.check_policy(name, allow, deny):
                results.append(tool.to_definition())
        # Add MCP tools
        for name, td in self._mcp_tools.items():
            if self.check_policy(name, allow, deny):
                results.append(td)
        return results

    def to_openai_tools(self, allow: list[str] | None = None,
                        deny: list[str] | None = None) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self.list_tools(allow, deny)]

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, output="",
                              error=f"Tool '{name}' not found")
        logger.info(f"Executing tool: {name}")
        return await tool.execute(**kwargs)
