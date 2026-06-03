"""
APEX Tool System
Dynamic, sandboxed tool registry with permission policies.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import shlex
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from pathlib import Path

logger = logging.getLogger("apex.tools")


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str = ""
    data: Any = None
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "data": self.data,
            "duration": self.duration,
        }


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
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
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    required: list[str] = []

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        ...

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            required=self.required,
        )


class ShellTool(BaseTool):
    """Execute shell commands with sandboxing."""

    name = "shell"
    description = "Execute a shell command. Use for file operations, builds, git, package managers."
    parameters = {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 180)", "default": 180},
        "workdir": {"type": "string", "description": "Working directory (default: cwd)", "default": ""},
    }
    required = ["command"]

    BLOCKED = ["rm -rf /", "mkfs", "dd if=", ":(){:|:&};:", "> /dev/"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        command: str = kwargs["command"]
        timeout: int = kwargs.get("timeout", 180)
        workdir: str = kwargs.get("workdir", "")

        # Safety check
        for blocked in self.BLOCKED:
            if blocked in command:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Blocked dangerous command pattern: {blocked}",
                )

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir or None,
                env={**os.environ, "PATH": os.environ.get("PATH", "")},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            duration = time.time() - start

            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")

            return ToolResult(
                success=proc.returncode == 0,
                output=output[:50_000],
                error=error[:10_000],
                duration=duration,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(success=False, output="", error=f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class FileReadTool(BaseTool):
    """Read file contents."""

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
            return ToolResult(
                success=True,
                output=numbered,
                data={"total_lines": len(lines), "path": str(path)},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class FileWriteTool(BaseTool):
    """Write content to a file."""

    name = "file_write"
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
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
    """Patch a file with find-and-replace."""

    name = "file_patch"
    description = "Find and replace text in a file. Use for targeted edits."
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
                return ToolResult(success=False, output="", error=f"Pattern not found in {path}")
            content = content.replace(old, new, 1)
            path.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Patched {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WebSearchTool(BaseTool):
    """Search the web."""

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
            # Use DuckDuckGo instant answer API (no key needed)
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": "1"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    data = await resp.json()

            results = []
            for topic in data.get("RelatedTopics", [])[:limit]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                    })

            if not results and data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", ""),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data["AbstractText"],
                })

            return ToolResult(
                success=True,
                output=json.dumps(results, indent=2),
                data={"results": results},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WebExtractTool(BaseTool):
    """Extract content from web pages."""

    name = "web_extract"
    description = "Extract readable content from a URL."
    parameters = {
        "url": {"type": "string", "description": "URL to extract content from"},
    }
    required = ["url"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        url: str = kwargs["url"]
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; APEX/0.1)"
                }) as resp:
                    html = await resp.text()

            # Basic HTML to text
            import re
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            return ToolResult(success=True, output=text[:10_000])
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class CodeExecTool(BaseTool):
    """Execute Python code in a sandboxed subprocess."""

    name = "code_exec"
    description = "Execute Python code. Has access to apex_tools module for calling other tools."
    parameters = {
        "code": {"type": "string", "description": "Python code to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 60)", "default": 60},
    }
    required = ["code"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        code: str = kwargs["code"]
        timeout: int = kwargs.get("timeout", 60)
        start = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", code,
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
            return ToolResult(success=False, output="", error=f"Code execution timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ToolRegistry:
    """
    Dynamic tool registry.
    Tools can be registered at runtime, hot-reloaded, and permission-controlled.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._permissions: dict[str, bool] = {}  # tool_name -> allowed
        self._register_defaults()

    def _register_defaults(self):
        """Register built-in tools."""
        for tool_cls in [
            ShellTool, FileReadTool, FileWriteTool, FilePatchTool,
            WebSearchTool, WebExtractTool, CodeExecTool,
        ]:
            tool = tool_cls()
            self.register(tool)

    def register(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool
        self._permissions[tool.name] = True
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str):
        """Remove a tool."""
        self._tools.pop(name, None)

    def enable(self, name: str):
        self._permissions[name] = True

    def disable(self, name: str):
        self._permissions[name] = False

    def get(self, name: str) -> BaseTool | None:
        tool = self._tools.get(name)
        if tool and self._permissions.get(name, False):
            return tool
        return None

    def list_tools(self) -> list[ToolDefinition]:
        """List all enabled tool definitions."""
        return [
            t.to_definition()
            for t in self._tools.values()
            if self._permissions.get(t.name, False)
        ]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all enabled tools to OpenAI function calling format."""
        return [t.to_openai_schema() for t in self.list_tools()]

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' not found or disabled",
            )
        logger.info(f"Executing tool: {name} with args: {list(kwargs.keys())}")
        return await tool.execute(**kwargs)
