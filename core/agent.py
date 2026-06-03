"""
APEX Agent Loop
The core reasoning engine. Async, tool-aware, with multi-model support.
"""
from __future__ import annotations

import asyncio
import json
import time
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from apex.config import Config
from apex.models import ModelRouter, Message, Role, ModelResponse
from apex.tools import ToolRegistry, ToolResult
from apex.memory import HybridMemory

logger = logging.getLogger("apex.agent")


@dataclass
class AgentContext:
    """Context for a single agent run."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages: list[Message] = field(default_factory=list)
    system_prompt: str = ""
    max_iterations: int = 25
    max_tool_calls: int = 50
    _tool_call_count: int = 0

    def add_message(self, role: Role | str, content: str, **kwargs):
        self.messages.append(Message(role=role, content=content, **kwargs))

    def add_user(self, content: str):
        self.add_message(Role.USER, content)

    def add_assistant(self, content: str, tool_calls: list | None = None):
        self.add_message(Role.ASSISTANT, content, tool_calls=tool_calls or [])

    def add_tool_result(self, tool_call_id: str, name: str, content: str):
        self.add_message(
            Role.TOOL, content,
            tool_call_id=tool_call_id,
            name=name,
        )


@dataclass
class AgentResult:
    """Result of an agent run."""
    content: str
    success: bool
    iterations: int
    tool_calls: int
    duration: float
    session_id: str
    messages: list[dict] = field(default_factory=list)


class Agent:
    """
    The APEX Agent — an autonomous AI reasoning loop.

    Flow:
    1. Receive user message
    2. Build context (system prompt + memory + history)
    3. Call model
    4. If model requests tool calls → execute them → add results → loop back to 3
    5. If model returns text → deliver to user
    """

    def __init__(
        self,
        config: Config,
        model_router: ModelRouter,
        tools: ToolRegistry,
        memory: HybridMemory,
    ):
        self.config = config
        self.models = model_router
        self.tools = tools
        self.memory = memory
        self._default_system = self._build_default_system()

    def _build_default_system(self) -> str:
        return f"""You are {self.config.name}, an autonomous AI agent.

## Your Capabilities
- Execute shell commands (builds, git, file ops, installs)
- Read/write/patch files
- Search the web and extract page content
- Execute Python code
- Remember information across sessions
- Search past conversations

## Rules
1. Always use tools to take action — never just describe what you'd do.
2. When you say "I will do X", immediately call the tool for X.
3. Keep working until the task is fully complete.
4. Use absolute file paths.
5. Verify results before reporting success.
6. If a tool fails, try an alternative approach.
7. Be concise — focus on actions and results, not narration.

## Memory
- Use memory_search to recall past context before asking the user to repeat info.
- Save important facts with memory_remember (key, content, category).
- Categories: user_pref, env_fact, skill_note, general.

## Session
- Session ID is available for tracking.
- Save session summaries when completing complex tasks.
"""

    async def run(
        self,
        user_message: str,
        system_prompt: str | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the agent loop."""
        ctx = AgentContext(
            session_id=session_id or str(uuid.uuid4())[:8],
            system_prompt=system_prompt or self._default_system,
        )

        # Load session history if resuming
        if session_id:
            saved = await self.memory.load_session(session_id)
            if saved:
                for msg in saved.get("messages", []):
                    ctx.messages.append(Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                    ))

        # Inject memory context
        memory_ctx = await self._build_memory_context(user_message)
        if memory_ctx:
            ctx.system_prompt += f"\n## Relevant Memory\n{memory_ctx}"

        ctx.add_user(user_message)

        start = time.time()
        iterations = 0
        total_tool_calls = 0
        final_content = ""

        while iterations < ctx.max_iterations:
            iterations += 1
            logger.info(f"[Session {ctx.session_id}] Iteration {iterations}")

            # Build messages for model
            msgs = [Message(role=Role.SYSTEM, content=ctx.system_prompt)] + ctx.messages

            # Get tool schemas
            tool_schemas = self.tools.to_openai_tools()

            # Call model
            try:
                response = await self.models.chat(
                    messages=msgs,
                    tools=tool_schemas if tool_schemas else None,
                )
            except Exception as e:
                logger.error(f"Model call failed: {e}")
                return AgentResult(
                    content=f"Error: Model call failed: {e}",
                    success=False,
                    iterations=iterations,
                    tool_calls=total_tool_calls,
                    duration=time.time() - start,
                    session_id=ctx.session_id,
                )

            # Check for tool calls
            if response.has_tool_calls:
                ctx.add_assistant(response.content or "", response.tool_calls)

                for tc in response.tool_calls:
                    total_tool_calls += 1
                    if total_tool_calls > ctx.max_tool_calls:
                        ctx.add_tool_result(
                            tc.get("id", ""),
                            tc.get("function", {}).get("name", "unknown"),
                            "Error: Max tool calls exceeded",
                        )
                        break

                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    tool_args_str = fn.get("arguments", "{}")
                    tool_call_id = tc.get("id", "")

                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}

                    logger.info(f"  Tool: {tool_name}({list(tool_args.keys())})")

                    # Handle built-in memory tools
                    if tool_name == "memory_remember":
                        await self.memory.remember(**tool_args)
                        result = ToolResult(success=True, output="Memory saved.")
                    elif tool_name == "memory_recall":
                        content = await self.memory.recall(tool_args.get("key", ""))
                        result = ToolResult(success=True, output=content or "Not found.")
                    elif tool_name == "memory_search":
                        entries = await self.memory.search(
                            tool_args.get("query", ""),
                            tool_args.get("limit", 5),
                        )
                        output = "\n".join(f"{e.key}: {e.content}" for e in entries)
                        result = ToolResult(success=True, output=output or "No results.")
                    elif tool_name == "memory_forget":
                        deleted = await self.memory.forget(tool_args.get("key", ""))
                        result = ToolResult(success=True, output=f"Deleted: {deleted}")
                    else:
                        result = await self.tools.execute(tool_name, **tool_args)

                    ctx.add_tool_result(
                        tool_call_id,
                        tool_name,
                        result.output if result.success else f"Error: {result.error}",
                    )

                if total_tool_calls > ctx.max_tool_calls:
                    break
                continue

            # No tool calls — this is the final response
            final_content = response.content
            ctx.add_assistant(final_content)
            break

        duration = time.time() - start

        # Save session
        await self.memory.save_session(
            ctx.session_id,
            [m.to_dict() for m in ctx.messages],
            summary=final_content[:200] if final_content else "",
        )

        return AgentResult(
            content=final_content or "No response generated.",
            success=bool(final_content),
            iterations=iterations,
            tool_calls=total_tool_calls,
            duration=duration,
            session_id=ctx.session_id,
            messages=[m.to_dict() for m in ctx.messages],
        )

    async def _build_memory_context(self, query: str) -> str:
        """Build memory context for the current query."""
        try:
            entries = await self.memory.search(query, limit=3)
            if not entries:
                return ""
            lines = []
            for e in entries:
                lines.append(f"- [{e.category}] {e.key}: {e.content[:200]}")
            return "\n".join(lines)
        except Exception:
            return ""

    async def stream(
        self,
        user_message: str,
        system_prompt: str | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ):
        """Stream agent response (yields text chunks)."""
        # For streaming, we do a simplified single-pass
        ctx = AgentContext(
            session_id=session_id or str(uuid.uuid4())[:8],
            system_prompt=system_prompt or self._default_system,
        )
        ctx.add_user(user_message)

        msgs = [Message(role=Role.SYSTEM, content=ctx.system_prompt)] + ctx.messages
        tool_schemas = self.tools.to_openai_tools()

        async for chunk in self.models.stream(
            messages=msgs,
            tools=tool_schemas if tool_schemas else None,
        ):
            yield chunk
