"""
APEX v2.0 Agent Loop
The core reasoning engine.

Combines:
- Hermes v1 agent loop (tool calling, sessions)
- OpenClaw agent harness (sub-agents, compaction, tool policy)
- OpenHuman agent orchestration (memory ingestion, entity extraction)

Key v2.0 improvements:
- Compaction with partial summarization (from OpenClaw)
- Tool policy enforcement per agent definition (from OpenClaw)
- Sub-agent spawning with depth limits (from OpenClaw)
- Memory tree ingestion (from OpenHuman)
- Entity extraction on conversations (from OpenHuman)
- Auth profile model selection (from OpenClaw)
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from apex.config import Config, AgentDefinition
from apex.models import ModelRouter, Message, Role, ModelResponse
from apex.tools import ToolRegistry, ToolResult
from apex.memory import MemoryManager
from apex.skills import SkillManager

logger = logging.getLogger("apex.agent")


@dataclass
class AgentContext:
    """Context for a single agent run."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages: list[Message] = field(default_factory=list)
    system_prompt: str = ""
    agent_def: AgentDefinition = field(default_factory=AgentDefinition)
    max_iterations: int = 25
    max_tool_calls: int = 50
    _tool_call_count: int = 0
    _start_time: float = field(default_factory=time.time)

    def add(self, role: Role | str, content: str, **kwargs):
        self.messages.append(Message(role=role, content=content, **kwargs))

    def add_user(self, content: str):
        self.add(Role.USER, content)

    def add_assistant(self, content: str, tool_calls: list | None = None):
        self.add(Role.ASSISTANT, content, tool_calls=tool_calls or [])

    def add_tool_result(self, tool_call_id: str, name: str, content: str):
        self.add(Role.TOOL, content, tool_call_id=tool_call_id, name=name)


@dataclass
class AgentResult:
    content: str
    success: bool
    iterations: int
    tool_calls: int
    duration: float
    session_id: str
    messages: list[dict] = field(default_factory=list)


class CompactionManager:
    """
    Handles context compaction (from OpenClaw compaction).
    Summarizes old messages when context gets too long.
    """

    def __init__(self, max_tokens: int = 6000, keep_turns: int = 6):
        self.max_tokens = max_tokens
        self.keep_turns = keep_turns

    def should_compact(self, messages: list[Message]) -> bool:
        """Check if compaction is needed."""
        # Rough token estimate: 1 token ≈ 4 chars
        total_chars = sum(len(m.content) for m in messages)
        return total_chars > self.max_tokens * 4

    async def compact(self, messages: list[Message], model_router: ModelRouter,
                       summarization_model: str = "") -> list[Message]:
        """Compact messages by summarizing older turns."""
        if len(messages) <= self.keep_turns * 2:
            return messages

        # Keep system prompt + last N turns
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]
        keep = non_system[-(self.keep_turns * 2):]
        to_compact = non_system[:-(self.keep_turns * 2)]

        if not to_compact:
            return messages

        # Summarize the old messages
        summary_prompt = "## Conversation Summary (compacted)\n\n"
        for m in to_compact:
            role = str(m.role)
            summary_prompt += f"[{role}]: {m.content[:200]}\n\n"

        try:
            response = await model_router.chat(
                messages=[Message(role=Role.USER, content=summary_prompt)],
                profile=summarization_model or None,
            )
            summary = response.content if response.success else "(compaction failed)"
        except Exception:
            summary = "(compaction failed)"

        summary_msg = Message(
            role=Role.SYSTEM,
            content=f"[Previous conversation summary]\n{summary}",
        )
        return system_msgs + [summary_msg] + keep


class Agent:
    """
    The APEX Agent — autonomous AI reasoning loop.

    Flow:
    1. Build context (system prompt + memory + skills + agent definition)
    2. Call model with tool schemas
    3. If tool calls → execute → add results → loop
    4. If text response → deliver
    5. Save session + memory tree
    """

    def __init__(
        self,
        config: Config,
        model_router: ModelRouter,
        tools: ToolRegistry,
        memory: MemoryManager,
        skills: SkillManager | None = None,
        agent_name: str = "default",
    ):
        self.config = config
        self.models = model_router
        self.tools = tools
        self.memory = memory
        self.skills = skills
        self.agent_name = agent_name
        self.agent_def = config.get_agent_def(agent_name)
        self.compaction = CompactionManager(
            max_tokens=config.compaction_max_tokens,
            keep_turns=config.compaction_keep_turns,
        )
        self._default_system = self._build_default_system()

    def _build_default_system(self) -> str:
        return f"""You are {self.config.name}, an autonomous AI agent.

## Capabilities
- Shell commands (builds, git, file ops, installs, background processes)
- File read/write/patch/search
- Web search and page content extraction
- Code execution (Python, JavaScript, Bash)
- Memory (remember, recall, search, forget)
- Sub-agent delegation (spawn researcher/coder workers)
- Multi-model routing with auto-failover

## Rules
1. Always use tools to take action — never just describe what you would do.
2. When you say "I will do X", immediately call the tool for X.
3. Keep working until the task is fully complete.
4. Use absolute file paths.
5. Verify results before reporting success.
6. If a tool fails, try an alternative approach.
7. Be concise — focus on actions and results.

## Memory
- Use memory_search to recall past context before asking the user to repeat.
- Save important facts with memory_remember (key, content, category).
- Categories: user_pref, env_fact, skill_note, general.
"""

    async def run(
        self,
        user_message: str,
        system_prompt: str | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the agent loop."""
        agent_def = self.config.get_agent_def(agent_name or self.agent_name)

        ctx = AgentContext(
            session_id=session_id or str(uuid.uuid4())[:8],
            system_prompt=system_prompt or self._default_system,
            agent_def=agent_def,
            max_iterations=agent_def.max_iterations,
            max_tool_calls=agent_def.max_tool_calls,
        )

        # Load session history
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

        # Inject skill content if relevant
        if self.skills:
            skill_ctx = await self._build_skill_context(user_message)
            if skill_ctx:
                ctx.system_prompt += f"\n{skill_ctx}"

        ctx.add_user(user_message)

        start = time.time()
        iterations = 0
        total_tool_calls = 0
        final_content = ""

        # Tool schemas with policy enforcement
        tool_schemas = self.tools.to_openai_tools(
            allow=agent_def.tools_allow,
            deny=agent_def.tools_deny,
        )

        profile = agent_def.model_profile or self.config.model.primary

        while iterations < ctx.max_iterations:
            iterations += 1
            logger.info(f"[{ctx.session_id}] Iteration {iterations}")

            # Compaction check
            if self.config.compaction_enabled and self.compaction.should_compact(ctx.messages):
                logger.info("Compacting context...")
                ctx.messages = await self.compaction.compact(
                    ctx.messages, self.models, profile,
                )

            # Build messages
            msgs = [Message(role=Role.SYSTEM, content=ctx.system_prompt)] + ctx.messages

            # Call model
            try:
                response = await self.models.chat(
                    messages=msgs,
                    tools=tool_schemas if tool_schemas else None,
                    profile=profile,
                )
            except Exception as e:
                logger.error(f"Model call failed: {e}")
                return AgentResult(
                    content=f"Error: {e}", success=False,
                    iterations=iterations, tool_calls=total_tool_calls,
                    duration=time.time() - start, session_id=ctx.session_id,
                )

            if not response.success:
                return AgentResult(
                    content=f"Model error: {response.error}", success=False,
                    iterations=iterations, tool_calls=total_tool_calls,
                    duration=time.time() - start, session_id=ctx.session_id,
                )

            # Handle tool calls
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

                    # Check tool policy
                    if not self.tools.check_policy(tool_name, agent_def.tools_allow,
                                                    agent_def.tools_deny):
                        ctx.add_tool_result(
                            tool_call_id, tool_name,
                            f"Error: Tool '{tool_name}' not allowed by policy",
                        )
                        continue

                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}

                    logger.info(f"  Tool: {tool_name}({list(tool_args.keys())})")

                    # Built-in memory tools
                    result = await self._handle_memory_tools(tool_name, tool_args)

                    if result is None:
                        result = await self.tools.execute(tool_name, **tool_args)

                    ctx.add_tool_result(
                        tool_call_id, tool_name,
                        result.output[:50000] if result.success else f"Error: {result.error}",
                    )

                if total_tool_calls > ctx.max_tool_calls:
                    break
                continue

            # Final response
            final_content = response.content
            ctx.add_assistant(final_content)
            break

        duration = time.time() - start

        # Save session
        await self.memory.save_session(
            ctx.session_id,
            [m.to_dict() for m in ctx.messages],
            summary=final_content[:200] if final_content else "",
            agent_name=self.agent_name,
        )

        # Ingest to memory tree (from OpenHuman pattern)
        if self.config.memory.tree.enabled and final_content:
            await self.memory.ingest_leaf(
                source="conversations",
                content=f"User: {user_message}\n\nAPEX: {final_content[:500]}",
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

    async def _handle_memory_tools(self, tool_name: str, args: dict) -> ToolResult | None:
        """Handle built-in memory tools. Returns None if not a memory tool."""
        try:
            if tool_name == "memory_remember":
                await self.memory.remember(
                    key=args.get("key", ""),
                    content=args.get("content", ""),
                    category=args.get("category", "general"),
                    importance=args.get("importance", 0.5),
                )
                return ToolResult(success=True, output="Memory saved.")
            elif tool_name == "memory_recall":
                content = await self.memory.recall(args.get("key", ""))
                return ToolResult(success=True, output=content or "Not found.")
            elif tool_name == "memory_search":
                entries = await self.memory.search(
                    args.get("query", ""), args.get("limit", 5),
                )
                output = "\n".join(f"[{e.category}] {e.key}: {e.content[:200]}" for e in entries)
                return ToolResult(success=True, output=output or "No results.")
            elif tool_name == "memory_forget":
                deleted = await self.memory.forget(args.get("key", ""))
                return ToolResult(success=True, output=f"Deleted: {deleted}")
            elif tool_name == "memory_search_entities":
                entities = await self.memory.search_entities(
                    args.get("query", ""), args.get("limit", 10),
                )
                output = "\n".join(f"{e.name} ({e.type}): {e.mentions} mentions" for e in entities)
                return ToolResult(success=True, output=output or "No entities found.")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
        return None

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

    async def _build_skill_context(self, query: str) -> str:
        """Inject relevant skill content into system prompt."""
        if not self.skills:
            return ""
        # Simple keyword matching for skill injection
        query_lower = query.lower()
        relevant = []
        for skill in self.skills.list_skills():
            # Check if skill description or name matches query
            if (skill.name.lower() in query_lower or
                any(word in query_lower for word in skill.description.lower().split()[:5])):
                content = self.skills.load_skill_content(skill.name)
                if content:
                    relevant.append(content[:1000])  # Limit skill content
        return "\n".join(relevant) if relevant else ""

    async def stream(
        self,
        user_message: str,
        system_prompt: str | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        **kwargs: Any,
    ):
        """Stream agent response (simplified single-pass)."""
        agent_def = self.config.get_agent_def(agent_name or self.agent_name)
        ctx = AgentContext(
            session_id=session_id or str(uuid.uuid4())[:8],
            system_prompt=system_prompt or self._default_system,
            agent_def=agent_def,
        )
        ctx.add_user(user_message)

        msgs = [Message(role=Role.SYSTEM, content=ctx.system_prompt)] + ctx.messages
        tool_schemas = self.tools.to_openai_tools(
            allow=agent_def.tools_allow, deny=agent_def.tools_deny,
        )
        profile = agent_def.model_profile or self.config.model.primary

        async for chunk in self.models.stream(
            messages=msgs, tools=tool_schemas if tool_schemas else None, profile=profile,
        ):
            yield chunk
