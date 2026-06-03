"""Tests for APEX agent framework."""
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from apex.config import Config, ModelConfig, MemoryConfig
from apex.tools import (
    ToolRegistry, ShellTool, FileReadTool, FileWriteTool,
    FilePatchTool, WebSearchTool, CodeExecTool, ToolResult,
)
from apex.memory import HybridMemory, SQLiteMemory, MemoryEntry
from apex.skills import SkillManager, Skill


# ── Config Tests ──

class TestConfig:
    def test_default_config(self):
        config = Config()
        assert config.name == "APEX"
        assert config.model.provider == "ollama"
        assert config.model.model == "llama3"
        assert config.tools.sandbox is True

    def test_config_with_custom(self):
        config = Config(name="Test", model=ModelConfig(provider="openai", model="gpt-4"))
        assert config.name == "Test"
        assert config.model.provider == "openai"
        assert config.model.model == "gpt-4"

    def test_config_save_load(self, tmp_path):
        config = Config(name="Saved", model=ModelConfig(provider="openai", model="gpt-4"))
        path = tmp_path / "config.yaml"
        config.save(path)
        loaded = Config.load(path)
        assert loaded.name == "Saved"
        assert loaded.model.provider == "openai"


# ── Tool Tests ──

class TestShellTool:
    @pytest.mark.asyncio
    async def test_simple_command(self):
        tool = ShellTool()
        result = await tool.execute(command="echo hello")
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_blocked_command(self):
        tool = ShellTool()
        result = await tool.execute(command="rm -rf / --no-preserve-root")
        assert not result.success
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self):
        tool = ShellTool()
        result = await tool.execute(command="sleep 10", timeout=1)
        assert not result.success
        assert "timed out" in result.error.lower()


class TestFileTools:
    @pytest.mark.asyncio
    async def test_write_and_read(self, tmp_path):
        write_tool = FileWriteTool()
        read_tool = FileReadTool()
        path = str(tmp_path / "test.txt")

        write_result = await write_tool.execute(path=path, content="hello world")
        assert write_result.success

        read_result = await read_tool.execute(path=path)
        assert read_result.success
        assert "hello world" in read_result.output

    @pytest.mark.asyncio
    async def test_patch(self, tmp_path):
        write_tool = FileWriteTool()
        patch_tool = FilePatchTool()
        path = str(tmp_path / "patch.txt")

        await write_tool.execute(path=path, content="hello world")
        result = await patch_tool.execute(path=path, old_string="hello", new_string="goodbye")
        assert result.success

        read_tool = FileReadTool()
        read_result = await read_tool.execute(path=path)
        assert "goodbye world" in read_result.output

    @pytest.mark.asyncio
    async def test_read_missing_file(self):
        tool = FileReadTool()
        result = await tool.execute(path="/nonexistent/file.txt")
        assert not result.success
        assert "not found" in result.error.lower()


class TestCodeExecTool:
    @pytest.mark.asyncio
    async def test_simple_code(self):
        tool = CodeExecTool()
        result = await tool.execute(code="print(2 + 2)")
        assert result.success
        assert "4" in result.output

    @pytest.mark.asyncio
    async def test_error_code(self):
        tool = CodeExecTool()
        result = await tool.execute(code="raise ValueError('test')")
        assert not result.success


# ── Tool Registry Tests ──

class TestToolRegistry:
    def test_default_tools_registered(self):
        registry = ToolRegistry()
        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert "shell" in names
        assert "file_read" in names
        assert "file_write" in names
        assert "file_patch" in names
        assert "web_search" in names
        assert "code_exec" in names

    def test_disable_enable(self):
        registry = ToolRegistry()
        registry.disable("shell")
        assert registry.get("shell") is None
        registry.enable("shell")
        assert registry.get("shell") is not None

    def test_register_custom_tool(self):
        registry = ToolRegistry()

        class CustomTool:
            name = "custom"
            description = "A custom tool"
            parameters = {}
            required = []

            async def execute(self, **kwargs):
                return ToolResult(success=True, output="custom result")

            def to_definition(self):
                from apex.tools import ToolDefinition
                return ToolDefinition(name=self.name, description=self.description,
                                       parameters=self.parameters, required=self.required)

        # Register as a proper BaseTool subclass
        from apex.tools import BaseTool

        class MyTool(BaseTool):
            name = "my_tool"
            description = "test"
            parameters = {}
            required = []
            async def execute(self, **kw):
                return ToolResult(success=True, output="ok")

        registry.register(MyTool())
        assert registry.get("my_tool") is not None

    def test_to_openai_tools(self):
        registry = ToolRegistry()
        schemas = registry.to_openai_tools()
        assert len(schemas) > 0
        assert all(s["type"] == "function" for s in schemas)


# ── Memory Tests ──

class TestSQLiteMemory:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_path):
        mem = SQLiteMemory(str(tmp_path / "test.db"))
        entry = MemoryEntry(key="test_key", content="test content", category="general")
        await mem.store(entry)

        result = await mem.retrieve("test_key")
        assert result is not None
        assert result.content == "test content"

    @pytest.mark.asyncio
    async def test_search(self, tmp_path):
        mem = SQLiteMemory(str(tmp_path / "test.db"))
        await mem.store(MemoryEntry(key="k1", content="python programming"))
        await mem.store(MemoryEntry(key="k2", content="javascript coding"))
        await mem.store(MemoryEntry(key="k3", content="python data science"))

        results = await mem.search("python")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        mem = SQLiteMemory(str(tmp_path / "test.db"))
        await mem.store(MemoryEntry(key="to_delete", content="temp"))
        assert await mem.delete("to_delete")
        assert await mem.retrieve("to_delete") is None

    @pytest.mark.asyncio
    async def test_session_management(self, tmp_path):
        mem = SQLiteMemory(str(tmp_path / "test.db"))
        await mem.save_session("sess1", [{"role": "user", "content": "hi"}], "greeting")
        loaded = await mem.load_session("sess1")
        assert loaded is not None
        assert loaded["summary"] == "greeting"


class TestHybridMemory:
    @pytest.mark.asyncio
    async def test_remember_recall(self, tmp_path):
        config = MemoryConfig(backend="hybrid")
        mem = HybridMemory(config, tmp_path / "data")
        await mem.remember("user_name", "H3KR", category="user_pref")
        result = await mem.recall("user_name")
        assert result == "H3KR"

    @pytest.mark.asyncio
    async def test_search(self, tmp_path):
        config = MemoryConfig(backend="hybrid")
        mem = HybridMemory(config, tmp_path / "data")
        await mem.remember("key1", "Python is great for scripting")
        await mem.remember("key2", "JavaScript runs in browsers")
        results = await mem.search("python")
        assert len(results) >= 1


# ── Skill Tests ──

class TestSkillManager:
    def test_discover(self, tmp_path):
        # Create a skill file
        skill_dir = tmp_path / "debugging"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: debugging
description: Debug stuff
category: dev
---
# Debugging
Some content.
""")
        manager = SkillManager(tmp_path)
        skills = manager.discover()
        assert len(skills) == 1
        assert skills[0].name == "debugging"

    def test_install_uninstall(self, tmp_path):
        manager = SkillManager(tmp_path)
        content = """---
name: test-skill
description: A test skill
---
# Test
Content here.
"""
        skill = manager.install("test-skill", content)
        assert skill.name == "test-skill"
        assert manager.get("test-skill") is not None

        assert manager.uninstall("test-skill")
        assert manager.get("test-skill") is None


# ── Model Tests ──

class TestMessage:
    def test_to_dict(self):
        from apex.models import Message, Role
        msg = Message(role=Role.USER, content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_to_dict_with_tool_calls(self):
        from apex.models import Message, Role
        msg = Message(
            role=Role.ASSISTANT, content="",
            tool_calls=[{"id": "1", "function": {"name": "shell"}}],
        )
        d = msg.to_dict()
        assert "tool_calls" in d

    def test_message_roles(self):
        from apex.models import Role
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"


class TestModelResponse:
    def test_has_tool_calls(self):
        from apex.models import ModelResponse
        r1 = ModelResponse(content="hi", tool_calls=[], model="test", provider="test")
        assert not r1.has_tool_calls

        r2 = ModelResponse(
            content="", tool_calls=[{"id": "1"}],
            model="test", provider="test",
        )
        assert r2.has_tool_calls


# ── Orchestrator Tests ──

class TestDAG:
    def test_add_and_ready(self):
        from apex.core.orchestrator import DAG, Task, TaskStatus
        dag = DAG()
        t1 = dag.add(Task(id="t1", goal="first"))
        t2 = dag.add(Task(id="t2", goal="second", dependencies=[t1]))

        ready = dag.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t1"

        # Complete t1
        dag.tasks[t1].status = TaskStatus.COMPLETED
        ready = dag.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t2"

    def test_is_complete(self):
        from apex.core.orchestrator import DAG, Task, TaskStatus
        dag = DAG()
        t1 = dag.add(Task(id="t1"))
        assert not dag.is_complete()
        dag.tasks[t1].status = TaskStatus.COMPLETED
        assert dag.is_complete()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
