"""
APEX v2.0 Configuration
Merges Hermes config patterns + OpenClaw plugin system + OpenHuman memory tree.
"""
from __future__ import annotations

import os
import yaml
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("apex.config")

APEX_HOME = Path(os.environ.get("APEX_HOME", Path.home() / ".apex"))
CONFIG_PATH = APEX_HOME / "config.yaml"
STATE_DB = APEX_HOME / "state" / "apex.sqlite"
AGENT_DB_TEMPLATE = "agents/{agent_id}/agent/apex-agent.sqlite"


# ── Model / Provider ──

@dataclass
class ModelProfile:
    """A named model configuration (from OpenClaw model-catalog pattern)."""
    provider: str = "ollama"       # ollama, openai, anthropic, google, custom
    model: str = "llama3"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    context_window: int = 8192
    supports_tools: bool = True
    supports_vision: bool = False
    reasoning: bool = False
    cost_per_1m_prompt: float = 0.0
    cost_per_1m_completion: float = 0.0


@dataclass
class ModelRouterConfig:
    """Multi-model routing with failover (from OpenClaw failover-policy)."""
    primary: str = "default"          # profile name
    fallbacks: list[str] = field(default_factory=list)
    cooldown_seconds: int = 60        # cooldown before retrying failed provider
    circuit_breaker_threshold: int = 3
    profiles: dict[str, ModelProfile] = field(default_factory=lambda: {
        "default": ModelProfile(),
    })


# ── Tools ──

@dataclass
class ToolPolicy:
    """Tool allowlist/deny list (from OpenClaw tool-policy)."""
    allow: list[str] = field(default_factory=lambda: ["*"])
    deny: list[str] = field(default_factory=list)
    sandbox: bool = True
    max_output_bytes: int = 50_000
    timeout_seconds: int = 300


@dataclass
class ToolConfig:
    policy: ToolPolicy = field(default_factory=ToolPolicy)
    enabled_builtins: list[str] = field(default_factory=lambda: [
        "shell", "file_read", "file_write", "file_patch",
        "web_search", "web_extract", "code_exec",
        "memory_remember", "memory_recall", "memory_search", "memory_forget",
    ])
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    plugins_dir: str = ""  # loaded from APEX_HOME/plugins by default


# ── Memory (from OpenHuman memory_tree) ──

@dataclass
class MemoryTreeConfig:
    """Tree-structured memory (from OpenHuman memory_tree)."""
    enabled: bool = True
    persist_dir: str = ""  # defaults to APEX_HOME/memory
    auto_summarize: bool = True
    max_leaf_tokens: int = 2000
    summarization_model: str = ""  # empty = use primary
    extraction_model: str = ""    # entity/relation extraction model
    sources: list[str] = field(default_factory=lambda: ["conversations", "files", "web"])
    entity_index: bool = True
    max_session_messages: int = 100
    vault_path: str = ""  # Obsidian-style markdown vault (from OpenHuman)


@dataclass
class MemoryConfig:
    backend: str = "hybrid"  # sqlite, vector, hybrid, tree
    tree: MemoryTreeConfig = field(default_factory=MemoryTreeConfig)
    auto_ingest: bool = True  # auto-extract entities from conversations


# ── Skills (from OpenClaw skills + ClawHub) ──

@dataclass
class SkillsConfig:
    """Skill system with ClawHub integration (from OpenClaw)."""
    dirs: list[str] = field(default_factory=list)
    auto_discover: bool = True
    clawhub_enabled: bool = True
    clawhub_url: str = "https://clawhub.com/api"
    install_on_demand: bool = True
    preflight_checks: bool = True


# ── Agent Definitions (from OpenClaw agent-registry) ──

@dataclass
class AgentDefinition:
    """A named agent archetype (from OpenClaw agent-registry)."""
    name: str = "default"
    description: str = ""
    model_profile: str = "default"
    system_prompt: str = ""
    tools_allow: list[str] = field(default_factory=lambda: ["*"])
    tools_deny: list[str] = field(default_factory=list)
    sandbox: bool = True
    max_iterations: int = 25
    max_tool_calls: int = 50
    subagent_depth: int = 2  # max nesting for spawn_subagent
    compaction: bool = True


# ── Gateway / Channels (from OpenClaw channels) ──

@dataclass
class ChannelConfig:
    """A messaging channel (from OpenClaw channels)."""
    platform: str = "telegram"  # telegram, discord, slack, signal, whatsapp, irc, matrix, cli, web
    enabled: bool = True
    token: str = ""
    bot_token: str = ""
    app_token: str = ""
    webhook_url: str = ""
    webhook_secret: str = ""
    allow_from: list[str] = field(default_factory=list)  # empty = allow all
    dm_policy: str = "open"  # open, pairing, allowlist
    group_policy: str = "mention"  # open, mention, disabled
    streaming: bool = True
    max_message_length: int = 4096
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayConfig:
    channels: dict[str, ChannelConfig] = field(default_factory=dict)
    web_port: int = 8080
    web_host: str = "127.0.0.1"
    acp_enabled: bool = True  # Agent Communication Protocol (from OpenClaw)


# ── Plugins (from OpenClaw plugin-sdk) ──

@dataclass
class PluginConfig:
    """Plugin system (from OpenClaw plugins)."""
    enabled: bool = True
    dirs: list[str] = field(default_factory=list)
    auto_load: bool = True
    allow_remote_install: bool = True
    trusted_sources: list[str] = field(default_factory=lambda: [
        "https://clawhub.com",
        "https://github.com/openclaw",
    ])


# ── Root Config ──

@dataclass
class Config:
    version: str = "2.0.0"
    name: str = "APEX"
    log_level: str = "INFO"
    debug: bool = False

    model: ModelRouterConfig = field(default_factory=ModelRouterConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    agents: dict[str, AgentDefinition] = field(default_factory=lambda: {
        "default": AgentDefinition(),
    })
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)

    # Compaction (from OpenClaw compaction)
    compaction_enabled: bool = True
    compaction_max_tokens: int = 6000
    compaction_keep_turns: int = 6

    # Security
    cwd_jail: bool = True  # restrict file ops to workspace (from OpenClaw)
    workspace_dir: str = ""

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        cfg_path = Path(path) if path else CONFIG_PATH
        if not cfg_path.exists():
            config = cls()
            config._create_default(cfg_path)
            return config
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> Config:
        """Build config from nested dict."""
        model_raw = raw.pop("model", {})
        tools_raw = raw.pop("tools", {})
        memory_raw = raw.pop("memory", {})
        skills_raw = raw.pop("skills", {})
        agents_raw = raw.pop("agents", {})
        gateway_raw = raw.pop("gateway", {})
        plugins_raw = raw.pop("plugins", {})

        # Parse model profiles
        profiles_raw = model_raw.pop("profiles", {})
        profiles = {k: ModelProfile(**v) for k, v in profiles_raw.items()}
        if not profiles:
            profiles = {"default": ModelProfile()}
        model = ModelRouterConfig(profiles=profiles, **{
            k: v for k, v in model_raw.items()
            if k in ModelRouterConfig.__dataclass_fields__
        })

        # Parse tool policy
        policy_raw = tools_raw.pop("policy", {})
        policy = ToolPolicy(**policy_raw) if policy_raw else ToolPolicy()
        tools = ToolConfig(policy=policy, **{
            k: v for k, v in tools_raw.items()
            if k in ToolConfig.__dataclass_fields__
        })

        # Parse memory tree
        tree_raw = memory_raw.pop("tree", {})
        tree = MemoryTreeConfig(**tree_raw) if tree_raw else MemoryTreeConfig()
        memory = MemoryConfig(tree=tree, **{
            k: v for k, v in memory_raw.items()
            if k in MemoryConfig.__dataclass_fields__
        })

        # Parse skills
        skills = SkillsConfig(**{
            k: v for k, v in skills_raw.items()
            if k in SkillsConfig.__dataclass_fields__
        }) if skills_raw else SkillsConfig()

        # Parse agent definitions
        agents = {k: AgentDefinition(**v) for k, v in agents_raw.items()}
        if not agents:
            agents = {"default": AgentDefinition()}

        # Parse gateway channels
        channels_raw = gateway_raw.pop("channels", {})
        channels = {k: ChannelConfig(**v) for k, v in channels_raw.items()}
        gateway = GatewayConfig(channels=channels, **{
            k: v for k, v in gateway_raw.items()
            if k in GatewayConfig.__dataclass_fields__
        }) if gateway_raw else GatewayConfig()

        # Parse plugins
        plugins = PluginConfig(**{
            k: v for k, v in plugins_raw.items()
            if k in PluginConfig.__dataclass_fields__
        }) if plugins_raw else PluginConfig()

        return cls(
            **{k: v for k, v in raw.items() if k in cls.__dataclass_fields__},
            model=model, tools=tools, memory=memory, skills=skills,
            agents=agents, gateway=gateway, plugins=plugins,
        )

    def _create_default(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        default = {
            "name": "APEX",
            "model": {
                "primary": "default",
                "fallbacks": [],
                "profiles": {
                    "default": {
                        "provider": "ollama",
                        "model": "llama3",
                        "temperature": 0.7,
                        "max_tokens": 4096,
                    },
                    "fast": {
                        "provider": "ollama",
                        "model": "llama3:8b",
                        "temperature": 0.5,
                        "max_tokens": 2048,
                    },
                    "smart": {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "temperature": 0.7,
                        "max_tokens": 8192,
                    },
                },
            },
            "tools": {
                "policy": {
                    "allow": ["*"],
                    "deny": [],
                    "sandbox": True,
                    "timeout_seconds": 300,
                },
            },
            "memory": {
                "backend": "hybrid",
                "tree": {
                    "enabled": True,
                    "auto_summarize": True,
                    "entity_index": True,
                    "sources": ["conversations", "files", "web"],
                },
            },
            "skills": {
                "auto_discover": True,
                "clawhub_enabled": True,
                "install_on_demand": True,
            },
            "agents": {
                "default": {
                    "model_profile": "default",
                    "max_iterations": 25,
                    "subagent_depth": 2,
                },
                "researcher": {
                    "model_profile": "fast",
                    "tools_allow": ["web_search", "web_extract", "file_read", "memory_*"],
                    "system_prompt": "You are a research agent. Gather and synthesize information.",
                },
                "coder": {
                    "model_profile": "smart",
                    "tools_allow": ["shell", "file_*", "code_exec", "memory_*"],
                    "system_prompt": "You are a coding agent. Write, test, and debug code.",
                },
            },
            "gateway": {
                "web_port": 8080,
                "acp_enabled": True,
            },
            "plugins": {
                "enabled": True,
                "auto_load": True,
            },
            "compaction_enabled": True,
            "cwd_jail": True,
            "log_level": "INFO",
        }
        with open(path, "w") as f:
            yaml.dump(default, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Created default config at {path}")

    def save(self, path: str | Path | None = None):
        cfg_path = Path(path) if path else CONFIG_PATH
        with open(cfg_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def get_model_profile(self, name: str | None = None) -> ModelProfile:
        name = name or self.model.primary
        return self.model.profiles.get(name, ModelProfile())

    def get_agent_def(self, name: str = "default") -> AgentDefinition:
        return self.agents.get(name, AgentDefinition())
