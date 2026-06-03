# 🦅 APEX v2.0 — Autonomous Personal EXecutor

A powerful AI agent framework built from scratch, combining the best of **Hermes**, **OpenClaw**, and **OpenHuman**.

## What's New in v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Architecture** | Basic loop | Plugin system + agent definitions registry |
| **Memory** | SQLite + Vector | SQLite + Vector + Memory Tree + Entity Index + Markdown Vault |
| **Models** | Ollama + OpenAI | Multi-profile with circuit breaker, cooldown, failover |
| **Tools** | 7 basic | 10+ with policy pipeline, MCP support, background processes |
| **Skills** | Static files | ClawHub integration, preflight checks, run logging |
| **Multi-agent** | Basic DAG | Sub-agent spawning with depth limits, tool filtering |
| **Gateway** | Telegram + CLI + Web | Telegram + Discord + Slack + CLI + Web + WebSocket |
| **Compaction** | None | Partial summarization with configurable thresholds |
| **Security** | Basic blocks | CWD jail, tool policy, per-agent permissions |
| **Sandbox** | None | Sandbox context, process isolation |

## Architecture

```
apex/
├── __init__.py              # Package version
├── main.py                  # CLI entry point
├── config.py                # YAML config (merges all 3 systems)
├── core/
│   ├── __init__.py
│   └── agent.py             # Agent loop + compaction
├── models/
│   └── __init__.py          # Multi-model with circuit breaker
├── tools/
│   └── __init__.py          # 10+ tools + policy + MCP
├── memory/
│   └── __init__.py          # SQLite + Vector + Tree + Vault
├── skills/
│   ├── __init__.py          # Skill manager + ClawHub
│   └── debugging/SKILL.md   # Sample skill
├── gateway/
│   └── __init__.py          # Telegram, Discord, CLI, Web
├── tests/
│   ├── __init__.py
│   └── test_core.py         # 29 tests (v1, needs update)
└── pyproject.toml
```

## Design Sources

### From OpenClaw
- Plugin system (tool/skill/channel plugins)
- Agent definitions registry with tool policies
- Compaction with partial summarization
- Multi-model auth profiles with circuit breaker
- Sub-agent spawning with depth limits (ACP)
- Bash tool with PTY, background processes, CWD jail
- Skills with ClawHub integration
- 20+ channel gateway
- SQLite state DB (Kysely pattern)
- Tool policy pipeline (allow/deny lists)

### From OpenHuman
- Memory tree (source trees + entity index + summarization)
- Obsidian-style markdown vault for persistent memory
- Entity/relation extraction pipeline
- Billion-token memory with tree-structured recall
- WhatsApp/data source integrations
- Superpowers pattern for agent capabilities

### From Hermes v1
- Simple, focused agent loop
- Built-in tools (shell, file, web, code)
- Session management
- Skill system with markdown files
- Multi-gateway support

## Config Example

```yaml
name: APEX
model:
  primary: default
  fallbacks: [fast, smart]
  profiles:
    default:
      provider: ollama
      model: llama3
    fast:
      provider: ollama
      model: llama3:8b
    smart:
      provider: openai
      model: gpt-4o

agents:
  default:
    model_profile: default
    max_iterations: 25
    subagent_depth: 2
  researcher:
    model_profile: fast
    tools_allow: [web_search, web_extract, file_read, memory_*]
  coder:
    model_profile: smart
    tools_allow: [shell, file_*, code_exec, memory_*]

tools:
  policy:
    allow: ["*"]
    sandbox: true
    timeout_seconds: 300

memory:
  backend: hybrid
  tree:
    enabled: true
    auto_summarize: true
    entity_index: true
    sources: [conversations, files, web]

gateway:
  web_port: 8080
  channels:
    telegram:
      enabled: false
      bot_token: "${TELEGRAM_BOT_TOKEN}"

compaction_enabled: true
cwd_jail: true
```

## Quick Start

```bash
pip install -e ".[dev]"
apex --chat
```

## License

MIT
