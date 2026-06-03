"""
APEX v2.0 — Autonomous Personal EXecutor
Built from the Hermes + OpenClaw + OpenHuman merger.

Architecture improvements over v1.0:
- Plugin system (from OpenClaw): dynamic tool/skill/channel plugins
- Memory tree (from OpenHuman): source trees + entity index + summarization
- Agent definitions registry (from OpenClaw): typed sub-agents with tool policy
- Multi-model with auth profiles and failover (from OpenClaw)
- Skills with install/discover (from OpenClaw ClawHub)
- ACP protocol support (from OpenClaw)
- Compaction with partial summarization (from OpenClaw)
- Tool policy pipeline (from OpenClaw)
- Channel gateway with 20+ platforms (from OpenClaw)
- CWD jail and sandbox context (from both)
- Obsidian-style markdown vault memory (from OpenHuman)
- Entity/relation extraction pipeline (from OpenHuman)
- Billions-token memory with tree-structured recall (from OpenHuman)
"""
__version__ = "2.0.0"
