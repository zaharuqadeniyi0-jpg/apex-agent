"""
APEX v2.0 Skill System
Combines Hermes v1 skills + OpenClaw skills/ClawHub + OpenHuman superpowers.

Key additions:
- ClawHub integration for skill discovery/install (from OpenClaw)
- Skill preflight checks (from OpenClaw)
- Skill run logging (from OpenClaw)
- Superpowers pattern (from OpenHuman)
- Agent-workflow skills (from OpenClaw agent-workflows)
"""
from __future__ import annotations

import re
import yaml
import json
import logging
import shutil
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("apex.skills")


@dataclass
class Skill:
    name: str
    description: str
    version: str = "1.0.0"
    category: str = "general"
    content: str = ""
    dependencies: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    path: Path | None = None
    source: str = "local"  # local, clawhub, plugin
    installed_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    use_count: int = 0

    @classmethod
    def from_file(cls, path: Path) -> Skill:
        content = path.read_text(encoding="utf-8")
        metadata: dict[str, Any] = {}
        if content.startswith("---"):
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if match:
                try:
                    metadata = yaml.safe_load(match.group(1)) or {}
                except yaml.YAMLError:
                    pass
                body = content[match.end():]
            else:
                body = content
        else:
            body = content

        name = metadata.get("name", path.stem)
        return cls(
            name=name,
            description=metadata.get("description", ""),
            version=metadata.get("version", "1.0.0"),
            category=metadata.get("category", "general"),
            content=body.strip(),
            dependencies=metadata.get("dependencies", []),
            path=path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "source": self.source,
            "dependencies": self.dependencies,
        }


@dataclass
class SkillRunLog:
    """Log of a skill execution (from OpenClaw skill-run-log)."""
    skill_name: str
    session_id: str
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    success: bool = False
    error: str = ""
    output_preview: str = ""


class SkillManager:
    """
    Manages skill discovery, loading, installation, and execution.
    Combines Hermes v1 + OpenClaw ClawHub + OpenHuman superpowers.
    """

    def __init__(self, dirs: list[Path] | None = None, clawhub_enabled: bool = True):
        self.dirs = dirs or []
        self.clawhub_enabled = clawhub_enabled
        self._skills: dict[str, Skill] = {}
        self._run_log: list[SkillRunLog] = []

    def discover(self) -> list[Skill]:
        """Scan skill directories and load all skills."""
        self._skills.clear()
        for d in self.dirs:
            if not d.exists():
                continue
            for skill_file in d.rglob("SKILL.md"):
                try:
                    skill = Skill.from_file(skill_file)
                    self._skills[skill.name] = skill
                    logger.debug(f"Discovered skill: {skill.name} ({skill.category})")
                except Exception as e:
                    logger.warning(f"Failed to load skill from {skill_file}: {e}")
        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_skills(self, category: str | None = None) -> list[Skill]:
        skills = list(self._skills.values())
        if category:
            skills = [s for s in skills if s.category == category]
        return skills

    def install(self, name: str, content: str, target_dir: Path | None = None,
                source: str = "local") -> Skill:
        """Install a new skill."""
        skill_dir = target_dir or (self.dirs[0] / name if self.dirs else Path(f"~/.apex/skills/{name}"))
        skill_dir = Path(skill_dir).expanduser()
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")
        skill = Skill.from_file(skill_file)
        skill.path = skill_file
        skill.source = source
        self._skills[name] = skill
        logger.info(f"Installed skill: {name} from {source}")
        return skill

    def uninstall(self, name: str) -> bool:
        skill = self._skills.pop(name, None)
        if skill and skill.path:
            skill_dir = skill.path.parent
            if skill_dir.exists():
                shutil.rmtree(skill_dir)
            return True
        return False

    def load_skill_content(self, name: str) -> str:
        """Get full content of a skill for injection into system prompt."""
        skill = self.get(name)
        if not skill:
            return ""
        skill.last_used = time.time()
        skill.use_count += 1
        return f"## Skill: {skill.name}\n{skill.content}"

    async def search_clawhub(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search ClawHub for skills (from OpenClaw clawhub pattern)."""
        if not self.clawhub_enabled:
            return []
        try:
            import aiohttp
            url = "https://clawhub.com/api/skills/search"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={"q": query, "limit": limit}, timeout=10) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            logger.warning(f"ClawHub search failed: {e}")
        return []

    async def install_from_clawhub(self, skill_id: str, target_dir: Path | None = None) -> Skill | None:
        """Install a skill from ClawHub."""
        try:
            import aiohttp
            url = f"https://clawhub.com/api/skills/{skill_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self.install(
                            data["name"], data["content"],
                            target_dir=target_dir, source="clawhub",
                        )
        except Exception as e:
            logger.error(f"ClawHub install failed: {e}")
        return None

    def log_run(self, log: SkillRunLog):
        self._run_log.append(log)
        # Keep last 1000 entries
        if len(self._run_log) > 1000:
            self._run_log = self._run_log[-1000:]

    def get_run_history(self, skill_name: str | None = None, limit: int = 20) -> list[SkillRunLog]:
        logs = self._run_log
        if skill_name:
            logs = [l for l in logs if l.skill_name == skill_name]
        return logs[-limit:]
