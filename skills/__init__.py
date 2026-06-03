"""
APEX Skill System
Versioned, dependency-aware skills with auto-install.
"""
from __future__ import annotations

import re
import yaml
import logging
import shutil
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
    files: dict[str, str] = field(default_factory=dict)  # filename -> content
    path: Path | None = None

    @classmethod
    def from_file(cls, path: Path) -> Skill:
        """Load a skill from a SKILL.md file."""
        content = path.read_text(encoding="utf-8")

        # Parse YAML frontmatter
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
            "dependencies": self.dependencies,
        }


class SkillManager:
    """Manages skill discovery, loading, and installation."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}

    def discover(self) -> list[Skill]:
        """Scan skills directory and load all skills."""
        self._skills.clear()
        for skill_file in self.skills_dir.rglob("SKILL.md"):
            try:
                skill = Skill.from_file(skill_file)
                self._skills[skill.name] = skill
                logger.debug(f"Discovered skill: {skill.name}")
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

    def install(self, name: str, content: str, target_dir: Path | None = None) -> Skill:
        """Install a new skill."""
        skill_dir = target_dir or (self.skills_dir / name)
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")

        skill = Skill.from_file(skill_file)
        skill.path = skill_file
        self._skills[name] = skill
        logger.info(f"Installed skill: {name}")
        return skill

    def uninstall(self, name: str) -> bool:
        """Remove a skill."""
        skill = self._skills.pop(name, None)
        if skill and skill.path:
            skill_dir = skill.path.parent
            if skill_dir.exists():
                shutil.rmtree(skill_dir)
            return True
        return False

    def load_skill_content(self, name: str) -> str:
        """Get the full content of a skill for injection into system prompt."""
        skill = self.get(name)
        if not skill:
            return ""
        return f"## Skill: {skill.name}\n{skill.content}"
