from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    instructions: str

    @classmethod
    def from_path(cls, path: Path) -> Skill:
        text = path.read_text(encoding="utf-8")

        # Parse YAML frontmatter if present
        frontmatter: dict[str, str] = {}
        body = text
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if fm_match:
            for line in fm_match.group(1).splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    frontmatter[key.strip()] = value.strip().strip("\"'")
            body = fm_match.group(2)

        name = frontmatter.get("name", path.stem)
        description = frontmatter.get("description", "")

        return cls(name=name, description=description, instructions=body.strip())

    @classmethod
    def from_directory(cls, directory: Path) -> Skill:
        return cls.from_path(directory / "SKILL.md")

    def render(self) -> str:
        return (
            f"<skill name=\"{self.name}\">\n"
            f"<description>{self.description}</description>\n"
            f"{self.instructions}\n"
            f"</skill>"
        )
