from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    source_dir: Path | None = field(default=None, repr=False)

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

        return cls(
            name=name,
            description=description,
            instructions=body.strip(),
            source_dir=path.parent,
        )

    @classmethod
    def from_directory(cls, directory: Path) -> Skill:
        return cls.from_path(directory / "SKILL.md")

    def render_metadata(self) -> str:
        return (
            f'<skill name="{self.name}">\n'
            f"<description>{self.description}</description>\n"
            f"</skill>"
        )

    def render(self) -> str:
        return (
            f'<skill name="{self.name}">\n'
            f"<description>{self.description}</description>\n"
            f"{self.instructions}\n"
            f"</skill>"
        )

    def list_files(self) -> list[str]:
        if self.source_dir is None:
            return []
        return [
            f.name
            for f in sorted(self.source_dir.iterdir())
            if f.is_file() and f.name != "SKILL.md"
        ]

    def read_file(self, filename: str) -> str:
        if self.source_dir is None:
            raise FileNotFoundError(f"Skill '{self.name}' has no source directory")
        path = (self.source_dir / filename).resolve()
        if not path.is_relative_to(self.source_dir.resolve()):
            raise ValueError(f"Path traversal not allowed: {filename!r}")
        return path.read_text(encoding="utf-8")
