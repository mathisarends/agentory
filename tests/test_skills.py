from pathlib import Path
from textwrap import dedent

import pytest

from agentory.skills import Skill


class TestSkillFromPath:
    def test_parse_with_frontmatter(self, tmp_path: Path) -> None:
        md = tmp_path / "SKILL.md"
        md.write_text(
            dedent("""\
                ---
                name: my-skill
                description: A test skill
                ---

                # Instructions

                Do something useful.
            """),
            encoding="utf-8",
        )
        skill = Skill.from_path(md)
        assert skill.name == "my-skill"
        assert skill.description == "A test skill"
        assert "Do something useful." in skill.instructions
        assert skill.source_dir == tmp_path

    def test_parse_without_frontmatter(self, tmp_path: Path) -> None:
        md = tmp_path / "plain.md"
        md.write_text("Just instructions here.", encoding="utf-8")
        skill = Skill.from_path(md)
        assert skill.name == "plain"
        assert skill.description == ""
        assert skill.instructions == "Just instructions here."

    def test_frontmatter_with_quoted_values(self, tmp_path: Path) -> None:
        md = tmp_path / "SKILL.md"
        md.write_text(
            dedent("""\
                ---
                name: "quoted-skill"
                description: 'A quoted description'
                ---

                Body.
            """),
            encoding="utf-8",
        )
        skill = Skill.from_path(md)
        assert skill.name == "quoted-skill"
        assert skill.description == "A quoted description"


class TestSkillFromDirectory:
    def test_loads_skill_md_from_directory(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            dedent("""\
                ---
                name: dir-skill
                description: From directory
                ---

                Directory skill instructions.
            """),
            encoding="utf-8",
        )
        skill = Skill.from_directory(tmp_path)
        assert skill.name == "dir-skill"
        assert "Directory skill instructions." in skill.instructions


class TestSkillRender:
    def test_render_xml_output(self) -> None:
        skill = Skill(
            name="test-skill",
            description="Test description",
            instructions="Step 1\nStep 2",
        )
        rendered = skill.render()
        assert '<skill name="test-skill">' in rendered
        assert "<description>Test description</description>" in rendered
        assert "Step 1\nStep 2" in rendered
        assert "</skill>" in rendered


class TestSkillRenderMetadata:
    def test_render_metadata_excludes_instructions(self) -> None:
        skill = Skill(
            name="test-skill",
            description="Test description",
            instructions="Step 1\nStep 2",
        )
        rendered = skill.render_metadata()
        assert '<skill name="test-skill">' in rendered
        assert "<description>Test description</description>" in rendered
        assert "Step 1" not in rendered
        assert "</skill>" in rendered


class TestSkillListFiles:
    def test_list_files_returns_bundled_files(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text(
            "---\nname: s\ndescription: d\n---\nBody.", encoding="utf-8"
        )
        (tmp_path / "FORMS.md").write_text("forms", encoding="utf-8")
        (tmp_path / "REFERENCE.md").write_text("ref", encoding="utf-8")
        skill = Skill.from_directory(tmp_path)
        files = skill.list_files()
        assert "FORMS.md" in files
        assert "REFERENCE.md" in files
        assert "SKILL.md" not in files

    def test_list_files_empty_when_no_source_dir(self) -> None:
        skill = Skill(name="s", description="d", instructions="x")
        assert skill.list_files() == []


class TestSkillReadFile:
    def test_read_bundled_file(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text(
            "---\nname: s\ndescription: d\n---\nBody.", encoding="utf-8"
        )
        (tmp_path / "FORMS.md").write_text(
            "Form filling instructions.", encoding="utf-8"
        )
        skill = Skill.from_directory(tmp_path)
        content = skill.read_file("FORMS.md")
        assert content == "Form filling instructions."

    def test_read_file_not_found(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text(
            "---\nname: s\ndescription: d\n---\nBody.", encoding="utf-8"
        )
        skill = Skill.from_directory(tmp_path)
        with pytest.raises(FileNotFoundError):
            skill.read_file("NOPE.md")

    def test_read_file_no_source_dir(self) -> None:
        skill = Skill(name="s", description="d", instructions="x")
        with pytest.raises(FileNotFoundError):
            skill.read_file("FORMS.md")

    def test_read_file_path_traversal_blocked(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text(
            "---\nname: s\ndescription: d\n---\nBody.", encoding="utf-8"
        )
        skill = Skill.from_directory(tmp_path)
        with pytest.raises(ValueError, match="Path traversal"):
            skill.read_file("../../etc/passwd")
