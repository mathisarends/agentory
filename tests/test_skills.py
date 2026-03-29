from pathlib import Path
from textwrap import dedent


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
