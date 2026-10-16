from pathlib import Path

import pytest

from synthia.knowledge.model import SourceKind
from synthia.knowledge.retrieve import Side, Variant
from synthia.knowledge.service import Knowledge
from tests.knowledge.corpus import opened, planted

WORDS = Variant(Side.WORDS)
MARK = "zebraword"
VAULT = {
    ".obsidian/app.json": "{}",
    ".trash/Gone.md": f"# Gone\n\n{MARK} [[Fusion]]\n",
    "Index.md": (
        "---\n"
        f"tag: {MARK}\n"
        "related:\n"
        '  - "[[Ranking]]"\n'
        "---\n"
        "# Index\n"
        "\n"
        f"{MARK} [[Fusion]] and [[fusion.md|the fusion note]]\n"
        f"{MARK} [[Fusion#Weights]] and ![[Fusion]]\n"
        f"{MARK} [[Archive/Plan]] [[Nowhere/Plan]] [[Missing note]]\n"
    ),
    "notes/Fusion.md": "# Fusion\n\n## Weights\n\nword weight\n",
    "notes/Ranking.md": "rank\n",
    "Archive/Plan.md": "# Old plan\n\nold\n",
    "Work/Plan.md": "# Plan\n\nnew\n",
}


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    return planted(tmp_path / "vault", VAULT)


async def _links(knowledge: Knowledge) -> list[tuple[str, str | None]]:
    found = await knowledge.search(MARK, variant=WORDS)
    passages = sorted(found.passages, key=lambda p: p.chunk.start_line)
    reached = await knowledge.links(passages)
    return [
        (target.title, target.anchor)
        for passage in passages
        for target in reached.get(passage.chunk.id, ())
    ]


async def test_a_folder_holding_dot_obsidian_is_a_vault_and_its_dot_folders_are_skipped(
    tmp_path: Path, vault: Path
) -> None:
    knowledge = await opened(tmp_path / "knowledge.db")
    source = await knowledge.add(vault)
    await knowledge.index()

    found = await knowledge.search(MARK, variant=WORDS)

    assert source.kind is SourceKind.VAULT
    assert {str(p.document.path) for p in found.passages} == {"Index.md"}


async def test_vault_links_resolve_as_obsidian_resolves_them(
    tmp_path: Path, vault: Path
) -> None:
    knowledge = await opened(tmp_path / "knowledge.db")
    await knowledge.add(vault)
    await knowledge.index()

    assert await _links(knowledge) == [
        ("Ranking", None),
        ("Fusion", None),
        ("Fusion", None),
        ("Fusion", "Weights"),
        ("Fusion", None),
        ("Old plan", None),
    ]


async def test_a_renamed_note_resolves_under_its_new_name_after_index(
    tmp_path: Path, vault: Path
) -> None:
    knowledge = await opened(tmp_path / "knowledge.db")
    await knowledge.add(vault)
    await knowledge.index()
    before = await _links(knowledge)
    index = vault / "Index.md"
    (vault / "notes/Ranking.md").rename(vault / "notes/Scoring.md")
    index.write_bytes(index.read_bytes().replace(b"[[Ranking]]", b"[[Scoring]]"))

    await knowledge.index()

    assert before[0] == ("Ranking", None)
    assert (await _links(knowledge))[0] == ("Scoring", None)
