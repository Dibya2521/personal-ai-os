from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.persona.library import PersonaLibrary, builtin_files, parse, resolve
from synthia.persona.model import PHRASES, PersonaError

JARVIS_LIKE = """
name = "Butler"
description = "a butler."
principles = ["Serve."]

[traits]
warmth = 1.0
formality = 1.0
wit = 1.0
vigilance = 1.0
verbosity = 1.0
"""


def write(folder: Path, key: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def traits_file(name: str, level: float, blend: str = "") -> str:
    sliders = "\n".join(f"{t} = {level}" for t in PHRASES)
    return f'name = "{name}"\ndescription = "d."\n{blend}\n[traits]\n{sliders}\n'


def test_the_builtins_are_the_default_and_its_three_ingredients() -> None:
    assert PersonaLibrary().names() == ("companion", "edith", "jarvis", "synthia")


def test_synthia_is_the_weighted_mean_of_its_blend() -> None:
    library = PersonaLibrary()
    synthia = library.get("synthia")
    weights = {"jarvis": 0.5, "companion": 0.3, "edith": 0.2}

    for trait in PHRASES:
        expected = sum(
            getattr(library.get(k).traits, trait) * w for k, w in weights.items()
        )
        assert getattr(synthia.traits, trait) == pytest.approx(expected)


def test_a_blend_keeps_its_own_principles_first_then_its_parts_without_repeats() -> (
    None
):
    library = PersonaLibrary()
    own = builtin_files()["synthia"].principles
    parts = [
        p for k in ("jarvis", "companion", "edith") for p in library.get(k).principles
    ]

    assert library.get("synthia").principles == (*own, *parts)


@pytest.mark.parametrize("key", ["companion", "edith", "jarvis", "synthia"])
def test_every_builtin_renders_a_prompt(key: str) -> None:
    persona = PersonaLibrary().get(key)

    assert persona.system_prompt().startswith(f"You are {persona.name}, ")


def test_a_user_file_replaces_a_builtin_and_the_blend_follows_it(
    tmp_path: Path,
) -> None:
    write(tmp_path, "jarvis", JARVIS_LIKE)
    library = PersonaLibrary(tmp_path)

    assert library.get("jarvis").name == "Butler"
    assert library.get("synthia").traits.wit == pytest.approx(
        0.5 * 1.0 + 0.3 * 0.45 + 0.2 * 0.15
    )
    assert "Serve." in library.get("synthia").principles


def test_a_user_folder_adds_personas_and_ignores_other_files(tmp_path: Path) -> None:
    write(tmp_path, "calm", traits_file("Calm", 0.1))
    (tmp_path / "notes.txt").write_text("not a persona", encoding="utf-8")
    (tmp_path / "old.toml").mkdir()

    names = PersonaLibrary(tmp_path).names()

    assert "calm" in names
    assert "notes" not in names
    assert "old" not in names


def test_a_missing_user_folder_means_builtins_only(tmp_path: Path) -> None:
    assert PersonaLibrary(tmp_path / "absent").names() == PersonaLibrary().names()


def test_an_unknown_persona_lists_the_ones_there_are() -> None:
    with pytest.raises(
        PersonaError, match="no persona 'friday'; there are companion, edith"
    ):
        PersonaLibrary().get("friday")


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("name = ", "not valid TOML"),
        ('name = "X"\ndescription = "d."\n', "exactly one of"),
        (traits_file("X", 0.5, "[blend]\njarvis = 1.0"), "exactly one of"),
        ('name = "X"\ndescription = "d."\n[blend]\n', "at least one persona"),
        ('name = "X"\ndescription = "d."\n[blend]\njarvis = -1.0\n', "blend.jarvis"),
        (traits_file("X", 1.5), "traits.warmth"),
        (traits_file("X", 0.5) + 'mood = "grumpy"\n', "traits.mood"),
        ('name = ""\ndescription = "d."\n[blend]\njarvis = 1.0\n', "name"),
    ],
)
def test_an_invalid_file_is_refused_with_its_name_and_the_problem(
    tmp_path: Path, text: str, problem: str
) -> None:
    write(tmp_path, "broken", text)

    with pytest.raises(PersonaError, match=r"broken\.toml") as caught:
        PersonaLibrary(tmp_path)
    assert problem in str(caught.value)


def test_a_blend_of_a_persona_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    write(
        tmp_path,
        "mix",
        'name = "Mix"\ndescription = "d."\n[blend]\njarvis = 1.0\nfriday = 1.0\n',
    )

    with pytest.raises(
        PersonaError, match="mix blends a persona that does not exist: friday"
    ):
        PersonaLibrary(tmp_path)


def test_a_blend_that_loops_is_refused_with_the_loop(tmp_path: Path) -> None:
    write(tmp_path, "a", 'name = "A"\ndescription = "d."\n[blend]\nb = 1.0\n')
    write(tmp_path, "b", 'name = "B"\ndescription = "d."\n[blend]\na = 1.0\n')

    with pytest.raises(PersonaError, match="loops: a -> b -> a"):
        PersonaLibrary(tmp_path)


def test_a_blend_of_equal_levels_is_exactly_that_level_despite_rounding() -> None:
    # Unclamped, 0.3 weighted 0.3 and 0.6 averages to 0.30000000000000004.
    files = {f"p{i}": parse(traits_file(f"P{i}", 0.3), f"p{i}") for i in range(2)}
    blend = "p0 = 0.3\np1 = 0.6"
    files["mix"] = parse(f'name = "Mix"\ndescription = "d."\n[blend]\n{blend}\n', "mix")

    assert resolve(files)["mix"].traits.warmth == 0.3


@given(
    st.lists(st.tuples(st.floats(0, 1), st.floats(1e-6, 1e6)), min_size=1, max_size=5)
)
def test_any_blend_stays_between_its_lowest_and_highest_part(
    parts: list[tuple[float, float]],
) -> None:
    files = {
        f"p{i}": parse(traits_file(f"P{i}", level), f"p{i}")
        for i, (level, _) in enumerate(parts)
    }
    blend = "\n".join(f"p{i} = {weight!r}" for i, (_, weight) in enumerate(parts))
    files["mix"] = parse(f'name = "Mix"\ndescription = "d."\n[blend]\n{blend}\n', "mix")

    mixed = resolve(files)["mix"].traits

    levels = [level for level, _ in parts]
    for trait in PHRASES:
        assert min(levels) <= getattr(mixed, trait) <= max(levels)
