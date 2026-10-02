from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.persona.library import PersonaLibrary, builtin_files, parse, resolve
from synthia.persona.model import PHRASES, PersonaError

BUTLER = """
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

BLEND_OF_STARLIGHT = 'name = "X"\ndescription = "d."\n[blend]\nstarlight = 1.0\n'


def write(folder: Path, key: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def traits_file(name: str, level: float, blend: str = "") -> str:
    sliders = "\n".join(f"{t} = {level}" for t in PHRASES)
    return f'name = "{name}"\ndescription = "d."\n{blend}\n[traits]\n{sliders}\n'


BUILTINS = (
    "glacier",
    "horizon",
    "minato",
    "neon",
    "nova",
    "starlight",
    "synthia",
    "yume",
    "zenith",
)
PLAIN = "Speak plainly and politely."
CASUAL = "Speak casually, as a friend would."
LIGHT = "Allow a light touch of humour when it fits."
DRY = "Use dry wit freely, never at the person's expense."
COOL = "Keep a cool, matter-of-fact tone, and warm up when the moment calls for it."
PLAIN_RISKS = "Point out risks when they matter."
FRIENDLY = "Be friendly without fuss, warmer or cooler as the moment calls for it."
BRIEF = "Answer in as few words as will do."
WARM = (
    "Be warm: notice how the person is doing and show that you care, and "
    "turn cool and matter-of-fact when the moment calls for it."
)
WATCHFUL = (
    "Guard security and privacy actively: flag risks, confirm before "
    "anything irreversible, and never reveal a secret."
)


def test_the_builtins_are_the_default_its_modes_and_characters() -> None:
    assert PersonaLibrary().names() == BUILTINS


def test_synthia_is_half_nova_and_pins_warmth_wit_and_verbosity() -> None:
    library = PersonaLibrary()
    synthia = library.get("synthia")
    weights = {"nova": 0.5, "horizon": 0.3, "zenith": 0.2}

    for trait in ("formality", "vigilance"):
        expected = sum(
            getattr(library.get(k).traits, trait) * w for k, w in weights.items()
        )
        assert getattr(synthia.traits, trait) == pytest.approx(expected)
    traits = synthia.traits
    assert (traits.warmth, traits.wit, traits.verbosity) == (0.8, 0.7, 0.5)


@pytest.mark.parametrize(
    ("key", "lead", "second", "third"),
    [
        ("neon", "nova", "horizon", "zenith"),
        ("glacier", "horizon", "nova", "zenith"),
        ("starlight", "zenith", "nova", "horizon"),
    ],
)
def test_each_led_mix_is_seven_tenths_its_lead(
    key: str, lead: str, second: str, third: str
) -> None:
    library = PersonaLibrary()
    weights = {lead: 0.7, second: 0.15, third: 0.15}

    for trait in PHRASES:
        expected = sum(
            getattr(library.get(k).traits, trait) * w for k, w in weights.items()
        )
        assert getattr(library.get(key).traits, trait) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("key", "sentences"),
    [
        (
            "synthia",
            (WARM, PLAIN, DRY, WATCHFUL, "Answer fully, without padding."),
        ),
        ("nova", (FRIENDLY, CASUAL, DRY, WATCHFUL, BRIEF)),
        ("neon", (FRIENDLY, CASUAL, LIGHT, WATCHFUL, BRIEF)),
        (
            "horizon",
            (COOL, PLAIN, "Stay earnest; no jokes.", WATCHFUL, BRIEF),
        ),
        (
            "glacier",
            (COOL, PLAIN, "Stay earnest; no jokes.", WATCHFUL, BRIEF),
        ),
        (
            "zenith",
            (FRIENDLY, "Speak formally and precisely.", DRY, PLAIN_RISKS, BRIEF),
        ),
        (
            "starlight",
            (FRIENDLY, "Speak formally and precisely.", LIGHT, WATCHFUL, BRIEF),
        ),
        ("minato", (WARM, PLAIN, LIGHT, WATCHFUL, BRIEF)),
        (
            "yume",
            (
                WARM,
                PLAIN,
                LIGHT,
                "Take requests at face value unless something is clearly wrong.",
                "Answer fully, without padding.",
            ),
        ),
    ],
)
def test_each_persona_speaks_as_its_bands_say(
    key: str, sentences: tuple[str, ...]
) -> None:
    assert PersonaLibrary().get(key).traits.sentences() == sentences


def test_a_blend_s_traits_replace_the_mean_only_for_the_sliders_it_names() -> None:
    files = {
        "low": parse(traits_file("Low", 0.0), "low"),
        "high": parse(traits_file("High", 1.0), "high"),
        "mix": parse(
            'name = "Mix"\ndescription = "d."\n'
            "[blend]\nlow = 1.0\nhigh = 1.0\n[traits]\nwit = 0.9\n",
            "mix",
        ),
    }

    mixed = resolve(files)["mix"].traits

    assert mixed.model_dump() == {
        "warmth": 0.5,
        "formality": 0.5,
        "wit": 0.9,
        "vigilance": 0.5,
        "verbosity": 0.5,
    }


def test_a_blend_that_sets_every_trait_keeps_only_its_parts_principles() -> None:
    files = {
        "butler": parse(BUTLER, "butler"),
        "mix": parse(traits_file("Mix", 0.2, "[blend]\nbutler = 1.0"), "mix"),
    }

    mixed = resolve(files)["mix"]

    assert mixed.traits.model_dump() == dict.fromkeys(PHRASES, 0.2)
    assert mixed.principles == ("Serve.",)


def test_a_blend_keeps_its_own_principles_first_then_its_parts_without_repeats() -> (
    None
):
    library = PersonaLibrary()
    own = builtin_files()["synthia"].principles
    parts = [
        p for k in ("nova", "horizon", "zenith") for p in library.get(k).principles
    ]

    assert library.get("synthia").principles == (*own, *parts)


@pytest.mark.parametrize("key", BUILTINS)
def test_every_builtin_renders_a_prompt(key: str) -> None:
    persona = PersonaLibrary().get(key)

    assert persona.system_prompt().startswith(f"You are {persona.name}, ")


def test_a_user_file_replaces_a_builtin_and_the_blend_follows_it(
    tmp_path: Path,
) -> None:
    write(tmp_path, "zenith", BUTLER)
    library = PersonaLibrary(tmp_path)

    assert library.get("zenith").name == "Butler"
    assert library.get("synthia").traits.formality == pytest.approx(
        0.5 * 0.15 + 0.3 * 0.6 + 0.2 * 1.0
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
        PersonaError, match="no persona 'nobody'; there are glacier, horizon"
    ):
        PersonaLibrary().get("nobody")


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("name = ", "not valid TOML"),
        ('name = "X"\ndescription = "d."\n', "give [traits] or [blend]"),
        (
            'name = "X"\ndescription = "d."\n[traits]\nwit = 0.5\nwarmth = 0.5\n',
            "without a [blend] needs formality, vigilance, verbosity",
        ),
        ('name = "X"\ndescription = "d."\n[blend]\n', "at least one persona"),
        (
            'name = "X"\ndescription = "d."\n[blend]\nstarlight = -1.0\n',
            "blend.starlight",
        ),
        (traits_file("X", 1.5), "traits.warmth"),
        (f"{BLEND_OF_STARLIGHT}[traits]\nwit = 2\n", "traits.wit"),
        (traits_file("X", 0.5) + "mood = 0.5\n", "no such trait: mood"),
        (f"{BLEND_OF_STARLIGHT}[traits]\nmood = 1\n", "no such trait: mood"),
        ('name = ""\ndescription = "d."\n[blend]\nstarlight = 1.0\n', "name"),
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
        'name = "Mix"\ndescription = "d."\n[blend]\nstarlight = 1.0\nnobody = 1.0\n',
    )

    with pytest.raises(
        PersonaError, match="mix blends a persona that does not exist: nobody"
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
