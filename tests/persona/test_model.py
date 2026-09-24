import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.persona.model import (
    HIGH_FROM,
    LOW_BELOW,
    PHRASES,
    Persona,
    PersonaError,
    Traits,
)

MIDDLE = Traits(warmth=0.5, formality=0.5, wit=0.5, vigilance=0.5, verbosity=0.5)


def persona(template: str | None = None) -> Persona:
    extra = {} if template is None else {"template": template}
    return Persona(
        name="Test",
        description="a persona for tests.",
        traits=MIDDLE,
        principles=("Be exact.", "Be kind."),
        **extra,
    )


def test_the_prompt_names_the_persona_and_lists_traits_and_principles() -> None:
    prompt = persona().system_prompt()

    assert prompt.startswith("You are Test, a persona for tests.")
    assert "- Be friendly without fuss." in prompt
    assert prompt.endswith("- Be exact.\n- Be kind.")
    assert prompt.count("\n- ") == len(PHRASES) + 2


@pytest.mark.parametrize(
    ("level", "band"),
    [
        (0.0, 0),
        (LOW_BELOW - 1e-9, 0),
        (LOW_BELOW, 1),
        (HIGH_FROM - 1e-9, 1),
        (HIGH_FROM, 2),
        (1.0, 2),
    ],
)
def test_each_level_falls_in_exactly_one_band(level: float, band: int) -> None:
    traits = MIDDLE.adjusted(wit=level)

    assert traits.sentences()[2] == PHRASES["wit"][band]


def test_moving_a_slider_changes_only_that_sentence() -> None:
    before = persona()
    after = before.adjusted(vigilance=1.0)

    changed = [
        i
        for i, (a, b) in enumerate(
            zip(before.traits.sentences(), after.traits.sentences(), strict=True)
        )
        if a != b
    ]
    assert changed == [3]
    assert before.traits.vigilance == 0.5  # the original is unchanged


def test_an_unknown_trait_names_the_valid_ones() -> None:
    with pytest.raises(PersonaError, match="no such trait: charm; traits are warmth"):
        MIDDLE.adjusted(charm=0.3)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_a_level_outside_zero_to_one_is_refused(value: float) -> None:
    with pytest.raises(PersonaError, match="from 0 to 1"):
        persona().adjusted(wit=value)


def test_a_template_with_an_unknown_placeholder_is_a_clear_error() -> None:
    with pytest.raises(PersonaError, match="bad template placeholder"):
        persona("You are $name and $mood.").system_prompt()


def test_a_template_cannot_reach_into_attributes() -> None:
    # With str.format, "{name.__class__}" would reach into the value's attributes.
    assert persona("{name.__class__} $name").system_prompt() == "{name.__class__} Test"


@given(st.fixed_dictionaries({name: st.floats(0, 1) for name in PHRASES}))
def test_any_valid_sliders_render_one_known_sentence_per_trait(
    levels: dict[str, float],
) -> None:
    sentences = Traits.model_validate(levels).sentences()

    for name, sentence in zip(PHRASES, sentences, strict=True):
        assert sentence in PHRASES[name]
