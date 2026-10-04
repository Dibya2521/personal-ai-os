from pathlib import Path

import pytest

from synthia.gateway.types import Reasoning
from synthia.interfaces.commands import (
    DEFAULT_IMAGE_QUESTION,
    AdjustPersona,
    Command,
    Exit,
    Forget,
    Help,
    Invalid,
    KeepPrivate,
    Plan,
    Reset,
    Say,
    ShowBudget,
    ShowImage,
    ShowModel,
    SwitchPersona,
    Think,
    UseRemote,
    parse,
)


@pytest.mark.parametrize(
    ("line", "command"),
    [
        ("  hello there  ", Say("hello there")),
        ("", Say("")),
        ("what does /budget do?", Say("what does /budget do?")),
        ("/budget", ShowBudget()),
        ("/model", ShowModel()),
        ("/reset", Reset()),
        ("/help", Help()),
        ("/exit", Exit()),
        ("/quit", Exit()),
        ("/persona", SwitchPersona("")),
        ("/persona horizon", SwitchPersona("horizon")),
        ("/persona set wit=0.3", AdjustPersona({"wit": 0.3})),
        ("/persona set wit=0.3  warmth=1", AdjustPersona({"wit": 0.3, "warmth": 1.0})),
        ("/image cat.png", ShowImage(Path("cat.png"), DEFAULT_IMAGE_QUESTION)),
        ("/image cat.png  is it asleep? ", ShowImage(Path("cat.png"), "is it asleep?")),
        (
            '/image "My Pictures/cat 1.png" is it asleep?',
            ShowImage(Path("My Pictures/cat 1.png"), "is it asleep?"),
        ),
        ("/think", Think(None)),
        ("/think high", Think(Reasoning.HIGH)),
        ("/think  OFF ", Think(Reasoning.OFF)),
        ("/think auto", Think(Reasoning.AUTO)),
        ("/remote", UseRemote(None)),
        ("/remote on", UseRemote(on=True)),
        ("/remote  OFF ", UseRemote(on=False)),
        ("/private", KeepPrivate(None)),
        ("/private ON", KeepPrivate(on=True)),
        ("/private off", KeepPrivate(on=False)),
        ("/forget", Forget()),
        ("/plan find a, then use it ", Plan("find a, then use it")),
    ],
)
def test_each_command_parses(line: str, command: Command) -> None:
    assert parse(line) == command


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("/dance", "unknown command /dance; /help lists them"),
        ("/budget now", "/budget takes no arguments"),
        ("/persona horizon please", "a persona name has no spaces"),
        ("/persona set", "at least one trait=number"),
        ("/persona set wit", "'wit' is not trait=number"),
        ("/persona set wit=high", "'wit=high' is not trait=number"),
        ("/image", "/image needs a path"),
        ('/image "cat.png is it asleep?', "no closing quote"),
        ('/image "" what?', "/image needs a path"),
        ("/think max", "'max' is not a thinking level; use one of off, low, medium"),
        ("/think high now", "'high now' is not a thinking level"),
        ("/remote yes", "'yes' is not on or off"),
        ("/remote on now", "'on now' is not on or off"),
        ("/private please", "'please' is not on or off"),
        ("/forget everything", "/forget takes no arguments"),
        ("/plan", "/plan needs a task"),
        ("/plan    ", "/plan needs a task"),
    ],
)
def test_mistakes_say_what_is_wrong(line: str, reason: str) -> None:
    result = parse(line)

    assert isinstance(result, Invalid)
    assert reason in result.reason
