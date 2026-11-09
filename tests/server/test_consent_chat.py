import tempfile
from pathlib import Path

import pytest

from synthia.gateway.assemble import Gateway
from synthia.gateway.router import Remote, Router
from synthia.gateway.usage import UsageLog
from synthia.interfaces.chat import NO_CONSENT_LOCAL, STAYS_LOCAL, ChatApp
from synthia.interfaces.commands import parse
from synthia.kernel.consent import CONSENT_FILE, ConsentBook, Provider, Scope
from synthia.persona.library import PersonaLibrary
from synthia.server.conversation import REMOTE_ON_PRIVATE, Conversation
from synthia.server.session import ChatSession, LastRoute
from tests.consenting import OPENROUTER
from tests.gateway.test_router import LOCAL, REMOTE, Fake, Setup
from tests.interfaces.test_chat import console


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


class Asker:
    def __init__(self, answer: bool | None) -> None:
        self.answer = answer
        self.asked: list[Provider] = []

    async def __call__(self, receiver: Provider) -> bool | None:
        self.asked.append(receiver)
        return self.answer


async def nothing_learned() -> int | None:
    return None


def talk(tmp_path: Path, setup: Setup, asker: Asker) -> Conversation:
    book, routes = ConsentBook(tmp_path), LastRoute()
    router = Router(
        Fake(LOCAL),
        Remote(
            Fake(REMOTE),
            setup.health,
            lambda scope: book.allows(OPENROUTER, scope),
            OPENROUTER,
        ),
        routes,
    )
    session = ChatSession(router, PersonaLibrary(), "synthia", routes)
    gateway = Gateway(
        router, router, UsageLog(tmp_path / "usage.db"), nothing_learned, book
    )
    return Conversation(session, gateway, asker)


async def test_a_yes_turns_remote_on_and_is_saved(tmp_path: Path, setup: Setup) -> None:
    asker = Asker(answer=True)
    chat = talk(tmp_path, setup, asker)

    first = await chat.remote(on=True)
    again = await chat.remote(on=True)

    assert first.notes == ("remote: on, turns may leave this machine",)
    assert again.notes == first.notes
    assert asker.asked == [OPENROUTER]
    (saved,) = ConsentBook(tmp_path).consents()
    assert (saved.granted, saved.by, saved.scope) == (True, "chat", Scope.CHAT)


async def test_a_no_keeps_remote_off_and_is_saved(tmp_path: Path, setup: Setup) -> None:
    chat = talk(tmp_path, setup, Asker(answer=False))

    reply = await chat.remote(on=True)

    assert reply.notes == (
        "remote: stays off, openrouter refused; every turn stays on this machine",
    )
    assert not chat.session.use_remote
    (saved,) = ConsentBook(tmp_path).consents()
    assert not saved.granted


async def test_a_client_that_cannot_ask_saves_nothing(
    tmp_path: Path, setup: Setup
) -> None:
    chat = talk(tmp_path, setup, Asker(answer=None))

    reply = await chat.remote(on=True)

    assert reply.errors == (
        (
            "remote: stays off, openrouter has no consent and this client cannot "
            "ask; run synthia consent grant openrouter"
        ),
    )
    assert not chat.session.use_remote
    assert not (tmp_path / CONSENT_FILE).exists()


async def test_a_grant_from_the_cli_is_not_asked_again(
    tmp_path: Path, setup: Setup
) -> None:
    ConsentBook(tmp_path).decide(OPENROUTER, Scope.CHAT, granted=True, by="cli")
    asker = Asker(answer=False)

    reply = await talk(tmp_path, setup, asker).remote(on=True)

    assert reply.notes == ("remote: on, turns may leave this machine",)
    assert asker.asked == []


async def test_a_damaged_file_is_reported_and_not_asked(
    tmp_path: Path, setup: Setup
) -> None:
    (tmp_path / CONSENT_FILE).write_bytes(b"not json")
    asker = Asker(answer=True)
    chat = talk(tmp_path, setup, asker)

    reply = await chat.remote(on=True)

    (problem,) = reply.errors
    assert "damaged" in problem
    assert asker.asked == []
    assert not chat.session.use_remote


async def test_a_write_that_fails_is_reported(
    tmp_path: Path, setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)
    chat = talk(tmp_path, setup, Asker(answer=True))

    reply = await chat.remote(on=True)

    (problem,) = reply.errors
    assert "cannot write" in problem
    assert not chat.session.use_remote


async def test_a_revoke_mid_chat_keeps_the_next_turn_local_and_says_so_once(
    tmp_path: Path, setup: Setup
) -> None:
    chat = talk(tmp_path, setup, Asker(answer=True))
    screen, out = console()
    app = ChatApp(chat, screen)
    await chat.remote(on=True)

    await app.handle(parse("before"))
    ConsentBook(tmp_path).decide(OPENROUTER, Scope.CHAT, granted=False, by="cli")
    await app.handle(parse("after"))
    await app.handle(parse("again"))

    text = out.getvalue()
    assert f"remote | {REMOTE.id}" in text
    assert f"local | {LOCAL.id}" in text
    assert text.count(NO_CONSENT_LOCAL) == 1


@pytest.mark.parametrize("private", [False, True])
async def test_a_confined_chat_goes_remote_only_after_a_private_grant(
    tmp_path: Path, setup: Setup, *, private: bool
) -> None:
    book = ConsentBook(tmp_path)
    book.decide(OPENROUTER, Scope.CHAT, granted=True, by="cli")
    if private:
        book.decide(OPENROUTER, Scope.PRIVATE, granted=True, by="cli")
    chat = talk(tmp_path, setup, Asker(answer=False))
    chat.session.confinement.confine("a private note was read")
    screen, out = console()

    reply = await chat.remote(on=True)
    await ChatApp(chat, screen).handle(parse("hi"))

    text = out.getvalue()
    assert (reply.notes == (REMOTE_ON_PRIVATE,)) is private
    assert (f"remote | {REMOTE.id}" in text) is private
    assert (STAYS_LOCAL in text) is not private


async def test_a_confined_chat_with_remote_off_is_not_told_it_is_held(
    tmp_path: Path, setup: Setup
) -> None:
    book = ConsentBook(tmp_path)
    book.decide(OPENROUTER, Scope.CHAT, granted=True, by="cli")
    book.decide(OPENROUTER, Scope.PRIVATE, granted=True, by="cli")
    chat = talk(tmp_path, setup, Asker(answer=False))
    chat.session.confinement.confine("a private note was read")
    screen, out = console()

    await ChatApp(chat, screen).handle(parse("hi"))

    assert f"local | {LOCAL.id}" in out.getvalue()
    assert STAYS_LOCAL not in out.getvalue()
