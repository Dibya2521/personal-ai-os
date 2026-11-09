import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from synthia.interfaces.chat import consent_question, terminal_consent
from synthia.interfaces.cli import app
from synthia.kernel.consent import CONSENT_FILE, ConsentBook, Provider, Scope
from tests.consenting import OPENROUTER

runner = CliRunner()
GONE = Provider(name="gone", url="https://gone.example/v1")
LISTED = f"openrouter ({OPENROUTER.url}) chat"
LISTED_PRIVATE = f"openrouter ({OPENROUTER.url}) private"


@pytest.fixture(autouse=True)
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("SYNTHIA_HOME", str(home))
    return home


def acme_at(home: Path, url: str) -> Provider:
    (home / "providers.toml").write_text(
        f'[providers.acme]\nbase_url = "{url}"\nmodel = "m"\ncontext_window = 8192\n',
        encoding="utf-8",
    )
    return Provider(name="acme", url=url)


def test_list_says_nothing_is_decided_at_first(home: Path) -> None:
    result = runner.invoke(app, ["consent", "list"])

    assert result.exit_code == 0, result.output
    assert result.output == (
        f"{LISTED}: not decided, stays on this machine\n"
        f"{LISTED_PRIVATE}: not decided, stays on this machine\n"
    )
    assert not (home / CONSENT_FILE).exists()


def allowed(home: Path) -> tuple[bool, bool]:
    book = ConsentBook(home)
    return book.allows(OPENROUTER, Scope.CHAT), book.allows(OPENROUTER, Scope.PRIVATE)


def test_a_plain_grant_keeps_private_text_on_this_machine(home: Path) -> None:
    runner.invoke(app, ["consent", "grant", "openrouter"])

    assert allowed(home) == (True, False)


def test_a_private_grant_lets_the_chat_and_private_text_go(home: Path) -> None:
    granted = runner.invoke(app, ["consent", "grant", "openrouter", "--private"])
    listed = runner.invoke(app, ["consent", "list"]).output.splitlines()

    assert granted.exit_code == 0, granted.output
    assert "with private memory and file passages" in granted.output
    assert allowed(home) == (True, True)
    assert len(listed) == 2
    assert all(": granted " in line for line in listed)


def test_revoking_private_keeps_the_chat_and_revoking_the_chat_takes_both(
    home: Path,
) -> None:
    runner.invoke(app, ["consent", "grant", "openrouter", "--private"])
    private = runner.invoke(app, ["consent", "revoke", "openrouter", "--private"])
    after_private = allowed(home)
    runner.invoke(app, ["consent", "grant", "openrouter", "--private"])
    runner.invoke(app, ["consent", "revoke", "openrouter"])

    assert "revoked: no private text goes to openrouter" in private.output
    assert after_private == (True, False)
    assert allowed(home) == (False, False)


def test_a_revoke_that_fails_halfway_leaves_no_private_grant(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["consent", "grant", "openrouter", "--private"])
    writes: list[Path] = []

    def second_write_fails(self: Path, target: Path) -> Path:
        writes.append(target)
        if len(writes) == 2:
            raise PermissionError(13, "Permission denied")
        os.replace(self, target)  # noqa: PTH105 - Path.replace is the patched method
        return Path(target)

    monkeypatch.setattr(Path, "replace", second_write_fails)
    revoked = runner.invoke(app, ["consent", "revoke", "openrouter"])

    assert revoked.exit_code == 2
    assert allowed(home) == (True, False)


def test_grant_then_revoke_is_listed_with_who_decided(home: Path) -> None:
    granted = runner.invoke(app, ["consent", "grant", "openrouter"])
    after_grant = ConsentBook(home).allows(OPENROUTER, Scope.CHAT)
    revoked = runner.invoke(app, ["consent", "revoke", "openrouter"])
    listed = runner.invoke(app, ["consent", "list"])

    assert (granted.exit_code, revoked.exit_code) == (0, 0)
    assert f"granted: chats may go to openrouter at {OPENROUTER.url}" in granted.output
    assert "revoked: nothing goes to openrouter" in revoked.output
    assert after_grant
    assert not ConsentBook(home).allows(OPENROUTER, Scope.CHAT)
    assert listed.output.startswith(f"{LISTED}: refused ")
    assert listed.output.endswith(" (cli)\n")


def test_a_grant_does_not_follow_the_name_to_a_new_address(home: Path) -> None:
    first = acme_at(home, "https://api.acme.example/v1")
    runner.invoke(app, ["consent", "grant", "acme"])
    moved = acme_at(home, "https://other-host.example/v1")

    listed = runner.invoke(app, ["consent", "list"]).output.splitlines()
    revoked = runner.invoke(app, ["consent", "revoke", "acme"])

    assert not ConsentBook(home).allows(moved, Scope.CHAT)
    assert f"acme ({moved.url}) chat: not decided, stays on this machine" in listed
    assert any(line.startswith(f"acme ({first.url}) chat: granted ") for line in listed)
    assert revoked.exit_code == 0
    assert not ConsentBook(home).allows(first, Scope.CHAT)


def test_a_decision_for_a_provider_no_longer_configured_is_listed(
    home: Path,
) -> None:
    ConsentBook(home).decide(GONE, Scope.CHAT, granted=True, by="chat")

    listed = runner.invoke(app, ["consent", "list"])
    revoked = runner.invoke(app, ["consent", "revoke", "gone"])

    assert f"gone ({GONE.url}) chat: granted " in listed.output
    assert revoked.exit_code == 0
    assert not ConsentBook(home).allows(GONE, Scope.CHAT)


def test_a_revoke_works_while_providers_toml_is_broken(home: Path) -> None:
    ConsentBook(home).decide(GONE, Scope.CHAT, granted=True, by="chat")
    (home / "providers.toml").write_text("not = [toml", encoding="utf-8")

    revoked = runner.invoke(app, ["consent", "revoke", "gone"])

    assert revoked.exit_code == 0, revoked.output
    assert not ConsentBook(home).allows(GONE, Scope.CHAT)


def test_an_unknown_provider_is_not_granted(home: Path) -> None:
    result = runner.invoke(app, ["consent", "grant", "nobody"])

    assert result.exit_code == 2
    assert "nobody" in result.output
    assert not (home / CONSENT_FILE).exists()


def test_revoking_what_was_never_decided_writes_nothing(home: Path) -> None:
    result = runner.invoke(app, ["consent", "revoke", "nobody"])

    assert result.exit_code == 0
    assert "nothing to revoke" in result.output
    assert not (home / CONSENT_FILE).exists()


def test_a_damaged_file_stops_every_command(home: Path) -> None:
    (home / CONSENT_FILE).write_bytes(b"not json")

    results = [
        runner.invoke(app, ["consent", *command])
        for command in (["list"], ["grant", "openrouter"], ["revoke", "openrouter"])
    ]

    assert [r.exit_code for r in results] == [2, 2, 2]
    assert all("damaged" in r.output for r in results)
    assert (home / CONSENT_FILE).read_bytes() == b"not json"


@pytest.mark.parametrize(
    ("typed", "expected"), [("y", True), (" YES ", True), ("", False), ("n", False)]
)
async def test_the_terminal_says_yes_only_to_y_or_yes(
    typed: str, *, expected: bool
) -> None:
    asked: list[str] = []

    def read(question: str) -> str:
        asked.append(question)
        return typed

    assert await terminal_consent(read)("openrouter", OPENROUTER.url) is expected
    assert asked == [consent_question("openrouter", OPENROUTER.url)]


async def test_the_terminal_closing_is_a_no() -> None:
    def closed(_question: str) -> str:
        raise EOFError

    assert await terminal_consent(closed)("openrouter", OPENROUTER.url) is False


def test_the_question_names_the_provider_its_address_and_the_way_back() -> None:
    question = consent_question("openrouter", OPENROUTER.url)

    assert f"openrouter at {OPENROUTER.url}" in question
    assert "synthia consent revoke openrouter" in question
    assert question.endswith("[y/N] ")
