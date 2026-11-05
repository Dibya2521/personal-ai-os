import datetime as dt
import json
import logging
import os
import stat
import tempfile
from pathlib import Path

import pytest

from synthia.kernel.consent import (
    CONSENT_FILE,
    MAX_CONSENT_BYTES,
    MAX_CONSENTS,
    ConsentBook,
    ConsentError,
    Provider,
    Scope,
)

WHEN = dt.datetime(2026, 11, 3, 9, 30, tzinfo=dt.UTC)
OPENROUTER = Provider(name="openrouter", url="https://openrouter.ai/api/v1")
GEMINI = Provider(name="gemini", url="https://gemini.example/v1")


def book(home: Path) -> ConsentBook:
    return ConsentBook(home, clock=lambda: WHEN)


def record(**changes: object) -> dict[str, object]:
    return {
        "receiver": {"kind": "provider", "name": "openrouter", "url": OPENROUTER.url},
        "scope": "chat",
        "granted": True,
        "decided_at": "2026-11-03T09:30:00Z",
        "by": "chat",
    } | changes


def file_of(*records: dict[str, object]) -> bytes:
    return json.dumps({"version": 1, "consents": list(records)}).encode()


def test_nothing_decided_allows_nothing_and_writes_nothing(tmp_path: Path) -> None:
    consents = book(tmp_path)

    assert consents.consents() == ()
    assert not consents.allows(OPENROUTER, Scope.CHAT)
    assert not (tmp_path / CONSENT_FILE).exists()


def test_a_decision_is_kept_and_the_newest_replaces_it(tmp_path: Path) -> None:
    consents = book(tmp_path)

    consents.decide(OPENROUTER, Scope.CHAT, granted=True, by="chat")
    granted = book(tmp_path).allows(OPENROUTER, Scope.CHAT)
    consents.decide(OPENROUTER, Scope.CHAT, granted=False, by="cli")

    assert granted
    (kept,) = book(tmp_path).consents()
    assert (kept.granted, kept.by, kept.decided_at) == (False, "cli", WHEN)
    assert not book(tmp_path).allows(OPENROUTER, Scope.CHAT)


def test_each_receiver_and_scope_is_decided_alone(tmp_path: Path) -> None:
    consents = book(tmp_path)

    consents.decide(OPENROUTER, Scope.CHAT, granted=True, by="cli")

    assert consents.allows(OPENROUTER, Scope.CHAT)
    assert not consents.allows(OPENROUTER, Scope.PRIVATE)
    assert not consents.allows(GEMINI, Scope.CHAT)


def test_a_grant_does_not_follow_the_name_to_another_address(tmp_path: Path) -> None:
    consents = book(tmp_path)
    moved = Provider(name="openrouter", url="https://other-host.example/v1")

    consents.decide(OPENROUTER, Scope.CHAT, granted=True, by="cli")

    assert not consents.allows(moved, Scope.CHAT)


def test_a_revoke_refuses_every_address_of_the_name(tmp_path: Path) -> None:
    consents = book(tmp_path)
    moved = Provider(name="openrouter", url="https://other-host.example/v1")
    current = Provider(name="openrouter", url="https://third.example/v1")
    consents.decide(OPENROUTER, Scope.CHAT, granted=True, by="cli")
    consents.decide(moved, Scope.CHAT, granted=True, by="chat")
    consents.decide(GEMINI, Scope.CHAT, granted=True, by="cli")
    consents.decide(OPENROUTER, Scope.PRIVATE, granted=True, by="cli")

    refused = consents.revoke("openrouter", Scope.CHAT, current=current, by="cli")

    assert [c.receiver for c in refused] == [OPENROUTER, moved, current]
    assert not any(consents.allows(r, Scope.CHAT) for r in (OPENROUTER, moved))
    assert consents.allows(GEMINI, Scope.CHAT)
    assert consents.allows(OPENROUTER, Scope.PRIVATE)
    assert len(consents.consents()) == 5


def test_revoking_a_name_never_decided_writes_nothing(tmp_path: Path) -> None:
    assert book(tmp_path).revoke("openrouter", Scope.CHAT, by="cli") == ()
    assert not (tmp_path / CONSENT_FILE).exists()


def test_a_full_book_refuses_one_more_decision(tmp_path: Path) -> None:
    (tmp_path / CONSENT_FILE).write_bytes(
        file_of(
            *(
                record(receiver={"name": f"p{i}", "url": OPENROUTER.url})
                for i in range(MAX_CONSENTS)
            )
        )
    )

    with pytest.raises(ConsentError, match=f"holds {MAX_CONSENTS} decisions"):
        book(tmp_path).decide(GEMINI, Scope.CHAT, granted=False, by="cli")


def test_the_file_holds_only_the_decisions(tmp_path: Path) -> None:
    book(tmp_path).decide(OPENROUTER, Scope.CHAT, granted=True, by="chat")

    data = json.loads((tmp_path / CONSENT_FILE).read_bytes())

    assert data == {"version": 1, "consents": [record()]}


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_only_the_owner_can_read_the_file(tmp_path: Path) -> None:
    book(tmp_path).decide(OPENROUTER, Scope.CHAT, granted=True, by="chat")

    mode = stat.S_IMODE((tmp_path / CONSENT_FILE).stat().st_mode)

    assert mode == 0o600


DAMAGED = {
    "not json": b"not json",
    "version 2": b'{"version": 2, "consents": []}',
    "extra field": b'{"version": 1, "consents": [], "extra": 1}',
    "bad name": file_of(
        record(receiver={"kind": "provider", "name": "Bad Name", "url": "u"})
    ),
    "name starting with a hyphen": file_of(
        record(receiver={"kind": "provider", "name": "-x", "url": "u"})
    ),
    "no address": file_of(record(receiver={"kind": "provider", "name": "openrouter"})),
    "naive time": file_of(record(decided_at="2026-11-03T09:30:00")),
    "granted as text": file_of(record(granted="true")),
    "granted as a number": file_of(record(granted=1)),
    "version as text": b'{"version": "1", "consents": []}',
    "a yes and a later no for one receiver": file_of(
        record(), record(granted=False, decided_at="2026-11-04T09:30:00Z")
    ),
}


@pytest.mark.parametrize("data", DAMAGED.values(), ids=DAMAGED.keys())
def test_a_damaged_file_allows_nothing_and_says_so(
    tmp_path: Path, data: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / CONSENT_FILE
    path.write_bytes(data)
    consents = book(tmp_path)

    with pytest.raises(ConsentError, match="damaged"):
        consents.consents()
    with caplog.at_level(logging.WARNING):
        allowed = consents.allows(OPENROUTER, Scope.CHAT)
    with pytest.raises(ConsentError, match="damaged"):
        consents.decide(OPENROUTER, Scope.CHAT, granted=True, by="cli")

    assert not allowed
    assert "damaged" in caplog.text
    assert path.read_bytes() == data


def test_an_oversize_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / CONSENT_FILE).write_bytes(b" " * (MAX_CONSENT_BYTES + 1))

    with pytest.raises(ConsentError, match="larger than"):
        book(tmp_path).consents()


def test_an_unreadable_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / CONSENT_FILE).mkdir()

    with pytest.raises(ConsentError, match="cannot read"):
        book(tmp_path).consents()
    assert not book(tmp_path).allows(OPENROUTER, Scope.CHAT)


def test_an_unwritable_home_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)

    with pytest.raises(ConsentError, match="cannot write"):
        book(tmp_path).decide(OPENROUTER, Scope.CHAT, granted=True, by="cli")
    assert not (tmp_path / CONSENT_FILE).exists()
