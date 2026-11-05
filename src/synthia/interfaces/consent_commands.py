"""What ``synthia consent`` does: list, grant and revoke what may leave the machine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from synthia.interfaces.key_commands import require_provider
from synthia.kernel.consent import ConsentBook, Provider, Scope

if TYPE_CHECKING:
    from collections.abc import Mapping

NOT_DECIDED: Final = "not decided, stays on this machine"
NEXT_REQUEST: Final = "; a running chat follows it from its next request"


def consent_lines(book: ConsentBook, providers: Mapping[str, Provider]) -> list[str]:
    """Return one line per receiver and scope saying what was decided, and when.

    ``providers`` maps each configured provider's name to its receiver. A
    decision for an address no longer configured gets a line too, so it can be
    seen and revoked.

    Raises:
        ConsentError: If the consent file cannot be read or is damaged.
    """
    decided = {(c.receiver, c.scope): c for c in book.consents()}
    receivers = list(
        dict.fromkeys([*providers.values(), *(receiver for receiver, _ in decided)])
    )
    lines: list[str] = []
    for receiver in receivers:
        where = f"{receiver.name} ({receiver.url}) chat"
        consent = decided.get((receiver, Scope.CHAT))
        if consent is None:
            lines.append(f"{where}: {NOT_DECIDED}")
            continue
        answer = "granted" if consent.granted else "refused"
        when = consent.decided_at.astimezone().strftime("%Y-%m-%d %H:%M")
        lines.append(f"{where}: {answer} {when} ({consent.by})")
    return lines


def grant(book: ConsentBook, provider: str, providers: Mapping[str, Provider]) -> str:
    """Let this chat's text go to ``provider``; return what to tell the person.

    Raises:
        ConsentError: If the consent file is damaged or cannot be written.
        KeyStoreError: If ``provider`` is not configured.
    """
    require_provider(provider, providers.keys())
    receiver = providers[provider]
    book.decide(receiver, Scope.CHAT, granted=True, by="cli")
    return (
        f"granted: chats may go to {provider} at {receiver.url} once remote is on"
        f"{NEXT_REQUEST}"
    )


def revoke(book: ConsentBook, provider: str, providers: Mapping[str, Provider]) -> str:
    """Keep every chat away from ``provider``; return what to tell the person.

    Every address ``provider`` was granted for is refused, including one no
    longer configured.

    Raises:
        ConsentError: If the consent file is damaged or cannot be written.
    """
    refused = book.revoke(
        provider, Scope.CHAT, current=providers.get(provider), by="cli"
    )
    if not refused:
        return f"nothing to revoke: {provider} was never decided"
    return f"revoked: nothing goes to {provider}{NEXT_REQUEST}"
