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
        for scope in Scope:
            where = f"{receiver.name} ({receiver.url}) {scope}"
            consent = decided.get((receiver, scope))
            if consent is None:
                lines.append(f"{where}: {NOT_DECIDED}")
                continue
            answer = "granted" if consent.granted else "refused"
            when = consent.decided_at.astimezone().strftime("%Y-%m-%d %H:%M")
            lines.append(f"{where}: {answer} {when} ({consent.by})")
    return lines


def grant(
    book: ConsentBook,
    provider: str,
    providers: Mapping[str, Provider],
    *,
    private: bool = False,
) -> str:
    """Let chats go to ``provider``, with ``private`` text too; say what changed.

    A private grant grants the chat as well, since private text only ever
    travels inside a chat.

    Raises:
        ConsentError: If the consent file is damaged or cannot be written.
        KeyStoreError: If ``provider`` is not configured.
    """
    require_provider(provider, providers.keys())
    receiver = providers[provider]
    book.decide(receiver, Scope.CHAT, granted=True, by="cli")
    if not private:
        return (
            f"granted: chats may go to {provider} at {receiver.url} once remote is on"
            f"{NEXT_REQUEST}"
        )
    book.decide(receiver, Scope.PRIVATE, granted=True, by="cli")
    return (
        f"granted: chats may go to {provider} at {receiver.url} once remote is on, "
        f"with private memory and file passages{NEXT_REQUEST}"
    )


def revoke(
    book: ConsentBook,
    provider: str,
    providers: Mapping[str, Provider],
    *,
    private: bool = False,
) -> str:
    """Keep chats away from ``provider``, or only ``private`` text; say what changed.

    Revoking the chat revokes private text too. Every address ``provider`` was
    granted for is refused, including one no longer configured.

    Raises:
        ConsentError: If the consent file is damaged or cannot be written.
    """
    # Private first: if the second write fails, no private grant outlives the chat's.
    scopes = (Scope.PRIVATE,) if private else (Scope.PRIVATE, Scope.CHAT)
    current = providers.get(provider)
    refused = [
        consent
        for scope in scopes
        for consent in book.revoke(provider, scope, current=current, by="cli")
    ]
    if not refused:
        return f"nothing to revoke: {provider} was never decided"
    if private:
        return f"revoked: no private text goes to {provider}{NEXT_REQUEST}"
    return f"revoked: nothing goes to {provider}{NEXT_REQUEST}"
