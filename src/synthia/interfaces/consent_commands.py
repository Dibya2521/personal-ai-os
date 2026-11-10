"""What ``synthia consent`` does: list, grant and revoke what may leave the machine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

from synthia.interfaces.key_commands import require_provider
from synthia.interfaces.mcp_hosts import HOSTS_FILE, HostEntry
from synthia.kernel.consent import ConsentBook, Host, Provider, Scope

if TYPE_CHECKING:
    from collections.abc import Mapping

    from synthia.kernel.consent import Consent

NOT_DECIDED: Final = "not decided, stays on this machine"
NEXT_REQUEST: Final = "; a running chat follows it from its next request"


def _label(receiver: Provider | Host) -> str:
    match receiver:
        case Provider():
            return f"{receiver.name} ({receiver.url})"
        case Host():
            return f"host {receiver.name}"
        case _:
            assert_never(receiver)


def consent_lines(
    book: ConsentBook,
    providers: Mapping[str, Provider],
    hosts: Mapping[str, HostEntry],
) -> list[str]:
    """Return one line per receiver and scope saying what was decided, and when.

    ``providers`` maps each configured provider's name to its receiver. A
    decision for an address no longer configured gets a line too, so it can be
    seen and revoked. Each host in ``hosts`` (``mcp_hosts.toml``) or in the
    book also gets a line saying what it may see; a host is decided only for
    private text.

    Raises:
        ConsentError: If the consent file cannot be read or is damaged.
    """
    decided = {(c.receiver, c.scope): c for c in book.consents()}
    receivers = list(
        dict.fromkeys(
            [
                *providers.values(),
                *(Host(name=name) for name in hosts),
                *(receiver for receiver, _ in decided),
            ]
        )
    )
    lines: list[str] = []
    for receiver in receivers:
        if isinstance(receiver, Host):
            lines.append(_sees(receiver, hosts.get(receiver.name, HostEntry())))
        scopes = (Scope.PRIVATE,) if isinstance(receiver, Host) else tuple(Scope)
        lines.extend(
            _decided(f"{_label(receiver)} {scope}", decided.get((receiver, scope)))
            for scope in scopes
        )
    return lines


def _sees(host: Host, entry: HostEntry) -> str:
    collections = ", ".join(entry.collections) or "none"
    memory = "yes" if entry.memory else "no"
    return f"host {host.name} sees: collections {collections}; memory {memory}"


def _decided(where: str, consent: Consent | None) -> str:
    if consent is None:
        return f"{where}: {NOT_DECIDED}"
    answer = "granted" if consent.granted else "refused"
    when = consent.decided_at.astimezone().strftime("%Y-%m-%d %H:%M")
    return f"{where}: {answer} {when} ({consent.by})"


def grant_host(book: ConsentBook, host: Host) -> str:
    """Let private passages and turns go to ``host``; say what changed.

    Raises:
        ConsentError: If the consent file is damaged or cannot be written.
    """
    book.decide(host, Scope.PRIVATE, granted=True, by="cli")
    return (
        f"granted: private memory and file passages may go to host {host.name}, "
        f"within what {HOSTS_FILE} shares with it, from its next call"
    )


def revoke_host(book: ConsentBook, host: Host) -> str:
    """Keep private passages and turns from ``host``; say what changed.

    Raises:
        ConsentError: If the consent file is damaged or cannot be written.
    """
    book.decide(host, Scope.PRIVATE, granted=False, by="cli")
    return f"revoked: no private text goes to host {host.name}, from its next call"


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
