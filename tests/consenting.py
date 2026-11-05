"""Consent stand-ins and grants for tests that send chats to a remote."""

from pathlib import Path
from typing import Final

from synthia.gateway.providers import OPENROUTER_FREE, RemoteProvider, load_providers
from synthia.kernel.consent import ConsentBook, Scope

OPENROUTER: Final = OPENROUTER_FREE.receiver


def consented(_scope: Scope) -> bool:
    """Consent to every scope, as a person who granted everything would."""
    return True


def refused(_scope: Scope) -> bool:
    """Consent to nothing, as a person who never decided would."""
    return False


def grant(home: Path, *providers: str | RemoteProvider) -> None:
    """Record that each provider may receive chats, as synthia consent grant does.

    A name must already be configured in ``home``, since a grant names the
    address its requests go to.
    """
    book = ConsentBook(home)
    configured = load_providers(home)
    for provider in providers:
        chosen = configured[provider] if isinstance(provider, str) else provider
        book.decide(chosen.receiver, Scope.CHAT, granted=True, by="cli")
