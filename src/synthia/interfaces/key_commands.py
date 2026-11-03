"""What ``synthia keys`` does: store, list and remove provider keys.

No part of a key is ever shown. ``list`` says only where each provider's key
comes from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from synthia.gateway.providers import key_variable
from synthia.kernel.keys import KeyStore, KeyStoreError, check_key

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

    from pydantic import SecretStr

IN_STORE: Final = "key store"
IN_ENV: Final = ".env or environment"
NO_KEY: Final = "no key"
DAEMON_KEEPS_KEYS: Final = (
    "; a running daemon keeps the keys it started with until synthia stop"
)


def key_lines(
    providers: Iterable[str],
    stored: AbstractSet[str],
    from_env: Mapping[str, SecretStr],
) -> list[str]:
    """Return one line per provider saying where its key comes from.

    ``stored`` holds the variables the key store gives a key for. A stored key
    no configured provider uses gets a line of its own, so it can be removed.
    """
    lines: list[str] = []
    used: set[str] = set()
    for name in providers:
        variable = key_variable(name)
        used.add(variable)
        if variable in stored:
            shadowed = (
                f", {variable} in {IN_ENV} unused" if variable in from_env else ""
            )
            lines.append(f"{name}: {IN_STORE}{shadowed}")
        elif variable in from_env:
            lines.append(f"{name}: {IN_ENV}")
        else:
            lines.append(f"{name}: {NO_KEY}")
    lines.extend(
        f"{variable}: {IN_STORE}, no provider uses it"
        for variable in sorted(stored - used)
    )
    return lines


def require_provider(provider: str, known: AbstractSet[str]) -> None:
    """Raise unless ``provider`` is configured, before its key is asked for.

    Raises:
        KeyStoreError: If no provider has that name.
    """
    if provider not in known:
        message = (
            f"no provider is named {provider!r}; the providers are "
            f"{', '.join(sorted(known))}"
        )
        raise KeyStoreError(message)


def set_key(store: KeyStore, provider: str, typed: str) -> str:
    """Store ``typed`` as ``provider``'s key and return what to tell the person.

    Raises:
        KeyStoreError: If the key cannot be a key or the store cannot be
            written.
    """
    store.set(key_variable(provider), check_key(typed.strip()))
    return f"stored {provider}'s key in the system key store{DAEMON_KEEPS_KEYS}"


def remove_key(
    store: KeyStore, provider: str, from_env: Mapping[str, SecretStr]
) -> str:
    """Delete ``provider``'s stored key and return what to tell the person.

    Raises:
        KeyStoreError: If the name cannot be a provider's or the store cannot
            be written.
    """
    variable = key_variable(provider)
    held = store.remove(variable)
    said = (
        f"removed {provider}'s key from the system key store"
        if held
        else f"the system key store held no key for {provider}"
    )
    if variable in from_env:
        said += f"; {variable} in {IN_ENV} is still used"
    return said + DAEMON_KEEPS_KEYS if held else said
