"""Provider keys kept in the operating system's credential store.

Each key is one credential (Windows Credential Manager, the macOS Keychain or
the Secret Service on Linux) under the service ``synthia``, named by the
variable that would otherwise hold it, such as ``SYNTHIA_OPENAI_API_KEY``. The
store cannot list its credentials, so ``SYNTHIA_HOME/keys.json`` records which
variables are stored, names only. An empty or missing index means the store is
never opened, so a person who keeps keys in ``.env`` never touches it.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, final

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from synthia.kernel.errors import ConfigError

if TYPE_CHECKING:
    from keyring.backend import KeyringBackend

KEY_SERVICE: Final = "synthia"
KEYS_INDEX_FILE: Final = "keys.json"
KEYS_INDEX_VERSION: Final = 1
MAX_KEYS_INDEX_BYTES: Final = 65_536
# Windows Credential Manager holds at most 2560 bytes, and keyring stores the
# key there as UTF-16.
MAX_KEY_CHARS: Final = 1024
KEY_VARIABLE: Final = r"^SYNTHIA_[A-Z0-9_]{1,32}_API_KEY$"
# A key travels in an HTTP header, where a space or control character would
# either be refused or split the header.
_KEY_TEXT: Final = re.compile(r"^[\x21-\x7e]+$")


class KeyStoreError(ConfigError):
    """The credential store or its index could not be read or written."""


class Credentials(Protocol):
    """Where the key values live; raises :class:`KeyStoreError` on any failure."""

    def get(self, variable: str) -> str | None:
        """Return the stored value, or ``None`` when there is none."""
        ...

    def put(self, variable: str, value: str) -> None:
        """Store ``value``, replacing any earlier one."""
        ...

    def delete(self, variable: str) -> bool:
        """Delete the stored value; return whether there was one."""
        ...


@final
class SystemCredentials:
    """The operating system's credential store, through ``keyring``.

    The backend is chosen on first use, because choosing it can mean
    connecting to the Secret Service, and most commands never need a key.
    """

    def __init__(self) -> None:
        self._backend: KeyringBackend | None = None

    # A backend is a plugin that can raise anything (the Windows one raises
    # pywintypes.error) with the key in its message, so every error becomes a
    # KeyStoreError naming only its type, raised outside the except block so
    # not even __context__ keeps the original.

    def get(self, variable: str) -> str | None:
        """Return the stored value, or ``None`` when there is none."""
        try:
            return self._keyring().get_password(KEY_SERVICE, variable)
        except Exception as error:  # noqa: BLE001
            failure = type(error).__name__
        raise _unavailable(action="read", failure=failure)

    def put(self, variable: str, value: str) -> None:
        """Store ``value``, replacing any earlier one."""
        try:
            # keyring's decorator on set_password drops its signature.
            self._keyring().set_password(KEY_SERVICE, variable, value)  # pyright: ignore[reportUnknownMemberType]
        except Exception as error:  # noqa: BLE001
            failure = type(error).__name__
        else:
            return
        raise _unavailable(action="write", failure=failure)

    def delete(self, variable: str) -> bool:
        """Delete the stored value; return whether there was one."""
        import keyring.errors  # noqa: PLC0415

        try:
            self._keyring().delete_password(KEY_SERVICE, variable)
        except keyring.errors.PasswordDeleteError:
            return False
        except Exception as error:  # noqa: BLE001
            failure = type(error).__name__
        else:
            return True
        raise _unavailable(action="delete from", failure=failure)

    def _keyring(self) -> KeyringBackend:
        if self._backend is None:
            import keyring  # noqa: PLC0415

            self._backend = keyring.get_keyring()
        return self._backend


def _unavailable(*, action: str, failure: str) -> KeyStoreError:
    return KeyStoreError(
        f"could not {action} the system key store ({failure}); unlock it and "
        "try again, or delete keys.json in SYNTHIA_HOME to stop using the store "
        "and set each key in .env as its SYNTHIA_<NAME>_API_KEY variable"
    )


class _Index(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    variables: tuple[str, ...] = Field(max_length=256)


def check_key(value: str) -> SecretStr:
    """Return ``value`` as a key, or raise when it cannot be one.

    Raises:
        KeyStoreError: If it is empty, longer than ``MAX_KEY_CHARS``, or holds
            a space, a control character or a character outside ASCII. The
            message never contains the value.
    """
    if not value:
        message = "the key is empty"
        raise KeyStoreError(message)
    if len(value) > MAX_KEY_CHARS:
        message = f"the key is longer than {MAX_KEY_CHARS} characters"
        raise KeyStoreError(message)
    if _KEY_TEXT.fullmatch(value) is None:
        message = (
            "the key holds a space, a control character or a character outside "
            "ASCII, which no provider key does"
        )
        raise KeyStoreError(message)
    return SecretStr(value)


@final
class KeyStore:
    """Provider keys in ``credentials``, indexed in ``home/keys.json``."""

    def __init__(self, home: Path, credentials: Credentials) -> None:
        self._index = home / KEYS_INDEX_FILE
        self._credentials = credentials

    def names(self) -> tuple[str, ...]:
        """Return the variables the index lists, without opening the store.

        Raises:
            KeyStoreError: If the index cannot be read or is damaged.
        """
        try:
            with self._index.open("rb") as file:
                data = file.read(MAX_KEYS_INDEX_BYTES + 1)
        except FileNotFoundError:
            return ()
        except OSError as error:
            message = f"cannot read {self._index}: {error.strerror}"
            raise KeyStoreError(message) from None
        if len(data) > MAX_KEYS_INDEX_BYTES:
            message = f"{self._index} is larger than {MAX_KEYS_INDEX_BYTES} bytes"
            raise KeyStoreError(message)
        try:
            index = _Index.model_validate_json(data)
        except ValidationError:
            raise self._damaged() from None
        if any(re.fullmatch(KEY_VARIABLE, v) is None for v in index.variables):
            raise self._damaged()
        return index.variables

    def load(self) -> dict[str, SecretStr]:
        """Return every indexed key the store still holds, by its variable.

        Raises:
            KeyStoreError: If the index is damaged or the store cannot be read.
        """
        keys: dict[str, SecretStr] = {}
        for variable in self.names():
            value = self._credentials.get(variable)
            if value:
                keys[variable] = SecretStr(value)
        return keys

    def set(self, variable: str, key: SecretStr) -> None:
        """Store ``key`` as ``variable``, replacing any earlier one.

        The credential is written before the index, so a crash between the two
        leaves an unlisted credential, which the next ``set`` lists.

        Raises:
            KeyStoreError: If ``variable`` is not a key variable, or the store
                or the index cannot be written.
        """
        _check_variable(variable)
        self._credentials.put(variable, key.get_secret_value())
        names = self.names()
        if variable not in names:
            self._write((*names, variable))

    def remove(self, variable: str) -> bool:
        """Delete ``variable``'s key; return whether the store held one.

        Raises:
            KeyStoreError: If ``variable`` is not a key variable, or the store
                or the index cannot be written.
        """
        _check_variable(variable)
        held = self._credentials.delete(variable)
        names = self.names()
        if variable in names:
            self._write(tuple(n for n in names if n != variable))
        return held

    def _write(self, variables: tuple[str, ...]) -> None:
        text = json.dumps(
            {"version": KEYS_INDEX_VERSION, "variables": sorted(variables)},
            indent=2,
        )
        try:
            self._index.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=self._index.parent,
                prefix=self._index.name,
                suffix=".part",
                delete_on_close=False,
            ) as part:
                part.write(text + "\n")
                part.close()
                Path(part.name).replace(self._index)
        except OSError as error:
            message = f"cannot write {self._index}: {error.strerror}"
            raise KeyStoreError(message) from None

    def _damaged(self) -> KeyStoreError:
        return KeyStoreError(
            f"{self._index} is damaged; delete it and store each key again "
            "with synthia keys set <provider>"
        )


def _check_variable(variable: str) -> None:
    if re.fullmatch(KEY_VARIABLE, variable) is None:
        message = f"{variable!r} is not a SYNTHIA_<NAME>_API_KEY variable"
        raise KeyStoreError(message)
