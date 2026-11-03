from typing import override

from keyring.backend import KeyringBackend
from keyring.errors import KeyringLocked, PasswordDeleteError


class MemoryKeyring(KeyringBackend):
    """A keyring backend in a dict, so no test reaches the real credential store."""

    priority = 1  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]

    def __init__(self) -> None:
        super().__init__()
        self.saved: dict[tuple[str, str], str] = {}

    @override
    def get_password(self, service: str, username: str) -> str | None:
        return self.saved.get((service, username))

    @override
    def set_password(self, service: str, username: str, password: str) -> None:
        self.saved[service, username] = password

    @override
    def delete_password(self, service: str, username: str) -> None:
        try:
            del self.saved[service, username]
        except KeyError:
            raise PasswordDeleteError(username) from None


class LockedKeyring(MemoryKeyring):
    """A backend whose every call fails, with the secret in its message.

    ``error`` is the exception type raised: keyring's own, or any other, as
    the Windows backend raises ``pywintypes.error``.
    """

    def __init__(self, leaked: str, error: type[Exception] = KeyringLocked) -> None:
        super().__init__()
        self.leaked = leaked
        self.error = error

    @override
    def get_password(self, service: str, username: str) -> str | None:
        raise self.error(self.leaked)

    @override
    def set_password(self, service: str, username: str, password: str) -> None:
        raise self.error(password)

    @override
    def delete_password(self, service: str, username: str) -> None:
        raise self.error(self.leaked)
