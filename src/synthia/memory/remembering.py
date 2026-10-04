"""One conversation's memory: what it keeps, and what it is told to forget.

The conversation's row is made at its first remembered turn, so a client that
only asks the daemon's status leaves nothing behind. A turn that cannot be
written is logged and the conversation goes on, as a trace that cannot be
written does: memory is kept beside the answers, never in their way.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synthia.memory.store import MemoryStore, Turn

logger = logging.getLogger(__name__)


class Remembering:
    """Keep one conversation's turns in ``store`` unless it is private."""

    def __init__(self, store: MemoryStore) -> None:
        self.private = False
        self._store = store
        self._conversation: int | None = None
        self._last: int | None = None

    async def remember(self, turn: Turn) -> None:
        """Keep ``turn``, unless the conversation is private."""
        if self.private:
            return
        try:
            if self._conversation is None:
                self._conversation = await self._store.begin_conversation(
                    turn.persona, turn.at
                )
            self._last = await self._store.add_turn(self._conversation, turn)
        except (sqlite3.Error, OSError):
            logger.exception("a turn was not remembered")

    async def forget_last(self) -> bool:
        """Forget the last remembered turn; False if there is none to forget.

        Raises:
            sqlite3.Error: If the store could not be changed.
            OSError: If its file could not be opened.
        """
        if self._last is None:
            return False
        forgotten = await self._store.forget_turn(self._last)
        self._last = None
        return forgotten

    def begin_again(self) -> None:
        """Remember the turns from here on as a new conversation."""
        self._conversation = None
        self._last = None
