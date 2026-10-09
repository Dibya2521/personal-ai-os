"""One conversation's memory: what it keeps, and what it is told to forget.

The conversation's row is made at its first remembered turn, so a client that
only asks the daemon's status leaves nothing behind. A turn that cannot be
written is logged and the conversation goes on, as a trace that cannot be
written does: memory is kept beside the answers, never in their way. The same
holds for the vector of a turn's meaning, made after the turn is written; a
turn left without one is embedded at the daemon's next start, and for the
reminder of earlier conversations put before each question. With facts being
learned, each remembered turn also waits for the background learner, which is
woken, never awaited.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Final

from synthia.memory.hybrid import CANDIDATES
from synthia.memory.recall import local_zone, shown
from synthia.memory.store import Asked

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import tzinfo

    from synthia.memory.facts import FactLearning
    from synthia.memory.hybrid import Recall
    from synthia.memory.store import Turn

logger = logging.getLogger(__name__)

REMINDED: Final = 3
# Precision first: on the recall set the right turn reaches it for 25 % of
# questions, an unrelated one for 6.7 %; misses are left to search_memory.
REMINDER_FLOOR: Final = 0.55
REMINDER_HEADER: Final = (
    "From SYNTHIA's memory of earlier conversations, not written by the "
    "person; use it only if it helps:"
)


class Remembering:
    """Keep one conversation's turns in ``recall``'s store unless it is private."""

    def __init__(self, recall: Recall, learning: FactLearning | None = None) -> None:
        """Remember into ``recall``; with ``learning``, learn facts from each turn."""
        self.private = False
        self._recall = recall
        self._store = recall.store
        self._learning = learning
        self._conversation: int | None = None
        self._last: int | None = None

    async def remember(self, turn: Turn) -> None:
        """Keep ``turn`` and its meaning, unless the conversation is private."""
        if self.private:
            return
        try:
            if self._conversation is None:
                self._conversation = await self._store.begin_conversation(
                    turn.persona, turn.at
                )
            self._last = await self._store.add_turn(
                self._conversation, turn, to_learn=self._learning is not None
            )
        except (sqlite3.Error, OSError):
            logger.exception("a turn was not remembered")
            return
        if self._learning is not None:
            self._learning.wake()
        try:
            await self._recall.keep(Asked(self._last, turn.at, turn.question))
        except Exception:
            logger.exception("a turn's meaning was not kept")

    async def forget_last(self) -> bool:
        """Forget the last remembered turn; False if there is none to forget.

        Raises:
            sqlite3.Error: If the store could not be changed.
            OSError: If its file could not be opened.
        """
        if self._last is None:
            return False
        forgotten = await self._store.forget_turn(self._last)
        self._recall.forget(self._last)
        self._last = None
        return forgotten

    async def reminder(
        self, text: str, zone: Callable[[], tzinfo | None] = local_zone
    ) -> str:
        """Return what other conversations hold close to ``text``, to go before it.

        Up to three turns scoring at least :data:`REMINDER_FLOOR`, under
        :data:`REMINDER_HEADER`; "" when none does, and always without an
        embedding model, since a word score alone says little about how
        close a turn is. A memory that cannot be read is logged and leaves
        the question as it is.
        """
        if self._recall.embedder is None:
            return ""
        try:
            found = await self._recall.find(
                text, limit=CANDIDATES, at_least=REMINDER_FLOOR
            )
        except Exception:
            logger.exception("earlier conversations were not searched")
            return ""
        others = [r.turn for r in found if r.turn.conversation != self._conversation]
        if not others:
            return ""
        local = zone()
        entries = "\n".join(shown(turn, local) for turn in others[:REMINDED])
        return f"{REMINDER_HEADER}\n{entries}\n\n"

    def begin_again(self) -> None:
        """Remember the turns from here on as a new conversation."""
        self._conversation = None
        self._last = None
