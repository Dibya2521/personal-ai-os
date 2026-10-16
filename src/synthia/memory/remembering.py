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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from synthia.memory.hybrid import CANDIDATES
from synthia.memory.recall import cannot_stay_local, local_zone, shown, surfaced
from synthia.memory.store import Asked

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from datetime import datetime, tzinfo

    from synthia.memory.facts import FactLearning
    from synthia.memory.hybrid import Recall
    from synthia.memory.store import Fact, Forgotten, Remembered, Turn

logger = logging.getLogger(__name__)

REMINDED: Final = 3
# Precision first: on the recall set the right turn reaches it for 25 % of
# questions, an unrelated one for 6.7 %; misses are left to search_memory.
REMINDER_FLOOR: Final = 0.55
REMINDER_HEADER: Final = (
    "From SYNTHIA's memory of earlier conversations, not written by the "
    "person; use it only if it helps:"
)
PINNED_HEADER: Final = (
    "What SYNTHIA has learned about the person in earlier conversations, "
    "newest first; the person may have changed since:"
)


@dataclass(frozen=True, slots=True)
class Resumed:
    """An earlier conversation taken up again: its turns, its last persona's name."""

    conversation: int
    persona: str
    turns: tuple[Remembered, ...]


@dataclass(frozen=True, slots=True)
class _Pinned:
    """The facts block a conversation took, and the (id, text) of each fact in it."""

    block: str
    facts: frozenset[tuple[int, str]]


def _pinned(facts: Iterable[Fact], max_chars: int) -> _Pinned:
    lines: list[str] = []
    taken: set[tuple[int, str]] = set()
    used = len(PINNED_HEADER)
    for fact in facts:
        if fact.until is not None:
            continue
        line = f"- {fact.text}"
        used += len(line) + 1
        if used > max_chars:
            break
        lines.append(line)
        taken.add((fact.id, fact.text))
    block = "\n".join((PINNED_HEADER, *lines)) if lines else ""
    return _Pinned(block, frozenset(taken))


class Remembering:
    """Keep one conversation's turns in ``recall``'s store unless it is private."""

    def __init__(
        self,
        recall: Recall,
        learning: FactLearning | None = None,
        *,
        can_stay_local: Callable[[], bool] = cannot_stay_local,
    ) -> None:
        """Remember into ``recall``; with ``learning``, learn facts from each turn.

        ``can_stay_local`` says whether a chat could go on with no remote
        model; only then are turns of private chats put in its reminders.
        """
        self.private = False
        self._recall = recall
        self._store = recall.store
        self._learning = learning
        self._can_stay_local = can_stay_local
        self._conversation: int | None = None
        # The newest turn and when it was asked: its id alone can be given to
        # another chat's turn once forgotten there.
        self._last: tuple[int, datetime] | None = None
        self._pinned: _Pinned | None = None

    async def remember(self, turn: Turn) -> None:
        """Keep ``turn`` and its meaning, unless the conversation is private.

        A turn not kept leaves nothing for :meth:`forget_last` to forget,
        since the conversation's last turn is now that one.
        """
        self._last = None
        if self.private:
            return
        try:
            if self._conversation is None:
                self._conversation = await self._store.begin_conversation(
                    turn.persona, turn.at
                )
            kept = await self._store.add_turn(
                self._conversation, turn, to_learn=self._learning is not None
            )
        except (sqlite3.Error, OSError):
            logger.exception("a turn was not remembered")
            return
        self._last = (kept, turn.at)
        if self._learning is not None:
            self._learning.wake()
        try:
            await self._recall.keep(Asked(kept, turn.at, turn.question))
        except Exception:
            logger.exception("a turn's meaning was not kept")

    async def forget_last(self) -> bool:
        """Forget the last turn, if it was remembered; False if there is none.

        Raises:
            sqlite3.Error: If the store could not be changed.
            OSError: If its file could not be opened.
        """
        if self._last is None:
            return False
        turn, at = self._last
        forgotten = await self._store.forget_turn(
            turn, conversation=self._conversation, at=at
        )
        if forgotten:
            self._recall.forget(turn)
        self._last = None
        return forgotten

    async def facts(self) -> list[Fact]:
        """Return the facts held true about the person, newest first.

        Raises:
            sqlite3.Error: If the store could not be read.
            OSError: If its file could not be opened.
        """
        return await self._store.facts()

    async def forget_fact(self, fact: int) -> Fact | None:
        """Forget the fact numbered ``fact`` and return it; None if there is none.

        Raises:
            sqlite3.Error: If the store could not be changed.
            OSError: If its file could not be opened.
        """
        known = {f.id: f for f in await self._store.facts(held=False)}
        if fact not in known or not await self._store.forget_fact(fact):
            return None
        return known[fact]

    async def forget_about(self, words: str) -> Forgotten:
        """Forget every fact and every turn holding all of ``words``.

        Raises:
            sqlite3.Error: If the store could not be changed.
            OSError: If its file could not be opened.
        """
        forgotten = await self._store.forget_about(words)
        for turn in forgotten.turns:
            self._recall.forget(turn)
        return forgotten

    async def pinned(self, max_chars: int) -> str:
        """Return the facts held, newest first, as lines for the system message.

        Whole facts only, up to ``max_chars`` with the header; "" when there
        are none. Taken once per conversation, since a system message that
        changes makes a server that reuses the start of the last prompt read
        it all again; taken afresh after :meth:`begin_again`, or once a fact
        in it is forgotten, here or in another conversation. A store that
        cannot be read is logged and pins nothing for that turn, since what
        was taken may hold a fact forgotten since.
        """
        try:
            facts = await self._store.facts(held=False)
        except (sqlite3.Error, OSError):
            logger.exception("facts about the person were not read")
            return ""
        known = {(fact.id, fact.text) for fact in facts}
        if self._pinned is None or not self._pinned.facts <= known:
            self._pinned = _pinned(facts, max_chars)
        return self._pinned.block

    async def reminder(
        self, text: str, zone: Callable[[], tzinfo | None] = local_zone
    ) -> str:
        """Return what other conversations hold close to ``text``, to go before it.

        Up to three turns scoring at least :data:`REMINDER_FLOOR`, under
        :data:`REMINDER_HEADER`; "" when none does, and always without an
        embedding model, since a word score alone says little about how
        close a turn is. A memory that cannot be read is logged and leaves
        the question as it is. A turn of a private chat is put here only as
        :func:`~synthia.memory.recall.surfaced` allows, confining this chat.
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
        others = surfaced(
            [r.turn for r in found if r.turn.conversation != self._conversation],
            REMINDED,
            self._can_stay_local,
        )
        if not others:
            return ""
        local = zone()
        entries = "\n".join(shown(turn, local) for turn in others)
        return f"{REMINDER_HEADER}\n{entries}\n\n"

    async def resume(self) -> Resumed | None:
        """Take up the conversation of the newest remembered turn; None if none is.

        Turns from here on are remembered in it, and the last forgettable
        turn is its newest.

        Raises:
            sqlite3.Error: If the store could not be read.
            OSError: If its file could not be opened.
        """
        last = await self._store.last_conversation()
        if last is None:
            return None
        conversation, persona = last
        turns = await self._store.conversation(conversation)
        if not turns:
            return None
        self._conversation = conversation
        self._last = (turns[-1].id, turns[-1].at)
        self._pinned = None
        return Resumed(conversation, persona, tuple(turns))

    def begin_again(self) -> None:
        """Remember the turns from here on as a new conversation."""
        self._conversation = None
        self._last = None
        self._pinned = None
