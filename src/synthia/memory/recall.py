"""``search_memory``: the tool the model calls to look back at earlier conversations.

It reads only this machine's memory and changes nothing, so it runs without
asking. Turns are found by their words and, when an embedding model is
installed, by their meaning (:mod:`synthia.memory.hybrid`), so the person's
own words need not match. Dates are days on this machine's clock, the same
days the clock tool names, so "last week" is turned into dates by the
model, which reads the clock first; nothing here parses words like
"yesterday".

A turn remembered from a chat that held private text is shown only to a chat
that can stay on this machine, a local model that takes tools being there to
answer it; that chat is confined before it reads the turn. Anywhere else the
turn is left out, as if it were not found.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import Field

from synthia.agent.tools import Effect, FunctionTool, Reach
from synthia.gateway.confinement import CONFINEMENT

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import tzinfo

    from synthia.memory.hybrid import Recall
    from synthia.memory.store import Remembered

MAX_FOUND: Final = 8
# Eight turns of two previews each stay near 6,000 characters: room for the
# answer in a 32,768-token window.
MAX_PREVIEW_CHARS: Final = 300
FIRST_DAY: Final = date(1900, 1, 1)
LAST_DAY: Final = date(9998, 12, 31)
type Day = Annotated[date, Field(ge=FIRST_DAY, le=LAST_DAY)]
"""A day a search may name; nearer year 1 or 9999 a time zone's offset overflows."""
NOTHING_FOUND: Final = "nothing found in earlier conversations"
RECALLED_PRIVATE: Final = "an earlier private conversation was recalled"


def local_zone() -> tzinfo | None:
    """Return this machine's time zone, the one its clock tool names days in."""
    return datetime.now().astimezone().tzinfo


def cannot_keep_private() -> bool:
    """Return False: without a router to ask, no chat is known to stay local."""
    return False


def surfaced(
    turns: Sequence[Remembered], most: int, can_keep_private: Callable[[], bool]
) -> list[Remembered]:
    """Return up to ``most`` of ``turns`` that the running chat may read.

    A confined turn is kept only in a chat's turn whose latch is in context
    and when ``can_keep_private``; the latch is then closed if one is returned.
    Otherwise confined turns are left out.
    """
    latch = CONFINEMENT.get()
    if latch is None or not can_keep_private():
        return [turn for turn in turns if not turn.confined][:most]
    shown = list(turns[:most])
    if any(turn.confined for turn in shown):
        latch.confine(RECALLED_PRIVATE)
    return shown


def preview(text: str) -> str:
    """Return ``text`` on one line, cut to ``MAX_PREVIEW_CHARS``."""
    flat = " ".join(text.split())
    if len(flat) <= MAX_PREVIEW_CHARS:
        return flat
    return flat[:MAX_PREVIEW_CHARS] + "..."


def day_range(
    since: date | None, until: date | None, zone: tzinfo | None
) -> tuple[datetime | None, datetime | None]:
    """Return the start of day ``since`` and the end of day ``until`` in ``zone``."""
    start = None if since is None else datetime.combine(since, time(), zone)
    end = (
        None
        if until is None
        else datetime.combine(until + timedelta(days=1), time(), zone)
    )
    return start, end


def shown(turn: Remembered, zone: tzinfo | None) -> str:
    """Return ``turn`` as one entry the model reads: when, what was said, what ran."""
    at = turn.at.astimezone(zone)
    lines = [
        f"{at:%A %Y-%m-%d %H:%M} (conversation {turn.conversation}, turn {turn.id})",
        f"  person: {preview(turn.question)}",
        f"  SYNTHIA: {preview(turn.answer)}",
    ]
    if turn.tools:
        lines.append(f"  tools used: {', '.join(use.name for use in turn.tools)}")
    return "\n".join(lines)


def memory_tool(
    recall: Recall,
    zone: Callable[[], tzinfo | None] = local_zone,
    *,
    can_keep_private: Callable[[], bool] = cannot_keep_private,
) -> FunctionTool:
    """Return the tool that searches what ``recall`` remembers.

    ``can_keep_private`` says whether a chat could go on once it holds private
    text, which decides whether turns of private chats are shown.
    """

    async def search_memory(
        query: Annotated[
            str,
            Field(
                description="what to look for, in any words; empty for the newest turns"
            ),
        ],
        since: Annotated[
            Day | None,
            Field(description="the first day to include, YYYY-MM-DD"),
        ] = None,
        until: Annotated[
            Day | None,
            Field(description="the last day to include, YYYY-MM-DD"),
        ] = None,
    ) -> str:
        """Search what the person and SYNTHIA said in earlier conversations.

        Use it when asked about something said or done before, such as "what
        did I tell you about X last week". For dates, read the current time
        first. Returns up to eight turns, best match first, each with its day.
        """
        local = zone()
        start, end = day_range(since, until, local)
        found = await recall.find(query, since=start, until=end, limit=MAX_FOUND)
        turns = surfaced([r.turn for r in found], MAX_FOUND, can_keep_private)
        if not turns:
            return NOTHING_FOUND
        return "\n".join(shown(turn, local) for turn in turns)

    return FunctionTool.of(search_memory, reach=Reach.LOCAL, effect=Effect.READ)
