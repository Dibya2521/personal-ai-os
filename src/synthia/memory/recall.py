"""``search_memory``: the tool the model calls to look back at earlier conversations.

It reads only this machine's memory and changes nothing, so it runs without
asking. Turns are found by their words and, when an embedding model is
installed, by their meaning (:mod:`synthia.memory.hybrid`), so the person's
own words need not match. Dates are days on this machine's clock, the same
days the clock tool names, so "last week" is turned into dates by the
model, which reads the clock first; nothing here parses words like
"yesterday".
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import Field

from synthia.agent.tools import Effect, FunctionTool, Reach

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import tzinfo

    from synthia.memory.hybrid import Recall
    from synthia.memory.store import Remembered

MAX_FOUND: Final = 8
# Eight turns of two previews each stay near 6,000 characters: room for the
# answer in a 32,768-token window.
MAX_PREVIEW_CHARS: Final = 300
NOTHING_FOUND: Final = "nothing found in earlier conversations"


def local_zone() -> tzinfo | None:
    """Return this machine's time zone, the one its clock tool names days in."""
    return datetime.now().astimezone().tzinfo


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= MAX_PREVIEW_CHARS:
        return flat
    return flat[:MAX_PREVIEW_CHARS] + "..."


def shown(turn: Remembered, zone: tzinfo | None) -> str:
    """Return ``turn`` as one entry the model reads: when, what was said, what ran."""
    at = turn.at.astimezone(zone)
    lines = [
        f"{at:%A %Y-%m-%d %H:%M} (conversation {turn.conversation}, turn {turn.id})",
        f"  person: {_preview(turn.question)}",
        f"  SYNTHIA: {_preview(turn.answer)}",
    ]
    if turn.tools:
        lines.append(f"  tools used: {', '.join(use.name for use in turn.tools)}")
    return "\n".join(lines)


def memory_tool(
    recall: Recall, zone: Callable[[], tzinfo | None] = local_zone
) -> FunctionTool:
    """Return the tool that searches what ``recall`` remembers."""

    async def search_memory(
        query: Annotated[
            str,
            Field(
                description="what to look for, in any words; empty for the newest turns"
            ),
        ],
        since: Annotated[
            date | None,
            Field(description="the first day to include, YYYY-MM-DD"),
        ] = None,
        until: Annotated[
            date | None,
            Field(description="the last day to include, YYYY-MM-DD"),
        ] = None,
    ) -> str:
        """Search what the person and SYNTHIA said in earlier conversations.

        Use it when asked about something said or done before, such as "what
        did I tell you about X last week". For dates, read the current time
        first. Returns up to eight turns, best match first, each with its day.
        """
        local = zone()
        start = None if since is None else datetime.combine(since, time(), local)
        end = (
            None
            if until is None
            else datetime.combine(until + timedelta(days=1), time(), local)
        )
        found = await recall.find(query, since=start, until=end, limit=MAX_FOUND)
        if not found:
            return NOTHING_FOUND
        return "\n".join(shown(r.turn, local) for r in found)

    return FunctionTool.of(search_memory, reach=Reach.LOCAL, effect=Effect.READ)
