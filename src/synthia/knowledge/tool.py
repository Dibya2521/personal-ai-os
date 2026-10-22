"""``search_knowledge``: the tool the chat model calls to read the person's files.

It reads only this machine's index and changes nothing, so it runs without
asking. Up to six passages come back, best first, each under its citation
and followed by the titles of the documents it links to, so the model can
search for one of them next. The citations it shows are added to the running
turn's list (:data:`CITED`), which the chat lists under the answer.

A passage from a private source is shown only to a chat that can stay on this
machine: the chat's latch is in context and a local model that takes tools is
there to answer. The latch is then closed before the result returns, so the
very next request of the turn is already local. Anywhere else private
passages are withheld and the result says how many. With no latch in context
nothing private is shown.

A search runs deep (:mod:`synthia.knowledge.deep`) only while :data:`DEEP` is
set, which only the session does, for the search a person asks for with
``/deep``: the model sees the same one parameter either way, so it can never
start a generation of tens of seconds on its own.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import Field

from synthia.agent.tools import Effect, FunctionTool, Reach
from synthia.gateway.confinement import CONFINEMENT
from synthia.knowledge.deep import NO_REWRITES, deep_search

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from synthia.gateway.protocol import ChatModel
    from synthia.knowledge.model import ChunkId
    from synthia.knowledge.retrieve import Found, Passage
    from synthia.knowledge.service import Knowledge
    from synthia.knowledge.store import LinkTarget

# Six passages stay near 15,000 characters: room for the answer in a
# 32,768-token window.
MAX_BODY_CHARS: Final = 2400
MIN_KEPT_CHARS: Final = MAX_BODY_CHARS // 2
"""A body is cut at its last line end only if that keeps at least this much."""
CUT: Final = "..."
NOTHING_FOUND: Final = "nothing found in the person's files"
CANNOT_STAY_LOCAL: Final = "this chat cannot stay on this machine"
LINKS_TO: Final = "links to: "
DESCRIPTION: Final = (
    "Search the person's own files (notes, documents and code) for passages "
    "that answer a question about their projects, notes or code. Each passage "
    "comes under its citation in square brackets: name the citations you used "
    "under your answer. A passage may end with the titles of documents it "
    "links to; search for one of those titles to follow the link."
)
SEARCH_KNOWLEDGE: Final = "search_knowledge"
CITED: Final[ContextVar[list[str] | None]] = ContextVar("cited", default=None)
"""The running turn's citations, in the order shown; None outside a chat's turn."""
DEEP: Final[ContextVar[bool]] = ContextVar("deep", default=False)
"""Whether a search now runs deep; set by the session around one search."""


def _private_read(sources: Sequence[str]) -> str:
    return f"private files from {', '.join(sources)} were read"


def _cut(body: str) -> str:
    if len(body) <= MAX_BODY_CHARS:
        return body
    end = body.rfind("\n", 0, MAX_BODY_CHARS + 1)
    if end < MIN_KEPT_CHARS:
        return body[:MAX_BODY_CHARS] + CUT
    return f"{body[:end]}\n{CUT}"


def _shown(passage: Passage, targets: Sequence[LinkTarget]) -> str:
    heading = passage.chunk.header or passage.document.title
    lines = [f"[{passage.citation()}] {heading}", _cut(passage.chunk.body)]
    titles = dict.fromkeys(target.title for target in targets)
    if titles:
        lines.append(LINKS_TO + ", ".join(titles))
    return "\n".join(lines)


def _withheld(count: int) -> str:
    passages = "passage was" if count == 1 else "passages were"
    return f"{count} private {passages} withheld: {CANNOT_STAY_LOCAL}"


def _result(found: Found, links: Mapping[ChunkId, Sequence[LinkTarget]]) -> str:
    parts = [_shown(p, links.get(p.chunk.id, ())) for p in found.passages]
    if found.withheld:
        parts.append(_withheld(found.withheld))
    return "\n\n".join(parts) or NOTHING_FOUND


def knowledge_tool(
    knowledge: Knowledge, *, can_stay_local: Callable[[], bool], model: ChatModel
) -> FunctionTool:
    """Return the tool that searches ``knowledge``.

    ``can_stay_local`` says whether a local model that takes tools is there
    now, which decides whether private passages are shown. ``model``, the
    gateway's, writes a deep search's rewrites.
    """

    async def search_knowledge(
        query: Annotated[
            str, Field(min_length=1, description="what to look for, in any words")
        ],
    ) -> str:
        latch = CONFINEMENT.get()
        show_private = latch is not None and can_stay_local()
        note = ""
        if DEEP.get():
            deep = await deep_search(knowledge, model, query, private=show_private)
            found = deep.found
            if deep.expansion is None:
                note = f"\n\n{NO_REWRITES}"
        else:
            found = await knowledge.search(query, private=show_private)
        private = dict.fromkeys(p.source for p in found.passages if p.private)
        if latch is not None and private:
            latch.confine(_private_read(list(private)))
        cited = CITED.get()
        if cited is not None:
            cited.extend(p.citation() for p in found.passages)
        return _result(found, await knowledge.links(found.passages)) + note

    # The model reads only a docstring's first paragraph, and this one is
    # longer than a summary line may be.
    search_knowledge.__doc__ = DESCRIPTION
    return FunctionTool.of(
        search_knowledge, reach=Reach.LOCAL, effect=Effect.READ, name=SEARCH_KNOWLEDGE
    )
