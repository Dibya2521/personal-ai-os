import asyncio
from contextvars import copy_context

from synthia.agent.tools import Effect, FunctionTool, Reach
from synthia.gateway.confinement import CONFINEMENT, Confinement, confined

READ_NOTE = "read a private note"


def private_reader() -> FunctionTool:
    """A tool that reads private text, as the knowledge tool will."""

    def read_note() -> str:
        """Read the person's private note."""
        latch = CONFINEMENT.get()
        if latch is not None:
            latch.confine(READ_NOTE)
        return "the note says: the door code is 4021"

    return FunctionTool.of(read_note, reach=Reach.LOCAL, effect=Effect.READ)


def test_a_latch_starts_open_keeps_its_first_reason_and_opens_on_release() -> None:
    latch = Confinement()
    was_open = latch.confined

    latch.confine("first")
    latch.confine("second")
    closed, reason = latch.confined, latch.reason
    latch.release()

    assert (was_open, closed, reason) == (False, True, "first")
    assert (latch.confined, latch.reason) == (False, None)


def test_only_a_closed_latch_in_context_confines() -> None:
    def in_context(latch: Confinement | None) -> bool:
        CONFINEMENT.set(latch)
        return confined()

    closed = Confinement()
    closed.confine(READ_NOTE)

    assert copy_context().run(in_context, None) is False
    assert copy_context().run(in_context, Confinement()) is False
    assert copy_context().run(in_context, closed) is True


async def test_a_latch_closed_in_a_task_of_the_turn_closes_it_for_the_turn() -> None:
    latch = Confinement()
    CONFINEMENT.set(latch)

    await asyncio.create_task(private_reader().run("{}"))

    assert confined()
