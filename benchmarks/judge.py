"""Judge whether a chat's answer says only what the passages it was shown say.

The judge reads the question, the answer, and the exact text of every
search result the chat's model was shown, and replies with a
:class:`Verdict`: supported or not, and the claims no passage states. An
answer given without any search has no passages, so a claim from the
model's own memory counts as unsupported even when it is true, and saying
the files do not hold the answer counts as supported.

The request goes to the local model at temperature 0 with reasoning off,
never to the remote one, so a verdict repeats and costs no budget. Two
worked examples go before the answer to judge as earlier turns, one
supported and one with two unsupported claims, and the instructions spell
out no empty list: asked to extract facts with an empty list spelled out,
the local model (qwen3.5-4b) returned it for 11 of 15 exchanges that held
facts, and with worked examples as turns instead, for none. The examples
are about a garden and a bicycle, nothing the person's files are likely to
hold. :func:`prompt_sha256` digests what the judge is asked with, so a
verdict can be tied to the prompt that gave it.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthia.gateway.structured import generate, schema_of
from synthia.gateway.types import ChatRequest, Message, Reasoning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from synthia.gateway.protocol import ChatModel

MAX_CLAIMS: Final = 8
MAX_CLAIM_CHARS: Final = 300
INSTRUCTIONS: Final = (
    "You check whether an answer says only what its passages say. The "
    "passages are what a search of the person's own files returned, and they "
    "are all the answer had to go on. Read the answer claim by claim. A claim "
    "is supported when a passage states it or it follows plainly from one. A "
    "claim no passage states is unsupported, even when it is true or common "
    "knowledge. Saying that the passages do not hold the answer is supported. "
    "Citations, greetings and offers to help further are not claims. When no "
    "passages were shown, every claim about the person's files is unsupported. "
    "If every claim is supported, set supported to true and list no claims. "
    "Otherwise set supported to false and list each unsupported claim in a few "
    "words, at most eight, the most important first."
)
PASSAGE_BREAK: Final = "\n\n---\n\n"
NO_PASSAGES: Final = "none: the answer was written without a search"


class Verdict(BaseModel):
    """The judge's reply: whether the answer is supported, and the claims that are not.

    Raises:
        ValueError: If ``supported`` is true while claims are listed, or false
            while none are.
    """

    model_config = ConfigDict(extra="forbid")

    supported: bool
    unsupported: list[
        Annotated[str, Field(min_length=1, max_length=MAX_CLAIM_CHARS)]
    ] = Field(max_length=MAX_CLAIMS)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.supported == bool(self.unsupported):
            message = (
                f"supported is {str(self.supported).lower()} with"
                f" {len(self.unsupported)} unsupported claims: true needs none"
                " listed, false needs at least one"
            )
            raise ValueError(message)
        return self


def judge_prompt(question: str, answer: str, passages: Sequence[str]) -> str:
    """Return the text the judge reads for ``answer`` to ``question``."""
    shown = PASSAGE_BREAK.join(passages) if passages else NO_PASSAGES
    return f"Question:\n{question}\n\nPassages:\n{shown}\n\nAnswer:\n{answer}"


EXAMPLES: Final = (
    Message.user(
        judge_prompt(
            "When are the tomatoes watered?",
            "Every second day, at six in the morning [garden/plan.md:3-5].",
            (
                (
                    "[garden/plan.md:3-5] Watering\nThe tomatoes are watered at "
                    "six in the morning, every second day. The hose reaches the "
                    "far bed."
                ),
            ),
        )
    ),
    Message.assistant('{"supported": true, "unsupported": []}'),
    Message.user(
        judge_prompt(
            "What does the bike need before the race?",
            "The rear brake pads need replacing, the tyres should be pumped to "
            "six bar, and the front wheel is out of true [bike/log.md:10-12].",
            (
                (
                    "[bike/log.md:10-12] Service\nThe chain was replaced on 4 "
                    "May. The rear brake pads are worn and need replacing."
                ),
            ),
        )
    ),
    Message.assistant(
        '{"supported": false, "unsupported": ["the tyres should be pumped to '
        'six bar", "the front wheel is out of true"]}'
    ),
)


def prompt_sha256(
    *, instructions: str = INSTRUCTIONS, examples: Sequence[Message] = EXAMPLES
) -> str:
    """Return the SHA-256, in hex, of what the judge is asked with.

    Taken over one JSON document holding the instructions, each example's
    role and text, and :class:`Verdict`'s JSON schema, so a change to any of
    them gives another digest.
    """
    document = {
        "instructions": instructions,
        "examples": [[m.role.value, m.text] for m in examples],
        "schema": schema_of(Verdict),
    }
    encoded = json.dumps(document, sort_keys=True, ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def judge_request(question: str, answer: str, passages: Sequence[str]) -> ChatRequest:
    """Return the request that asks the local model to judge ``answer``."""
    return ChatRequest(
        (
            Message.system(INSTRUCTIONS),
            *EXAMPLES,
            Message.user(judge_prompt(question, answer, passages)),
        ),
        temperature=0.0,
        reasoning=Reasoning.OFF,
        use_remote=False,
    )


async def judge(
    model: ChatModel, question: str, answer: str, passages: Sequence[str]
) -> Verdict:
    """Return ``model``'s verdict on ``answer``, given the ``passages`` it was shown.

    Raises:
        StructuredOutputError: If the model's replies never match
            :class:`Verdict`, each shown back with its errors to repair.
        GatewayError: Whatever else the model raised.
    """
    return await generate(model, judge_request(question, answer, passages), Verdict)
