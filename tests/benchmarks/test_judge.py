import hashlib
import json

import pytest
from judge import (
    EXAMPLES,
    INSTRUCTIONS,
    MAX_CLAIM_CHARS,
    MAX_CLAIMS,
    NO_PASSAGES,
    PASSAGE_BREAK,
    Verdict,
    judge,
    judge_prompt,
    judge_request,
    prompt_sha256,
)

from synthia.gateway.structured import StructuredOutputError, schema_of
from synthia.gateway.types import ChatRequest, Message, Reasoning, Role
from tests.knowledge.test_deep import Local

QUESTION = "Which host runs the database?"
ANSWER = "kestrel [notes/kestrel.md:1-6]"
PASSAGES = (
    "[notes/kestrel.md:1-6] Hosts\nthe database server host kestrel",
    "[notes/pune.md:1-3] Pune trip\nmy sister lives in pune city",
)


def verdict(*claims: str) -> str:
    return json.dumps({"supported": not claims, "unsupported": list(claims)})


async def test_judge_sends_one_local_request_with_the_examples_as_turns() -> None:
    model = Local(verdict())

    found = await judge(model, QUESTION, ANSWER, PASSAGES)

    assert found == Verdict(supported=True, unsupported=[])
    (request,) = model.requests
    assert request == ChatRequest(
        judge_request(QUESTION, ANSWER, PASSAGES).messages,
        temperature=0.0,
        response_schema=schema_of(Verdict),
        reasoning=Reasoning.OFF,
        use_remote=False,
    )
    assert [m.role for m in request.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]
    assert request.messages[0].text == INSTRUCTIONS
    assert request.messages[1:5] == EXAMPLES
    assert request.messages[-1].text == (
        f"Question:\n{QUESTION}\n\nPassages:\n{PASSAGES[0]}{PASSAGE_BREAK}"
        f"{PASSAGES[1]}\n\nAnswer:\n{ANSWER}"
    )


def test_an_answer_with_no_search_is_judged_against_no_passages() -> None:
    assert judge_prompt(QUESTION, ANSWER, ()) == (
        f"Question:\n{QUESTION}\n\nPassages:\n{NO_PASSAGES}\n\nAnswer:\n{ANSWER}"
    )


def test_the_examples_are_one_supported_and_one_with_two_claims() -> None:
    supported = Verdict.model_validate_json(EXAMPLES[1].text)
    unsupported = Verdict.model_validate_json(EXAMPLES[3].text)

    assert (supported.supported, supported.unsupported) == (True, [])
    assert unsupported.supported is False
    assert len(unsupported.unsupported) == 2


def test_the_instructions_spell_out_no_json() -> None:
    assert not {"[", "]", "{", "}"} & set(INSTRUCTIONS)


@pytest.mark.parametrize(
    "bad",
    [
        json.dumps({"supported": True, "unsupported": ["a claim"]}),
        json.dumps({"supported": False, "unsupported": []}),
        verdict(*[f"claim {n}" for n in range(MAX_CLAIMS + 1)]),
        verdict("c" * (MAX_CLAIM_CHARS + 1)),
        verdict(""),
        '{"supported": true, "unsupported": [], "why": "x"}',
        "yes, it is supported",
    ],
    ids=[
        "true_with_claims",
        "false_without",
        "nine",
        "long",
        "empty_claim",
        "extra",
        "prose",
    ],
)
async def test_a_bad_verdict_is_repaired(bad: str) -> None:
    model = Local(bad, verdict("the host is in Pune"))

    found = await judge(model, QUESTION, ANSWER, PASSAGES)

    assert found.unsupported == ["the host is in Pune"]
    assert len(model.requests) == 2
    assert model.requests[1].messages[-2].text == bad


def test_prompt_sha256_is_of_the_instructions_examples_and_schema() -> None:
    document = {
        "instructions": INSTRUCTIONS,
        "examples": [[m.role.value, m.text] for m in EXAMPLES],
        "schema": schema_of(Verdict),
    }
    encoded = json.dumps(document, sort_keys=True, ensure_ascii=True).encode()

    assert prompt_sha256() == hashlib.sha256(encoded).hexdigest()
    assert prompt_sha256(instructions=f"{INSTRUCTIONS} ") != prompt_sha256()
    assert prompt_sha256(examples=EXAMPLES[:2]) != prompt_sha256()
    flipped = (Message.assistant(EXAMPLES[0].text), *EXAMPLES[1:])
    assert prompt_sha256(examples=flipped) != prompt_sha256()


async def test_a_verdict_never_repaired_fails() -> None:
    model = Local("no", "still no", "never")

    with pytest.raises(StructuredOutputError, match="no valid Verdict"):
        await judge(model, QUESTION, ANSWER, PASSAGES)
