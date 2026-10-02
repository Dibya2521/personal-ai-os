# 0008. Thinking depth fits the question, as a time allowance

Status: accepted.

## Context

Thinking models reason before they answer: Qwen3.5 locally, and many of the
free models behind OpenRouter. The first real local run asked Qwen3.5-4B a
question through llama.cpp's server, which lets the model think by default,
and the model spent all 512 of its output tokens thinking and gave no answer.
Turning thinking off everywhere would make hard questions worse; leaving it
on unbounded makes a greeting wait. The depth has to fit the question, and the
limit has to hold on two providers that can be controlled in different ways.

## Decision

- **Each request carries a level**: `off`, `low`, `medium`, `high` or `auto`.
  A request with no level sends no setting at all, so the provider's default
  applies and the request body is exactly what it was before levels existed.
- **`auto` is decided by fixed rules on the latest user message**, with no
  extra request (`synthia/gateway/reasoning.py`). Each sign adds points once:
  code (a fenced block, a line shaped like code, a traceback) 2; mathematics
  words 2, or else arithmetic between numbers 1; reasoning words such as why,
  compare or trade-off 2; proof or depth words such as prove or step by step 3;
  two or more questions 1; more than 80 words 1. 0 points is `off` for 8 words
  or fewer and `low` above that; 1 is `low`, 2 `medium`, 3 or more `high`.
  Length alone never goes past `low`, so a long pasted log with a trivial
  question does not buy a long think. The word lists are English only.
- **The router decides `auto`**, after it has chosen the route, so the level is
  decided once per request and published with the route decision.
- **A level is a time allowance**, the same wait on any model: low 5 s, medium
  20 s, high 60 s. These are starting values, not measurements.
- **Remote: a token limit, sent up front.** OpenRouter cannot be told to stop
  mid-thought, so the allowance becomes `reasoning.max_tokens`: the allowance
  times the speed of the remote models, measured as completion tokens per
  second of whole calls over the last 7 UTC days, with the local model's calls
  left out. Before anything is measured the speed is taken as 25 tokens/s: the
  slowest of 4 free models measured end to end on 2026-09-23 ran at 26.5
  tokens/s (the fastest at 73.6), so the slow end keeps the wait inside the
  allowance. `off` is sent as `reasoning.effort: "none"`.
- **Local: thinking is ended on time.** llama.cpp's server passes
  `chat_template_kwargs.enable_thinking` to the model's chat template, which is
  Qwen3.5's only thinking switch. With `"reasoning_control": true` in the
  request, `POST /v1/chat/completions/control` with `{"id": <completion id>,
  "action": "reasoning_end"}` closes the thinking and the answer follows. The
  local model starts a timer at the first thought; if no answer text, tool call
  or finish has come when the allowance runs out, it posts that once. A refused
  or failed post is logged and never raised, because the answer still comes,
  only when the model stops thinking by itself.
- **The chat sends `auto` by default.** `/think off|low|medium|high|auto`
  changes the level from the next turn on. A dim "thinking N s" line counts the
  seconds while the model thinks and disappears when the answer starts, and the
  line after each answer names the level that was sent.

## Options considered

- **Thinking always off on the local model.** Fast, but a 4-billion-parameter
  model loses the reasoning that makes it useful on mathematics and code.
- **llama.cpp's `--reasoning-budget N`.** One token budget for every request,
  fixed when the server starts; it cannot fit the question.
- **A token budget on both providers.** A token count is a different wait on
  each model: 26.5 to 73.6 tokens/s across the free remote models, and a local
  model's speed depends on the machine it runs on. A time allowance means the
  same wait everywhere, and each provider turns it into what it can enforce.
- **Ask a model to judge each question first.** It costs a request out of a
  budget of 50 a day and delays every answer by a round trip. The rules are
  free, instant, deterministic and tested.
- **OpenRouter's `reasoning.effort` for every level.** Each model maps effort
  its own way; a token limit means the same on all of them. Effort is used only
  for `off`, where `none` is unambiguous.

## Consequences

The same question always gets the same depth, and a greeting never waits for
a think. The two providers are treated differently on purpose: only llama.cpp
can end thinking on command, so it is stopped live, while the remote model
gets its limit before it starts. A remote model that ignores `reasoning`
thinks as it would have anyway; nothing fails.

Checked against llama-server b11130 with Qwen3.5-4B: the completion id is each
chunk's `id` field; the post was answered `{"success": true}` and the answer
began 1.2 seconds later, after two more thinking chunks that were already on
their way. An unknown id is answered 200 with `{"success": false}`, not with an
error status, so the reply body is read. The allowances and the rule weights
are starting values: a benchmark of real questions should set them.
