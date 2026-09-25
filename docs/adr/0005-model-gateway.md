# 0005. One gateway for every model, and refuse before sending

Status: accepted.

## Context

SYNTHIA's reasoning comes from two kinds of model. The remote one is
`openrouter/free`, which sends each request to a free model chosen at random
from a catalogue that changes over time; an account that has bought less than
10 dollars of credit may send it 50 requests per UTC day and 20 per minute. The
local one is a small model on the machine, weaker but always there. Every
feature above them (chat, then agents, memory and voice) needs an answer
without caring which model gave it.

Two facts about the remote side shape everything else. A request counts
against the daily limit once it reaches the provider, whether it succeeds or
fails. And a streamed answer that has started cannot be taken back: its words
are on screen, or already spoken.

## Decision

- **One protocol.** Every model is a `ChatModel` whose `stream(request)`
  yields chunks; a complete answer is the stream collected. Streaming is the
  primitive because speech must start with the first sentence.
- **One adapter for the OpenAI-compatible chat API** serves both OpenRouter
  and the llama.cpp server. A provider is a base URL, headers and a capability
  record (context window, images, tools).
- **Guards compose as wrappers around the remote adapter**, outermost first:
  retry with full-jitter backoff, a circuit breaker, then a rate limiter and a
  budget claim. The limiter waits before the claim, because a wait can be
  cancelled for free and a claim is never given back. Both are innermost so
  every attempt is counted, as the provider counts attempts.
- **The budget is claimed before the request leaves the machine**, in one
  SQLite `BEGIN IMMEDIATE` transaction, and a request over the cap is refused
  locally. The count survives restarts and is shared by every process.
- **Rate limiting uses a sliding-window log**: at most 20 requests in any 60
  seconds.
- **A router chooses per request**: remote first; local for background jobs,
  for requests the remote cannot take, while the remote circuit is open, when
  10 or fewer requests are left for the day, or when the rate limit would hold
  a request over 5 seconds. A remote failure before the first chunk falls back
  to local once; after it, nothing switches.
- **Tests replay recorded exchanges** and never reach the network.

## Options considered

- **A framework such as LangChain or LiteLLM.** Routing, retries and budgets
  come ready made, but the reasoning that matters here (claim before sending,
  limiter before claim, no switch after the first chunk) would live inside
  someone else's abstractions, and the part worth engineering would be glue.
- **Check the budget after sending**, from the provider's response headers.
  Simpler, but a failed request still counts at the provider, so the 51st
  request of a day would be sent and fail there instead of being refused here.
- **A token bucket for the rate limit.** The usual choice, but a bucket of 20
  per minute admits 20 at once and then one every 3 seconds: 30 requests in the
  first 60 seconds, which a provider counting over a sliding window rejects.
  The log costs one timestamp per recent request, 20 numbers.
- **Local first.** Keeps every conversation on the machine, but a 4B local
  model is far weaker than the free remote ones, and the remote budget would go
  unused on most days.
- **Fall back mid-answer.** Would rescue more failures, but the second model
  would repeat or contradict words the person has already read or heard.

## Consequences

A new provider is one adapter, and a model leaving the free catalogue changes a
setting, not code. The guards and the router are tested in isolation with fake
models and clocks, and the whole path with recorded real responses, so CI
spends nothing. The reserve of 10 and the 5-second wait are reasoned, not
measured: 10 is a fifth of the day's 50, kept for requests only the remote can
serve, and 5 seconds assumes a local 4B model on a CPU is
slower than a short wait for the remote. Both are revisited once the local
model's speed is measured.
