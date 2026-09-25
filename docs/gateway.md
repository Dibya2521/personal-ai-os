# The model gateway

Every language model SYNTHIA uses, remote or local, sits behind one interface.
The rest of the system asks the gateway for an answer and never knows which
model gave it, how many attempts it took, or what was counted on the way. This
document describes how a request travels, what each layer guarantees, and the
numbers each one runs on.

## One interface

A model is anything with `stream(request)`, which yields the answer as chunks:
text as it is generated, fragments of tool calls, and at the end a finish reason
and token counts. A complete answer is the stream collected. Streaming is the
primitive, not an option, because a spoken answer has to start with its first
sentence, long before the model has finished.

Requests and chunks are provider-neutral types (`synthia/gateway/types.py`):
messages with text and image parts, tool definitions, temperature, a token
limit and an optional JSON Schema for the reply. An adapter translates them to
one provider's wire format; nothing above the adapter ever sees that format.

Both providers SYNTHIA uses today, OpenRouter and the llama.cpp server, speak
the OpenAI-compatible chat API, so a single adapter serves both
(`synthia/gateway/openai_compat.py`). A provider is a base URL, headers and a
record of what its model can do: context window, images, tools. A third
provider speaking another protocol would be one more adapter behind the same
interface.

## The path of a request

```text
caller
  -> accounting                 records the call once it finishes
  -> router                     picks remote or local, per request
       remote: retry -> circuit breaker -> rate limit + budget -> adapter -> OpenRouter
       local:  adapter -> llama.cpp server on 127.0.0.1
```

The chat is built this way by `build_gateway` (`synthia/gateway/assemble.py`),
the one place that knows how the pieces fit. Everything else receives the
finished model.

## Routing: remote first, local when it is the better answer

The free remote models are far stronger than a 4-billion-parameter local one,
so a request goes remote unless the first of these rules says otherwise:

1. **It is a background job.** The daily remote budget is kept for talking.
2. **It needs images or tools the remote model cannot take.**
3. **The remote circuit is open.** The provider has just failed several times,
   so the answer starts at once from the local model instead of after a timeout.
4. **The remote budget is down to the reserve** (`SYNTHIA_REMOTE_RESERVE`, 10
   by default). The last 10 of the day's 50 requests are kept for what only
   the remote model can do.
5. **The rate limit would hold the request for more than 5 seconds.** A
   short wait for a strong model is worth it; a long one is not.

Each rule applies only if a local model is installed, has finished loading and
can serve the request. While the local server is still loading, it counts as
absent. With no local model, every request goes remote while any budget is
left at all.

If the remote model fails before sending its first chunk, the same request is
sent to the local model once. After the first chunk nothing switches: those
words are already on screen or spoken, and a second model would repeat or
contradict them. Every decision is logged and published as a `RouteDecided`
event with its reason, so a trace shows where each answer came from and why
(`synthia/gateway/router.py`).

## The remote guards

The remote adapter is wrapped in three layers. Their order matters, and each
one's reason is below.

| Layer, outermost first | Rule | Defaults |
| --- | --- | --- |
| retry | send again after a transient failure, never once output has started | 3 attempts; wait drawn from 0 to min(8 s, 0.5 s x 2^(n-1)); a `Retry-After` up to 30 s is honoured, a longer one fails at once |
| circuit breaker | after repeated failures, stop calling the provider for a while | opens after 3 failures in a row; one probe after 30 s |
| rate limit | at most N requests in any 60 seconds | 20 (`SYNTHIA_REMOTE_RPM`) |
| budget | at most N requests per UTC day, claimed before sending | 50 (`SYNTHIA_REMOTE_DAILY_CAP`) |

**Retry** (`synthia/gateway/retry.py`). Only a failure that may succeed when the
identical request is sent again is retried: HTTP 429, 408 and 5xx, a dropped
connection or a timeout. A rejected request, a bad key or an empty account is
raised at once, since sending it again changes nothing. The wait is "full
jitter": drawn at random between zero and the exponential ceiling, so many
clients that failed together do not retry together. Nothing is retried once a
chunk has been delivered, because the caller has already shown it.

**Circuit breaker** (`synthia/gateway/circuit.py`). Only the failures that say
something about the provider's health count toward opening it, which are the
same retryable ones. While open, calls fail immediately, which is what lets the
router go local at once. After the cooldown one probe is let through: success
closes the circuit, failure opens it for another cooldown. It sits inside the
retry so every failed attempt is evidence, not only every failed request.

**Rate limit** (`synthia/gateway/ratelimit.py`). The usual token bucket is not
safe here: a bucket of 20 per minute lets 20 through at once and then one every
3 seconds, which is 30 requests in the first 60 seconds. If the provider counts
over a sliding window, that is rate limited. This limiter keeps the time of each
recent request and admits a new one only while fewer than 20 fall inside the
last 60 seconds, which satisfies a fixed-window rule and a sliding one alike.
Waiters are served in arrival order; one cancelled while waiting takes no slot.

**Budget** (`synthia/gateway/budget.py`). OpenRouter's free models allow 50
requests per UTC day to an account that has bought less than 10 dollars of
credit (1000 at or above), and a request counts once it reaches the provider,
answered or not. So a request is claimed from the budget before it is sent, and
one that would exceed the cap is refused on this machine with the time the
budget resets. The ledger is a SQLite file in `SYNTHIA_HOME/db/gateway.db`, so
the count survives restarts and is shared by every process on the machine. A
claim is one `BEGIN IMMEDIATE` transaction, which takes the write lock before
reading, so two processes can never both see room for the last request. Days
are UTC days because the provider's counter resets at UTC midnight; a local
midnight would disagree with it for hours every day.

**Why the limiter comes before the claim.** A rate-limit wait can be cancelled
for free; a budget claim is never given back. Waiting first means a request
cancelled while it waits costs nothing (`synthia/gateway/metered.py`). Both sit
innermost, inside the retry, so every attempt is metered, exactly as the
provider meters attempts.

## Errors

Every failure the gateway raises on purpose is a `GatewayError` with a
`retryable` flag, and the retry and the circuit breaker decide from that flag
alone (`synthia/gateway/errors.py`).

| Error | Cause | Retryable |
| --- | --- | --- |
| `BadRequestError` | HTTP 400, 404, 413, 422: the request itself was rejected | no |
| `AuthError` | HTTP 401, 403: key missing, wrong, or not allowed this model | no |
| `PaymentRequiredError` | HTTP 402: no credit for this request | no |
| `RateLimitedError` | HTTP 429, with the provider's `Retry-After` when it sends one | yes |
| `ProviderError` | HTTP 5xx or 408, or a failure reported inside the stream | yes |
| `ConnectionFailedError` | no response at all: refused, reset or timed out | yes |
| `MalformedStreamError` | something that is not a valid chunk, such as a proxy's HTML page answering 200 | no |
| `IncompleteResponseError` | the stream ended before the answer was whole | no |
| `CircuitOpenError` | the circuit is open, so the provider was not called | no |
| `BudgetExhaustedError` | today's requests are used up; says when the budget resets | no |

A provider's error message is cut to 300 characters and scrubbed of the API key
before it appears anywhere, since some providers echo the key back.

## The local model

The local model is Qwen3.5-4B in the Q4_K_M quantisation (2.55 GB) with its
vision projector (0.63 GB), run by llama.cpp's own server
(`synthia/models/server.py`). `synthia models install` fetches the model
and the llama.cpp builds this machine can use, each checked against a pinned
SHA-256 digest, within the disk budget.

- **Private to this process.** The server listens on 127.0.0.1 only, on a port
  chosen fresh for each launch, and requires a random 32-byte key, also fresh
  for each launch. The key is passed in the server's environment
  (`LLAMA_API_KEY`), not on its command line, where other users could read it
  in the process list. So no other program, and no web page sending requests
  to localhost, can use the model.
- **Fastest build first, with a floor.** Installed builds are tried in the
  order Metal, CUDA, Vulkan, CPU (`synthia/models/backends.py`). A build that
  fails to start or never answers its health check within 180 seconds is
  skipped for the rest of the process and the next one is launched. The
  default install always includes the CPU build as the floor (on an Apple
  Silicon Mac the one build runs on Metal or the CPU), so there is always a
  last build to fall back to.
  `SYNTHIA_LOCAL_BACKEND` limits the choice to one build. The order is by the
  usual speed of each backend and has not been measured on a machine yet; the
  180 seconds allows for loading about 3 GB from a slow disk and is not
  measured either.
- **Supervised on its own thread.** The chat's event loop runs only during a
  turn, so a server supervised on it would load and be health-checked only then.
  It runs on a thread and event loop of its own instead
  (`synthia/models/service.py`), and restarts after a crash, each time on a new
  port with a new key.
- **Out of reach of Ctrl+C.** Pressing Ctrl+C stops the answer being streamed.
  The server is started in its own process group on Windows and its own session
  on Linux and macOS, so the terminal's interrupt does not also kill it and
  force a reload of 3 GB.
- **Context of 32,768 tokens** (`SYNTHIA_LOCAL_CONTEXT`). That is the smallest
  window any model behind `openrouter/free` has, so adding the local model never
  shrinks what the router can promise a caller.

## Accounting

Every finished call is recorded per UTC day and per model that actually
answered: calls, tokens and seconds (`synthia/gateway/usage.py`). Because
`openrouter/free` picks a model per request, a day's totals show which free
models served it. `synthia budget` prints today's figures; `synthia budget
--check` also asks OpenRouter for its own count of today's free-model requests,
which costs no request, and says whether the two agree. A difference means
requests were made that this machine did not meter.

## Two optional layers

These are built and tested, and a caller wraps a model in them where they fit.
The chat uses neither.

- **Cache** (`synthia/gateway/cache.py`). Serves an exact repeat of a request
  from SQLite for 7 days. The key is a SHA-256 over everything that shapes the
  answer: model, messages (images by the hash of their bytes), tools,
  temperature, token limit and schema. Only a complete answer is stored. It is
  meant for evaluations and repeated development runs; a conversation asking
  the same thing twice should not get the same words.
- **Structured output** (`synthia/gateway/structured.py`). Asks for JSON
  matching a pydantic model's schema and validates the reply whatever the
  model claims. A reply that fails is sent back with its exact errors, at most
  twice. Each repair is a new request, so it is routed and metered like any
  other.

## Tests never touch the network

Provider tests replay real exchanges recorded once into `tests/cassettes/`
(`synthia/gateway/cassette.py`), so CI spends no budget and runs offline. A
cassette stores no request headers at all, so an `Authorization` header can
never be written; the request body only as a hash of its canonical JSON plus a
short summary; response headers from an allowlist (`content-type`,
`retry-after`, `x-ratelimit-*`); and response bodies with every known secret
replaced. A test scans every cassette for anything shaped like a key or a bearer
token. `uv run python scripts/record_cassettes.py` records them again.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `SYNTHIA_OPENROUTER_API_KEY` | unset | the remote model's key; without it the chat does not start |
| `SYNTHIA_OPENROUTER_MODEL` | `openrouter/free` | picks a free model per request |
| `SYNTHIA_OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | change only to go through a proxy |
| `SYNTHIA_REMOTE_DAILY_CAP` | `50` | remote requests per UTC day |
| `SYNTHIA_REMOTE_RPM` | `20` | remote requests in any 60 seconds |
| `SYNTHIA_REMOTE_RESERVE` | `10` | at or below this many left, requests go local |
| `SYNTHIA_LOCAL_MODEL` | `qwen3.5-4b` | a name from `synthia models list` |
| `SYNTHIA_LOCAL_BACKEND` | `auto` | `auto`, `cpu`, `vulkan`, `cuda` or `metal` |
| `SYNTHIA_LOCAL_CONTEXT` | `32768` | tokens the local model holds |
