# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

SYNTHIA can think: a terminal chat through a model gateway that routes between
a free remote model and an optional local one, under a daily budget.

### Added

- `synthia chat`: streaming answers from SYNTHIA, with `/persona`, `/persona
  set`, `/image`, `/budget`, `/model`, `/reset`, `/help` and `/exit`. After
  each answer one line names the route, the model, the tokens and the seconds
  taken. Ctrl+C stops an answer and keeps the chat; at the prompt, Ctrl+C,
  Ctrl+D or `/exit` leaves. An answer that failed or was stopped is dropped
  from the history, question and all. Log records go to
  `SYNTHIA_HOME/logs/synthia.log`, not into the conversation.
- A model gateway: one streaming interface for every model and one adapter for
  the OpenAI-compatible chat API, used for OpenRouter (`openrouter/free` by
  default) and for llama.cpp's server.
- A thinking level on each request: `off`, `low`, `medium`, `high` or `auto`.
  OpenRouter receives it as `reasoning.effort` (`none` for off), llama.cpp's
  server as the chat template's `enable_thinking` switch. A request without a
  level is sent exactly as before.
- `auto` is decided by the router from the latest message, by fixed rules
  with no extra request: a greeting or short lookup does not think; code,
  mathematics, why and how questions, comparisons and proofs think more; a
  long message alone adds only a little.
- Each level is a time allowance, the same wait on any model: low 5 s, medium
  20 s, high 60 s. For OpenRouter it becomes `reasoning.max_tokens`: the
  allowance times the remote models' tokens per second measured over the last
  7 days, or 25 tokens/s before anything is measured.
- Routing per request: remote first; local for background jobs, for images or
  tools the remote cannot take, while the remote circuit is open, when 10 or
  fewer of the day's requests are left, or when the rate limit would hold a
  request over 5 seconds. When the remote fails before its first chunk, the
  request is sent to the local model once; after that nothing switches.
- Remote guards: retry of transient failures (3 attempts, full-jitter backoff,
  `Retry-After` honoured up to 30 s), a circuit breaker (opens after 3
  failures in a row, probes after 30 s), a sliding-window rate limit (20 in any
  60 s) and a daily budget (50 per UTC day) claimed before each request is
  sent and refused locally once spent.
- `synthia budget`: today's remote requests, tokens and time per model;
  `--check` also asks OpenRouter for its own count, which costs no request,
  and says whether the two match.
- `synthia models list`, `install` and `remove`: the local model (Qwen3.5-4B
  with its vision projector) and the llama.cpp builds for this machine. With
  no names, install takes the model named by `SYNTHIA_LOCAL_MODEL`, the build
  likely fastest here, and the CPU build as a fallback. It shows what it will
  download and asks first (`--yes` skips the question), resumes an
  interrupted download, keeps a file only when its SHA-256 matches, and
  refuses to go over the disk budget or leave less than 1 GB free.
- The chat starts the local model when it is installed, on loopback with a
  fresh port and key per launch, trying builds in the order Metal, CUDA,
  Vulkan, CPU and falling back when one fails to start. Until it has loaded,
  requests go remote. It runs in its own process group, so Ctrl+C in the chat
  stops the answer without stopping the server and forcing a 3 GB reload. Its
  output goes to `SYNTHIA_HOME/logs/llama-server.log`.
- Personas as TOML data, with five trait sliders (warmth, formality, wit,
  vigilance, verbosity) and principles. Built in: `jarvis`, `companion`,
  `edith`, and the default `synthia`, a weighted blend of the three. A file in
  `SYNTHIA_HOME/personas` adds a persona or replaces a built-in.
- Structured output validated against a pydantic model, with up to two repair
  requests, and an opt-in cache of complete answers, for callers that want
  them.
- Settings `SYNTHIA_OPENROUTER_MODEL`, `SYNTHIA_OPENROUTER_BASE_URL`,
  `SYNTHIA_REMOTE_DAILY_CAP`, `SYNTHIA_REMOTE_RPM`, `SYNTHIA_REMOTE_RESERVE`,
  `SYNTHIA_PERSONA`, `SYNTHIA_LOCAL_MODEL`, `SYNTHIA_LOCAL_BACKEND` and
  `SYNTHIA_LOCAL_CONTEXT`, each documented in `.env.example`.
- `synthia doctor` reports whether the local model can run: which model and
  builds are installed, a warning when they are not (the remote model alone
  still works), and a failure when `SYNTHIA_LOCAL_MODEL` names no known model.
- Documentation of the gateway (`docs/gateway.md`) and decision records 0005
  to 0007.

### Changed

- `synthia doctor`'s accelerator line without an NVIDIA driver now reads "no
  NVIDIA driver, so no CUDA build" instead of "no CUDA GPU, inference runs on
  the CPU", which stopped being true once a Vulkan build can use other GPUs.

## [0.1.0] - 2026-09-23

The foundation: a kernel for everything else to stand on, and a repository
that guards itself.

### Added

- `synthia doctor`, which reports Python, operating system, cores, memory,
  CUDA availability, free disk against the budget, and whether a remote model
  key is set, exiting with the worst result.
- `synthia --version`.
- Typed settings from `SYNTHIA_` environment variables and `.env`, with secrets
  held as `SecretStr` and `.env.example` kept identical to the settings by a
  test.
- Structured logging in JSON or console form, with correlation ids that follow
  work into the tasks it starts and secret redaction on the formatted text.
- A typed asynchronous event bus with per-subscriber bounded queues and a
  choice of backpressure or drop-oldest on overflow.
- A one-for-one service supervisor with permanent, transient and temporary
  restart modes, restart intensity, exponential backoff and a graceful stop
  with a timeout.
- CI on Linux and Windows with Python 3.12 and 3.13; pre-commit with ruff,
  pyright strict, pytest, secret scanning and a file-size limit.
- Architecture overview and decision records 0001 to 0004.

[Unreleased]: https://github.com/Dibya2521/personal-ai-os/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Dibya2521/personal-ai-os/releases/tag/v0.1.0
