# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Deep search, only when the person asks for it: `synthia knowledge search
  "<question>" --deep` and `/deep <question>` in `synthia chat`. The local
  model rewrites the question as up to 3 search queries in the words files
  likely use and drafts a short passage that answers it; the question, the
  rewrites and the passage are searched alike, each passage keeping its best
  score. It runs on the local model only, so the question never leaves the
  machine, and a generation costs about 35 s per search on a laptop CPU
  (up to 136 s measured), so it is never the default and the chat model
  cannot start it. On 61 questions about this repository it raised nDCG@10
  from 0.461 to 0.486 overall, and by 0.08 on questions that name an
  identifier. `--deep` prints a line with what the model wrote and the
  seconds spent before the passages; `/deep` runs that search first, shown
  as a `search_knowledge` call the model made, and the model may search again
  before it answers. A model reply that never parses falls back to the plain
  search and says so.
- The chat model can search the person's folders added with
  `synthia knowledge add`: `search_knowledge` returns up to 6 passages, each
  under its citation (`name/path:start-end`) with the titles of the documents
  it links to, and `synthia chat` lists the citations shown under the answer
  (`sources` in the daemon's turn report). A private folder's passages are
  shown only in a chat a local model with tools can answer, which then stays
  on this machine until `/reset`; anywhere else they are withheld and counted.
- A chat that has read private text stays on this machine until `/reset`:
  every later request of it, the rest of the same turn included, goes to the
  local model with the route reason `chat holds private text`, and
  `/remote on` says why it stays off. Its turns are remembered marked, are
  never learned from, and reach another chat's reminder, `search_memory` or
  `--resume` only where that chat can stay local too, confining it first.
  `synthia chat` says once when a chat becomes confined, and puts a line
  saying the chat holds private text before each tool call it asks about.
- `/plan <task>` in `synthia chat`: SYNTHIA plans the task as steps, does each
  step with its tools, re-plans once if a step does not finish, then answers
  from the results. The chat shows the plan, each step as it begins and
  `answer:` before the answer; only the task and the answer are kept in the
  conversation. `synthia trace` shows the plan and each step's work under it.
- While the local model reads a long message before answering, `synthia chat`
  shows a dim "reading the message: N of M tokens, S s" line, updated each
  time the server reports a batch read (up to 2,048 tokens), then the
  thinking line or the answer.
- `synthia serve` runs SYNTHIA as a daemon on 127.0.0.1, on a free port
  written with a random token to `SYNTHIA_HOME/daemon.json` (readable only by
  its owner). It starts the local model and the MCP servers once for every
  conversation. `/health` answers anyone; conversations are JSON-RPC 2.0 over
  a WebSocket at `/ws`, need the token, and are refused from web pages (any
  request with an `Origin` header). Each client gets its own conversation, is
  asked before every tool call that needs a yes, and a client that goes away
  is a no. A client can stop the daemon; Ctrl+C stops it too.
- `synthia stop` stops the daemon: answers still running are cancelled, and
  the local model and the MCP servers stop with it. `synthia status` shows
  whether it runs, its version, since when, its pid and port, the other
  conversations open in it, and the local model: `ready`, `starting`
  (loading, or waiting to restart), `stopped` or `none`.
- The daemon keeps a trace of its own: each start, end and restart of the
  local model's server, with the error that ended it. `synthia trace` shows
  it like a chat's, and `synthia trace --list` counts its restarts.
- SYNTHIA remembers: every finished turn of every chat (the question, the
  answer, the persona, where it was answered, and the tool calls with their
  results cut to 2,000 characters) is kept in `SYNTHIA_HOME/memory.db`, on
  this machine. A stopped or failed turn is not kept. `/private on` stops
  remembering the rest of a chat, `/private off` starts again, and
  `/forget` takes the last turn out of the chat and out of memory.
- A `search_memory` tool lets SYNTHIA look back at earlier chats by words
  and days ("what did I tell you about my cat last week?"): it reads the
  clock for the dates, then returns up to eight earlier turns, best match
  first, each with its day. It only reads this machine, so it never asks.
  With an embedding model installed it also finds a turn by meaning ("what
  is my pet called?" finds "my cat is called Miso"), while a rare name or
  code in the question still counts; each turn is embedded when it is
  remembered, and turns from before are embedded in the background when
  `synthia serve` starts.
- With an embedding model installed, SYNTHIA is reminded of earlier
  conversations without having to search: before each question, up to three
  turns from other chats that come close to it are put in front of it,
  marked as SYNTHIA's memory and not the person's words. Only close turns
  qualify: on a 60-question recall set a reminder came with the turn the
  question was about for 25 % of questions, with something else for 3 %,
  and with nothing for the rest. The chat's history keeps the question
  alone, so reminders never fill the context window.
- SYNTHIA learns lasting facts about the person from each turn ("Has a
  sister in Pune.", "Prefers short answers."). With a local model and an
  embedding model installed, the daemon asks the local model in the
  background, never the remote one, after each remembered turn; private
  chats are never learned from. A newer fact that changes an older one ends
  the older one, and a near-identical repeat is not kept twice. On a
  24-exchange set with the expected facts written before any run, the local
  qwen3.5-4b found them with precision 0.952 and recall 0.95 (a found fact
  matching an expected one at cosine 0.7), and handled 8 of 8 changed
  facts. The facts held, newest first and up to about 1,000 tokens, go into
  each conversation's system prompt when it starts, after `/reset`, and
  again when a fact in it is forgotten; with `/remote on` they go to the
  remote model with the rest of the prompt. `/facts` lists them by number,
  `/forget fact <number>` forgets one, and `/forget about <words>` forgets
  every fact and every remembered turn holding all the words, with the
  facts learned from those turns, and says how many of each. A forgotten
  fact leaves the prompt of every open chat at its next turn; what was said
  in the current chat stays in it until `/reset`.
- The facts held are consolidated: each pair at cosine 0.80 or more goes to
  the local model, asked in both orders whether the two are one fact, and
  the older of a pair that is one is kept as replaced by the newer. On 28
  pairs that say the same fact and 28 that do not, written before any run,
  qwen3.5-4b merged 23 of the first and none of the second. A pair it calls
  two facts is remembered and never asked about again. The daemon does this
  after 10 minutes without a turn, at most once an hour and only when a turn
  was remembered since the last run or that run failed, and stops before its
  next question if a turn comes; `synthia memory consolidate` asks the running daemon to do
  it now and says how many pairs it asked about and merged. Each run goes to
  the daemon's trace, and `synthia trace --list` counts them.
- `synthia chat --resume` takes up the last conversation: the one holding
  the newest remembered turn, as the persona of that turn, found by its
  name (`--persona` still wins). Its turns come back as what the person
  asked and what SYNTHIA answered, without their tool calls, and new turns
  are remembered as part of it; the chat says the date and time of its last
  turn and how many turns came back. Only the newest turns that fit the
  context window come back; the chat says how many older ones did not, and
  `search_memory` can still find them.
- Long chats keep fitting in the model's context window. Before each turn
  the oldest turns leave, whole, until the request fits with a quarter of the
  window kept free for the answer. With a local model, the turns that left
  are folded into a short summary (at most 1,500 tokens, written on this
  machine, never sent to the remote model) that SYNTHIA keeps seeing; without
  one they are only dropped. How many characters make a token is learned
  from the prompt size the model reports for each turn.
- `synthia models` knows two sentence embedding models, for memory to find
  earlier turns by meaning: bge-small-en-v1.5 (MIT, 127.6 MB) and
  all-MiniLM-L6-v2 quantised to 8 bits (Apache-2.0, 22.4 MB), each pinned
  to a revision and to SHA-256 digests. `synthia models install` with no
  names now also installs all-MiniLM-L6-v2, which found what was asked
  about at least as often as bge-small on a 40-question recall set, in about
  half the time: on Windows without an NVIDIA card it
  fetches 3.25 GB instead of 3.23 GB.
- `synthia models` knows a reranker, the cross-encoder
  ms-marco-MiniLM-L6-v2 (Apache-2.0), which reads a question and a passage
  together and scores how well the passage answers it: quantised to 8 bits
  as `ms-marco-minilm-l6-int8` (22.8 MB) and at full precision as
  `ms-marco-minilm-l6` (87.5 MB), each pinned to a revision and to SHA-256
  digests. Neither is installed unless named.
- `synthia knowledge` reads folders the person adds and finds passages in
  them. `add <folder>` adds one (`--private`, `--exclude <path>`), `remove`
  forgets it and everything read from it, `list` shows them, `index` reads
  them now and prints what changed, and `search "<question>"` prints the
  best passages, each with where it is as `name/path:start-end`, its score
  and its first 8 lines, control characters shown escaped. The daemon does
  the work, started if needed, and also reads every folder again after 10
  minutes without a turn, at most once an hour, stopping before its next
  file when a turn starts or another command needs the index and going on
  at the next idle time; each pass goes to its trace. It needs an embedding
  model (`synthia models install`). Links between notes reach the note
  Obsidian would open: `[[Note]]` by file name without case, `[[Folder/Note]]`
  by its path from the folder's root, else by the path it ends with; when
  names repeat, notes in the linking note's folder come first, then the
  shorter path. Frontmatter `related` and `patterns` lists, Markdown links and
  Python imports are links too. `search --variant` picks the ranking (`words`,
  `meaning`, `fused`, `fused:<word weight>`); adding `+hop1x<weight>` or
  `+hop2x<weight>` follows the links of the best 5 passages one or two steps:
  each passage of a document reached scores its own match plus the weight
  times the best score among the passages linking to it.

### Changed

- `synthia chat` talks to the daemon instead of running SYNTHIA itself. It
  uses the running daemon, or starts one in the background and waits up to
  90 seconds for it; a daemon left from another version is restarted when no
  other chat is open in it. The chat looks and behaves as before: tool calls
  that need a yes are still asked in the terminal, and the daemon's start-up
  warnings (an MCP server that did not start) are shown when the chat opens.
  The conversation itself lives in the daemon, so it ends if the daemon
  stops; its remembered turns stay, and `synthia chat --resume` takes it up.
- Python 3.14 is supported and tested in CI on Linux and Windows, beside 3.12
  and 3.13. Python 3.15 is not yet: onnxruntime, which every model here runs
  on, ships no 3.15 wheel (none up to 1.31.0).

### Fixed

- A local answer no longer fails with "no response ... ReadTimeout" when the
  local model takes more than 60 seconds to read a long prompt before its
  first word, as the CPU build does on a busy machine. The local server's
  requests have no read limit (a crash still ends them at once, and Ctrl+C
  stops an answer); the remote model keeps its 60 seconds.

## [0.2.0] - 2026-10-03

SYNTHIA can think and act: a terminal chat through a model gateway that
answers from a local model on the machine, and from a free remote model, under
a daily budget, only when asked, and that uses tools, asking before every call
that changes something or leaves the machine.

### Added

- `synthia chat`: streaming answers from SYNTHIA, with `/persona`, `/persona
  set`, `/image`, `/think`, `/remote`, `/budget`, `/model`, `/reset`, `/help`
  and `/exit`. Every turn stays on the machine until `/remote on` (or
  `synthia chat --remote`) lets turns go to the remote model; `/remote off`
  keeps them local again. `/think off|low|medium|high|auto` sets how much
  each answer may think (default `auto`); `/think` alone shows it. While the model thinks, a
  dim "thinking N s" line counts the seconds and disappears when the answer
  starts; the thinking itself is never shown. After each answer one line
  names the route, the model, the thinking level sent, the tokens and the
  seconds taken. Ctrl+C stops an answer and keeps the chat; at the prompt, Ctrl+C,
  Ctrl+D or `/exit` leaves. An answer that failed or was stopped is dropped
  from the history, question and all. Log records go to
  `SYNTHIA_HOME/logs/synthia.log`, not into the conversation.
- Tools in `synthia chat`: SYNTHIA can call the clock, an exact calculator,
  and read or list files in the folders named in `SYNTHIA_FILE_ROOTS`. Each
  call shows as one dim line (the tool, its arguments, ok or why not, the
  seconds). A tool that would change something or reach outside the
  machine runs only after `y` at a `run <tool> <arguments>? [y/N]`
  question, one call at a time; arguments too long for that line are shown
  whole above it, with terminal control characters shown escaped. `/tools`
  lists every tool, where it works, what it may change, and whether it asks
  first. A tool's output reaches the
  model quoted as untrusted data, with chat markup in it defused; text in it
  shaped like an instruction is flagged on the call's line.
- `run_python` in `synthia chat`: SYNTHIA can run a Python program it
  writes, after a `y` for each run. It runs as a separate `python -I`
  process in a new empty directory (deleted afterwards), with none of your
  environment variables, so no keys. It is stopped after 30 seconds or 20,000
  bytes of output, and every process it started is ended with it (a Job
  Object on Windows, a process group elsewhere). It is not a sandbox: the
  code can read and change your files and reach the network.
- `fetch_url` and outside agents in `synthia chat`, each asking before every
  call. `fetch_url` reads an http or https page as text (HTML reduced to
  what a reader sees), at most 2,000,000 bytes (a longer page comes back
  marked as cut there) and 20 seconds, with a new
  client per call, so no cookie or key is carried. Claude Code and Gemini
  CLI, when found on PATH, become `ask_claude` and `ask_gemini`. The task
  is sent on standard input, never as an argument, so no command line
  re-parsing can run part of it. Each call may take up to 10 minutes, the
  agent's progress goes to `SYNTHIA_HOME/logs/agents/<name>.log`, and it
  sees no `SYNTHIA_*` variable. Agents that are not installed are not
  offered.
- MCP servers in `synthia chat`: servers listed in `SYNTHIA_HOME/mcp.toml`
  (`[servers.<name>]` with `command`, and optional `env`, `cwd`, `reach`,
  `effect`) start with the chat and their tools join as `<name>__<tool>`.
  They speak MCP over stdio (protocol versions 2024-11-05, 2025-03-26 and
  2025-06-18). A server counts as reaching outside and changing things unless
  the file says otherwise, so every call asks first. It never sees
  `SYNTHIA_*` variables, so no key. Its standard error goes to
  `SYNTHIA_HOME/logs/mcp/<name>.log`. A server that will not start, answers
  wrongly or dies costs only its own tools, and a broken `mcp.toml` is named
  in red while the chat goes on.
- `synthia trace`: every chat records each turn, model step and tool call as
  one JSON line in `SYNTHIA_HOME/traces/<session>.jsonl` (texts cut to 2,000
  characters with their full length kept; the key is never in it), and
  `synthia trace` shows the latest session as a tree: per model step the
  route, the model, the thinking level sent, tokens and seconds; per tool
  call its arguments, ok or failed, flags, seconds and the start of its
  result; then the answer, or why the turn ended without one.
  `synthia trace <session>` shows another, `synthia trace --list` lists them.
- A model gateway: one streaming interface for every model and one adapter for
  the OpenAI-compatible chat API, used for OpenRouter's `openrouter/free`,
  which picks a free model per request, and for llama.cpp's server.
- A thinking level on each request: `off`, `low`, `medium`, `high` or `auto`.
  OpenRouter receives `off` as `reasoning.effort` `none`, and `low` to `high`
  as a `reasoning.max_tokens` limit (the time allowance below);
  llama.cpp's server receives the chat template's `enable_thinking` switch. A
  request without a level is sent exactly as before.
- `auto` is decided by the router from the latest message, by fixed rules
  with no extra request: a greeting or short lookup does not think; code,
  mathematics, why and how questions, comparisons and proofs think more; a
  long message alone adds only a little.
- Each level is a time allowance, the same wait on any model: low 5 s, medium
  20 s, high 60 s. For OpenRouter it becomes `reasoning.max_tokens`: the
  allowance times the remote models' tokens per second measured over the last
  7 days, or 25 tokens/s before anything is measured.
- The local model's thinking ends on time: once it has thought for its
  level's allowance without starting the answer, llama.cpp's server is told
  to end the thinking (`POST /v1/chat/completions/control`, `reasoning_end`)
  and the answer follows, about 1 s later in a real run. If the server
  refuses, that is logged and the answer comes when the model stops by itself.
- Routing per request: local first. A request leaves the machine only when
  it asks for the remote model; otherwise it goes to the local model, waits
  for one that is still starting, or fails saying why (none installed, or it
  cannot take images or tools), and is never sent out instead. So SYNTHIA
  works with no key and no network, and a remote outage changes nothing
  unless remote was asked for. A request that asks gets the remote first;
  local for background jobs, for images or tools the remote cannot take,
  while the remote circuit is open, when 10 or fewer of the day's requests
  are left, or when the rate limit would hold a request over 5 seconds. When
  the remote fails before its first chunk, the request is sent to the local
  model once; after that nothing switches.
- Remote guards: retry of transient failures (3 attempts, full-jitter backoff,
  `Retry-After` honoured up to 30 s), a circuit breaker (opens after 3
  failures in a row, probes after 30 s), a sliding-window rate limit (20 in any
  60 s) and a daily budget claimed before each request is sent and refused
  locally once spent. The budget's cap is the one OpenRouter reports for the
  key (50 on the free tier, 1000 once 10 credits are bought), asked in the
  background when remote is first switched on in a chat, and by `synthia
  budget --check`, never before, and stored so
  it holds across restarts; 50 until OpenRouter has answered.
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
  Vulkan, CPU and falling back when one fails to start. A turn sent before it
  has loaded waits for it, up to the 180 s start timeout. It runs in its own
  process group, so Ctrl+C in the chat
  stops the answer without stopping the server and forcing a 3 GB reload. Its
  output goes to `SYNTHIA_HOME/logs/llama-server.log`.
- Personas as TOML data, with five trait sliders (warmth, formality, wit,
  vigilance, verbosity) and principles. Built in: three pure characters,
  `nova` (casual, sassy, watchful, brief), `horizon` (security and privacy
  first) and `zenith` (formal, precise, dry wit); the default `synthia`,
  `nova` 0.5, `horizon` 0.3 and `zenith` 0.2 with warmth 0.8, wit 0.7 and
  verbosity 0.5 pinned over the mean, so it is warm, polite, witty, watchful
  and answers fully; three led mixes, 0.7 of the lead and 0.15 of each other,
  `neon` (Nova), `glacier` (Horizon) and `starlight` (Zenith); `minato`, a
  calm, humble mentor with nerves of steel and a kind heart, inspired by
  Minato Namikaze in Naruto; and `yume`, a gentle companion for
  conversation, inspired by Kaoruko Waguri in The Fragrant Flower Blooms with
  Dignity. A blend may pin any trait over its mean this way. Warmth is a
  resting tone: a warm persona turns cool when the moment calls for it, and a
  cool one warms up. A file in `SYNTHIA_HOME/personas` adds a persona or
  replaces a built-in.
- Structured output validated against a pydantic model, with up to two repair
  requests, and an opt-in cache of complete answers, for callers that want
  them.
- Settings `SYNTHIA_PERSONA`, `SYNTHIA_LOCAL_MODEL`, `SYNTHIA_LOCAL_BACKEND`
  and `SYNTHIA_FILE_ROOTS` (the folders file tools may read; none by
  default), each documented in `.env.example`. Provider facts
  are not settings: the rate limit, the daily cap, the 10-request reserve and
  the local context (the remote's 32,768-token window, or less if the model's
  GGUF header says it was trained for less) are derived or fixed in code.
- A warning on stderr naming every `SYNTHIA_` variable, in the environment or
  `.env`, that is not a setting and so does nothing; an empty one counts too.
- `synthia doctor` reports whether the local model can run: which model and
  builds are installed, and a failure when they are not, or when
  `SYNTHIA_LOCAL_MODEL` names no known model, since SYNTHIA then cannot answer
  offline. No OpenRouter key is not a problem: the remote model is optional.
- Documentation of the gateway (`docs/gateway.md`) and decision records 0005
  to 0009, and of agency (`docs/agency.md`: tools, permissions, limits, MCP
  servers, the trace) with decision record 0010.
- An import layer check (import-linter, `uv run lint-imports`) in pre-commit
  and CI: `interfaces`, `mcp`, `tools`, `agent`, `models`, `gateway`,
  `persona`, `kernel`, each importing only the ones below it.

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

[Unreleased]: https://github.com/Dibya2521/personal-ai-os/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Dibya2521/personal-ai-os/releases/tag/v0.2.0
[0.1.0]: https://github.com/Dibya2521/personal-ai-os/releases/tag/v0.1.0
