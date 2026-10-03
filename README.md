# personal-ai-os

**SYNTHIA, a personal AI operating system.** One system that listens, sees,
remembers, reasons and acts on the computer it runs on, in the spirit of JARVIS
rather than of a chat window. Built local-first, it runs on any device, even on a
potato device: local models on whatever GPU or CPU the machine has, so it keeps
working with no network, and outside services such as a remote model only when
you ask for them.

The design takes the "LLM OS" idea literally. The language model is the
processor, the context window is working memory, long-term memory and knowledge
are the disk, tools are system calls, the microphone, camera and screen are
peripherals, and a kernel schedules, supervises and guards all of it.

| OS concept | In SYNTHIA |
| --- | --- |
| kernel | typed event bus, service supervisor, settings, logging (`synthia.kernel`) |
| processor | local and remote language models behind one gateway |
| working memory | a context window that is budgeted, not just filled |
| disk | episodic memory, a vector index and knowledge sources with citations |
| system calls | tools with declared capabilities and approval for anything destructive |
| peripherals | voice in and out, camera, screen |
| shell | the `synthia` command first, then a daemon API and a web interface |

## Status

**Phase 2 of 13: agency.** On top of the kernel, SYNTHIA can think and act:
`synthia chat` talks through a model gateway that answers from a local model,
and from a free remote model, under a daily budget, only when asked, and it
uses tools, each call that changes something or leaves the machine asked for
first. The rest is built
phase by phase, and released when a capability works end to end.
[`docs/architecture.md`](docs/architecture.md) describes the whole design and
the order it is built in.

## Quickstart

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). Works on Linux and
Windows.

```bash
git clone https://github.com/Dibya2521/personal-ai-os
cd personal-ai-os
uv sync --all-groups
uv run synthia doctor
```

`doctor` checks what the rest of the system plans around: the Python version,
processor cores, memory, whether a CUDA GPU is present, whether the local model
and a llama.cpp build to run it are installed (without them SYNTHIA cannot
answer offline, so that is a failure), free disk against the configured
budget, and whether a remote model key is set (never its value; optional). Each
check reports `ok`, `warn` or `fail`; the command exits with the worst result,
`0`, `1` or `2`, and `--json` prints the same report for a script.

SYNTHIA answers from a model on your own machine. `uv run synthia models
install` shows what it would download for this machine (about 3.2 GB without an
NVIDIA GPU), asks, and installs it. Then:

```bash
uv run synthia chat
```

Every turn stays on the machine, with no key and no network needed. A stronger
free remote model is optional: put an [OpenRouter](https://openrouter.ai/keys)
key in `.env` as `SYNTHIA_OPENROUTER_API_KEY`, and turns go to it only after
`/remote on` in the chat (or `synthia chat --remote`). `/help` lists the
chat's commands.

## Configuration

Every setting is an environment variable prefixed `SYNTHIA_`, documented in
[`.env.example`](.env.example). Copy it to `.env` and fill in what you need;
`.env` is ignored by git. Settings are validated at start-up, secrets are held
as `SecretStr` and masked in every log line, and a test keeps `.env.example`
identical to the settings class.

## What is built

**Kernel.**

- **Event bus.** Components publish typed events instead of calling each
  other. Each subscriber has its own bounded queue and consumer task, so it sees
  events in publish order and never slows another subscriber down. A full queue
  either applies backpressure or drops the oldest event, chosen per subscriber.
- **Supervisor.** Long-running services restart on failure, one for one, in the
  Erlang/OTP model: permanent, transient and temporary restart modes, restart
  intensity that escalates a fault which is not transient, exponential backoff,
  and a graceful stop with a timeout for a service that ignores it.
- **Settings and logging.** Typed configuration from the environment, one log
  pipeline for the whole process in JSON or console form, correlation ids that
  follow a piece of work into the tasks it starts, and secret redaction applied
  to the fully formatted text.

**Cognition.** How a request is routed, guarded and answered is described
in [`docs/gateway.md`](docs/gateway.md).

- **Model gateway.** One streaming interface for every model, and one adapter
  for the OpenAI-compatible API that both OpenRouter and llama.cpp speak.
  Local first: a request goes to `openrouter/free` only when it asks, and is
  otherwise never sent out, even when the local model cannot serve it. A
  request that asks goes remote first, and to the local model for background
  jobs, for images or tools the remote cannot take, when the remote circuit
  is open, when 10 or fewer of the day's requests are left, or when the rate
  limit would make it wait over 5 seconds. A remote failure before the first
  word falls back to local once.
- **Guards.** Retry with full-jitter backoff, a circuit breaker, a
  sliding-window rate limit (20 in any 60 seconds) and a daily budget (the cap
  OpenRouter reports for the key, 50 on the free tier) claimed before a request
  leaves the machine, so the provider never
  sees a request over the limit. Every call is accounted per model, and
  `synthia budget --check` compares the count with OpenRouter's own.
- **Local model.** Qwen3.5-4B with vision, run by llama.cpp's server on
  loopback with a fresh port and key per launch. `synthia models` lists,
  installs and removes it and the llama.cpp builds, resumably, verified by
  SHA-256 and within the disk budget. Installed builds are tried in the order
  Metal, CUDA, Vulkan, CPU, and one that fails to start falls back to the
  next.
- **Personas.** SYNTHIA's character is data: five trait sliders rendered to
  fixed sentences, plus principles. Three pure characters: Nova (casual,
  sassy, tactical), Horizon (security and privacy first) and Zenith (formal
  and precise). The default SYNTHIA is mostly Nova, a good share of Horizon
  and some Zenith: a warm, quick-witted companion that guards the person's
  data. Neon, Glacier and Starlight each let one of the three lead with a
  little of the other two. Two more characters stand on their own: Minato, a
  calm mentor with nerves of steel and a kind heart, and Yume, a gentle
  companion for conversation. Warmth is a resting tone the moment can move
  either way. `/persona` switches or adjusts it mid-conversation, and a TOML
  file in `SYNTHIA_HOME/personas` adds or replaces one.
- **Chat.** Streaming answers with a line after each naming the route, the
  model, the tokens and the seconds taken. Ctrl+C stops an answer and keeps the
  chat; only finished exchanges enter the history. Log records go to
  `SYNTHIA_HOME/logs/synthia.log`, not into the conversation.

**Agency.** How SYNTHIA acts through tools is described in
[`docs/agency.md`](docs/agency.md), and why it is shaped this way in
[decision record 0010](docs/adr/0010-tool-calling-and-permissions.md).

- **Agent loop.** The model calls tools, reads their results and answers, for
  at most 8 steps; the last step offers no tools, so it must answer. Calls in
  one answer run at the same time, each stopped after 60 seconds unless the
  tool sets its own limit, and a failing tool returns its error instead of
  ending the turn.
- **Permissions.** Every tool states where its work happens (on this machine
  or outside it) and whether it changes anything. A tool that only reads on
  this machine runs at once; every other call asks first, one yes per call.
  The question shows the arguments whole, with terminal control characters
  escaped. The model's text cannot change any of this.
- **Tools.** The clock, an exact calculator, reading and listing files inside
  the folders named in `SYNTHIA_FILE_ROOTS`, and Python in a separate process
  (30 seconds, 20,000 bytes of output, a fresh empty folder, none of your
  environment variables). On command, a web page fetched as text (2,000,000
  bytes, 20 seconds) and a task handed to Claude Code or Gemini CLI when either
  is installed (10 minutes, the task sent on standard input). Any MCP server
  listed in `SYNTHIA_HOME/mcp.toml` adds its tools. Every process a tool
  starts is ended with it, children included; `docs/agency.md` names the
  two narrow ways a process can still escape.
- **Untrusted output.** Tool results reach the model quoted and marked
  untrusted, with chat control tokens defused; lines shaped like instructions
  are flagged on the call's line.
- **Trace.** Every turn, model step and tool call is written to
  `SYNTHIA_HOME/traces/`, and `synthia trace` shows a session as a tree.
- **Plan and execute.** A planner asks the model for a list of steps, runs
  each as its own loop, re-plans once if a step does not finish, and writes
  the answer from the results. It is built and tested, and not yet used by
  the chat.

**Repository.** Every commit passes ruff with every rule enabled, pyright in
strict mode, an import check that keeps each package from importing the ones
above it, and the test suite at 100 percent branch coverage of the package,
enforced by pre-commit and again by CI on Linux and Windows. Secret scanning and
a file-size limit guard the public history. Tests replay recorded model
responses and never reach the network, so CI spends no budget.

## Development

```bash
uv run pre-commit install
uv run pytest
```

[`CONTRIBUTING.md`](CONTRIBUTING.md) has the conventions and
[`docs/adr/`](docs/adr/README.md) records why the project is built the way it
is.

## Licence

[Apache-2.0](LICENSE).
