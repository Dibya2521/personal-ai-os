# personal-ai-os

**SYNTHIA, a personal AI operating system.** One system that listens, sees,
remembers, reasons and acts on the computer it runs on, in the spirit of JARVIS
rather than of a chat window. Built local-first, it runs on any device, even on a
potato device: local models on whatever GPU or CPU the machine has, a remote
model when the machine cannot carry one.

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

**Phase 1 of 13: cognition.** On top of the kernel, SYNTHIA can now think:
`synthia chat` talks through a model gateway that routes each request between
a free remote model and a local one, under a daily budget. The rest is built
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
and a llama.cpp build to run it are installed, free disk against the configured
budget, and whether a remote model key is set (never its value). Each
check reports `ok`, `warn` or `fail`; the command exits with the worst result,
`0`, `1` or `2`, and `--json` prints the same report for a script.

To chat, put a free [OpenRouter](https://openrouter.ai/keys) key in `.env` as
`SYNTHIA_OPENROUTER_API_KEY`, then:

```bash
uv run synthia chat
```

`/help` lists the chat's commands. The local model is optional: `uv run
synthia models install` shows what it would download for this machine (about
3.2 GB without an NVIDIA GPU), asks, and installs it; from then on the chat
starts it too.

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
  for the OpenAI-compatible API that both OpenRouter and llama.cpp speak. Each
  request goes to `openrouter/free` first and to the local model for
  background jobs, for images or tools the remote cannot take, when the
  remote circuit is open, when 10 or fewer of the day's requests are left, or
  when the rate limit would make it wait over 5 seconds. A remote failure before the first word falls back to local once.
- **Guards.** Retry with full-jitter backoff, a circuit breaker, a
  sliding-window rate limit (20 in any 60 seconds) and a daily budget (50 per
  UTC day) claimed before a request leaves the machine, so the provider never
  sees a request over the limit. Every call is accounted per model, and
  `synthia budget --check` compares the count with OpenRouter's own.
- **Local model.** Qwen3.5-4B with vision, run by llama.cpp's server on
  loopback with a fresh port and key per launch. `synthia models` lists,
  installs and removes it and the llama.cpp builds, resumably, verified by
  SHA-256 and within the disk budget. Installed builds are tried in the order
  Metal, CUDA, Vulkan, CPU, and one that fails to start falls back to the
  next.
- **Personas.** SYNTHIA's character is data: five trait sliders rendered to
  fixed sentences, plus principles. The default blends JARVIS, a warm
  companion and EDITH; `/persona` switches or adjusts it mid-conversation, and
  a TOML file in `SYNTHIA_HOME/personas` adds or replaces one.
- **Chat.** Streaming answers with a line after each naming the route, the
  model, the tokens and the seconds taken. Ctrl+C stops an answer and keeps the
  chat; only finished exchanges enter the history. Log records go to
  `SYNTHIA_HOME/logs/synthia.log`, not into the conversation.

**Repository.** Every commit passes ruff with every rule enabled, pyright in
strict mode, and the test suite at 100 percent branch coverage of the package,
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
