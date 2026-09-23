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

**Phase 0 of 13: the foundation.** The kernel and the command line exist;
the intelligence is built on them phase by phase, and released when a
capability works end to end.
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
processor cores, memory, whether a CUDA GPU is present, free disk against the
configured budget, and whether a remote model key is set (never its value). Each
check reports `ok`, `warn` or `fail`; the command exits with the worst result,
`0`, `1` or `2`, and `--json` prints the same report for a script.

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

**Repository.** Every commit passes ruff with every rule enabled, pyright in
strict mode, and the test suite at 100 percent branch coverage of the package,
enforced by pre-commit and again by CI on Linux and Windows. Secret scanning and
a file-size limit guard the public history.

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
