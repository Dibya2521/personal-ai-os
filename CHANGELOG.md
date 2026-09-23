# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

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

[0.1.0]: https://github.com/Dibya2521/personal-ai-os/releases/tag/v0.1.0
