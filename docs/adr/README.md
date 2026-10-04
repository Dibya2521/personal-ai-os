# Architecture decision records

Each record states a decision, the options that were considered, and why this
one won. A record is never rewritten; a reversed decision gets a new record that
supersedes the old one.

| Record | Decision | Status |
| --- | --- | --- |
| [0001](0001-toolchain.md) | uv, ruff with every rule, pyright strict, pytest, pre-commit | accepted |
| [0002](0002-events-and-supervision.md) | an event bus and an OTP-style supervisor at the core | accepted |
| [0003](0003-configuration-and-secrets.md) | configuration from the environment, secrets masked everywhere | accepted |
| [0004](0004-local-first.md) | local first, optional weight, remote models behind a budget | accepted |
| [0005](0005-model-gateway.md) | one gateway for every model; the budget is claimed before sending | accepted; routing superseded by 0009 |
| [0006](0006-local-runtime-and-model.md) | llama.cpp's server and Qwen3.5-4B, pinned by digest, fastest build first | accepted |
| [0007](0007-personas-as-data.md) | personas as TOML data, five traits rendered to fixed sentences, blends | accepted |
| [0008](0008-adaptive-reasoning.md) | thinking depth as a time allowance: token limit remote, stopped live locally | accepted |
| [0009](0009-self-contained.md) | self-contained: local first, the remote and other outside services only on command | accepted |
| [0010](0010-tool-calling-and-permissions.md) | tool calling: the model asks, code decides from reach and effect; output is quoted data | accepted |
| [0011](0011-daemon.md) | one daemon holds SYNTHIA; interfaces speak JSON-RPC 2.0 to it over a local, token-guarded WebSocket | accepted |
