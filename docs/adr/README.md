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
| [0005](0005-model-gateway.md) | one gateway for every model; the budget is claimed before sending | accepted |
| [0006](0006-local-runtime-and-model.md) | llama.cpp's server and Qwen3.5-4B, pinned by digest, fastest build first | accepted |
| [0007](0007-personas-as-data.md) | personas as TOML data, five traits rendered to fixed sentences, blends | accepted |
