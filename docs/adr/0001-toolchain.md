# 0001. Toolchain

Status: accepted.

## Decision

- **uv** manages Python, the virtual environment and the lockfile. CI installs
  with `uv sync --locked`, so a stale lockfile fails the build instead of
  drifting.
- **hatchling** builds the package from a `src/` layout. The distribution is
  `personal-ai-os`; the import package is `synthia`.
- **ruff** lints with every rule selected and a four-entry ignore list, and
  formats.
- **pyright** type-checks in strict mode.
- **pytest** with branch coverage, `hypothesis` for properties, and
  `pytest-asyncio` for the event loop. A warning fails the run.
- **pre-commit** runs all of it on every commit, plus `detect-secrets`, a size
  limit and line-ending checks. `pre-commit-uv` builds hook environments with uv.

## Options considered

- **pip and venv, or Poetry**, instead of uv. uv resolves and installs an order
  of magnitude faster and manages the interpreter too, which matters on CI
  across two operating systems.
- **mypy** instead of pyright. Both are sound choices; pyright's strict mode is
  stricter by default and faster on every commit.
- **pylint** alongside ruff. Ruff covers almost everything pylint would add here,
  and one fewer tool runs on every commit.
- **gitleaks** instead of detect-secrets. Gitleaks installs a Go toolchain
  through pre-commit, which is heavy on a development machine with little disk;
  detect-secrets is a Python package.

## Consequences

Starting strict costs little and rules out a class of defects from the first
line. When a rule does not fit a case, the exception is written next to the code
with its reason, so every relaxation is visible.

Development pins Python 3.12, the version the machine learning libraries this
project will depend on ship wheels for most reliably. CI also runs 3.13.
