# Contributing

## Setup

```bash
uv sync --all-groups
uv run pre-commit install
```

## Gates

Every commit passes these, locally through pre-commit and again in CI on Linux
and Windows:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest --cov
```

## Conventions

- **Commits:** `type(scope): summary` in the imperative. Types are `feat`, `fix`,
  `docs`, `test`, `build`, `ci`, `bench`, `refactor`, `perf`, `chore` and
  `release`. The body explains why: the problem, any measurement, and the option
  chosen over the alternatives.
- **Docstrings:** Google convention. A one-line summary in the imperative, then
  detail only where a reader would otherwise get it wrong, and `Raises:` when the
  function raises.
- **Comments** explain why, never what.
- **Tests:** unit tests for mechanisms, property tests for invariants, and at
  least one test built from input the design did not anticipate. Coverage of
  the package stays at 100 percent of branches; a line coverage shows to be
  unreachable is deleted, not tested.
- **Numbers** in documentation come from a command that can be re-run.
- **Line endings** are LF everywhere.
- **Never commit** a `.env` file, a model file, a dataset or a recording.
