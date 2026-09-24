"""The ``synthia`` command."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from synthia import __version__
from synthia.interfaces import doctor as doctor_checks
from synthia.interfaces.chat import run_chat
from synthia.interfaces.usage_report import budget_report
from synthia.kernel.config import load_settings
from synthia.kernel.errors import ConfigError
from synthia.persona.model import PersonaError

app = typer.Typer(
    name="synthia",
    help="SYNTHIA, a personal AI operating system.",
    no_args_is_help=True,
    add_completion=False,
)

STATUS_STYLE = {
    doctor_checks.Status.OK: "[green]ok[/]",
    doctor_checks.Status.WARN: "[yellow]warn[/]",
    doctor_checks.Status.FAIL: "[red]fail[/]",
}


def _print_version(value: bool) -> None:  # noqa: FBT001 - typer passes the flag positionally
    if value:
        typer.echo(f"synthia {__version__}")
        raise typer.Exit


@app.callback()
def main(
    *,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = False,
) -> None:
    """SYNTHIA, a personal AI operating system."""
    del version


@app.command()
def doctor(
    *,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the report as JSON.")
    ] = False,
) -> None:
    """Check this machine and the configuration.

    Exits 0 when every check passes, 1 on a warning and 2 on a failure.
    """
    try:
        settings = load_settings()
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None

    checks = doctor_checks.evaluate(settings, doctor_checks.gather_facts(settings.home))
    if as_json:
        typer.echo(
            json.dumps(
                [asdict(c) | {"status": c.status.name.lower()} for c in checks],
                indent=2,
            )
        )
    else:
        table = Table(show_header=False, box=None, pad_edge=False)
        for check in checks:
            table.add_row(STATUS_STYLE[check.status], check.name, check.detail)
        Console().print(table)
    raise typer.Exit(doctor_checks.overall(checks))


@app.command()
def chat(
    *,
    persona: Annotated[
        str | None,
        typer.Option(help="Persona to start as. Default: SYNTHIA_PERSONA."),
    ] = None,
) -> None:
    """Talk to SYNTHIA in the terminal.

    Ctrl+C stops an answer; Ctrl+C at the prompt, Ctrl+D or /exit leaves.
    """
    try:
        settings = load_settings()
        run_chat(settings, persona or settings.persona, Console())
    except (ConfigError, PersonaError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None


@app.command()
def budget(
    *,
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="Also ask OpenRouter for its own count; not a model request.",
        ),
    ] = False,
) -> None:
    """Show today's remote requests and tokens, per UTC day.

    Exits 1 when --check could not be completed.
    """
    try:
        settings = load_settings()
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    report = asyncio.run(budget_report(settings, check=check))
    for line in report.lines:
        typer.echo(line)
    if not report.complete:
        raise typer.Exit(doctor_checks.Status.WARN)
