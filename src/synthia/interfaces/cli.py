"""The ``synthia`` command."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from synthia import __version__
from synthia.interfaces import doctor as doctor_checks
from synthia.interfaces import model_commands
from synthia.interfaces.chat import run_chat
from synthia.interfaces.usage_report import budget_report
from synthia.kernel.config import Settings, load_settings
from synthia.kernel.errors import ConfigError, SynthiaError
from synthia.kernel.logs import logging_to
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.models.service import SERVER_LOG, LocalService, find_local
from synthia.persona.model import PersonaError

if TYPE_CHECKING:
    from synthia.models.catalogue import Model, Runtime

app = typer.Typer(
    name="synthia",
    help="SYNTHIA, a personal AI operating system.",
    no_args_is_help=True,
    add_completion=False,
)

models_app = typer.Typer(
    help="Install and remove the local runtime and models.", no_args_is_help=True
)
app.add_typer(models_app, name="models")

CHAT_LOG = Path("logs") / "synthia.log"

# A generous read timeout: a slow CDN may pause between chunks of a 2.7 GB file.
DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

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

    facts = doctor_checks.gather_facts(settings, _installer(settings))
    checks = doctor_checks.evaluate(settings, facts)
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
        with logging_to(settings, settings.home / CHAT_LOG):
            run_chat(
                settings, persona or settings.persona, Console(), local=_local(settings)
            )
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


def _settings() -> Settings:
    try:
        return load_settings()
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None


def _installer(settings: Settings) -> Installer:
    return Installer(settings.home, int(settings.disk_budget_gb * BYTES_PER_GB))


def _items(names: list[str]) -> tuple[Runtime | Model, ...]:
    try:
        return model_commands.resolve(names)
    except KeyError as error:
        typer.echo(f"error: not in the catalogue: {error.args[0]}", err=True)
        typer.echo("see: synthia models list", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None


def _local(settings: Settings) -> LocalService | None:
    target = model_commands.this_machine().target
    setup = find_local(settings, _installer(settings), target)
    return None if setup is None else LocalService(setup, settings.home / SERVER_LOG)


def _local_model(settings: Settings) -> Model:
    try:
        return model_commands.configured(settings.local_model)
    except KeyError:
        typer.echo(
            f"error: SYNTHIA_LOCAL_MODEL={settings.local_model} is not a model "
            "in the catalogue",
            err=True,
        )
        typer.echo("see: synthia models list", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None


def _yes(_: str) -> bool:
    return True


def _ask(question: str) -> bool:
    return typer.confirm(question, default=False)


@models_app.command("list")
def models_list() -> None:
    """Show everything that can be installed; * marks what suits this machine."""
    settings = _settings()
    table = model_commands.list_table(
        _installer(settings), model_commands.this_machine(), settings.local_model
    )
    Console().print(table)


@models_app.command("install")
def models_install(
    names: Annotated[
        list[str] | None,
        typer.Argument(help="What to install. Default: what suits this machine."),
    ] = None,
    *,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask first.")] = False,
) -> None:
    """Download, verify and install runtimes and models.

    Exits 1 when an install fails; running again resumes it.
    """
    settings = _settings()
    items = (
        _items(names)
        if names
        else model_commands.suggested(
            model_commands.this_machine(), _local_model(settings)
        )
    )
    if not items:
        typer.echo(
            "error: no llama.cpp build suits this machine; SYNTHIA runs remote only",
            err=True,
        )
        raise typer.Exit(doctor_checks.Status.FAIL)
    confirm = _yes if yes else _ask

    async def run() -> None:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
            await model_commands.install(
                _installer(settings), client, items, Console(), confirm
            )

    try:
        asyncio.run(run())
    except SynthiaError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.WARN) from None


@models_app.command("remove")
def models_remove(
    names: Annotated[list[str], typer.Argument(help="What to remove.")],
) -> None:
    """Delete installed runtimes or models, and any half-downloaded files."""
    items = _items(names)
    model_commands.remove(_installer(_settings()), items, Console())
