"""The ``synthia`` command."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import httpx
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from synthia import __version__
from synthia.agent.trace import TRACES, read_trace, sessions
from synthia.gateway.assemble import NO_MODEL
from synthia.gateway.providers import (
    OPENROUTER_FREE,
    load_providers,
    remote_key,
    select_remote,
)
from synthia.interfaces import consent_commands, key_commands, model_commands
from synthia.interfaces import doctor as doctor_checks
from synthia.interfaces.chat import run_chat, visible
from synthia.interfaces.daemon_client import (
    DaemonError,
    ask_daemon,
    consolidate_memory,
    current_daemon,
    daemon_status,
    indexed_from,
    removed_from,
    searched_from,
    source_from,
    sources_from,
    stop_running,
)
from synthia.interfaces.mcp_hosts import HOSTS_FILE, load_hosts
from synthia.interfaces.mcp_server import serve_host
from synthia.interfaces.trace_view import build_tree, summary
from synthia.interfaces.usage_report import budget_report
from synthia.kernel.config import (
    DEFAULT_ENV_FILE,
    Brain,
    Settings,
    api_key_variables,
    load_settings,
    unknown_variables,
)
from synthia.kernel.consent import ConsentBook, ConsentError, Host, Provider, Scope
from synthia.kernel.errors import ConfigError, SynthiaError
from synthia.kernel.jsonrpc import RpcError
from synthia.kernel.keys import KeyStore, SystemCredentials
from synthia.kernel.logs import logging_to
from synthia.knowledge.service import SEARCH_LIMIT
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.models.server import Launch
from synthia.models.service import SERVER_LOG, LocalService, LocalSetup, find_local
from synthia.persona.model import PersonaError
from synthia.server.api import MAX_SEARCH_LIMIT
from synthia.server.daemon import run_daemon
from synthia.server.upkeep import INDEX_IDLE_S

if TYPE_CHECKING:
    from collections.abc import Callable

    from synthia.interfaces.daemon_client import Expanded, FoundPassage, KnownSource
    from synthia.kernel.jsonrpc import Params
    from synthia.knowledge.index import IndexReport
    from synthia.models.catalogue import Item, Model

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

memory_app = typer.Typer(
    help="Look after what SYNTHIA remembers.", no_args_is_help=True
)
app.add_typer(memory_app, name="memory")

knowledge_app = typer.Typer(
    help="Add folders for SYNTHIA to read, and search what they hold.",
    no_args_is_help=True,
)
app.add_typer(knowledge_app, name="knowledge")

keys_app = typer.Typer(
    help="Keep remote providers' keys in the system key store.",
    no_args_is_help=True,
)
app.add_typer(keys_app, name="keys")

consent_app = typer.Typer(
    help="Decide which remote providers may receive what you type in chats, and "
    "which agents using SYNTHIA may receive private text.",
    no_args_is_help=True,
)
app.add_typer(consent_app, name="consent")

mcp_app = typer.Typer(
    help="Let another agent (Claude Code, Claude Desktop, Codex) search what "
    "SYNTHIA holds.",
    no_args_is_help=True,
)
app.add_typer(mcp_app, name="mcp")

CHAT_LOG = Path("logs") / "synthia.log"
MCP_LOG = Path("logs") / "mcp-serve.log"
SHOWN_BODY_LINES = 8
INDENT = "    "
SECONDS_PER_MINUTE = 60

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
    # Half an emoji (a lone surrogate, valid in JSON from a model or a server)
    # cannot be encoded, and printing it would end the command; show it escaped.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")


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
    settings = _settings()
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
    remote: Annotated[
        bool | None,
        typer.Option(
            help="Let turns go to the remote model from the start, as /remote on "
            "does, or keep them local. Default: SYNTHIA_BRAIN."
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            help="Take up the newest remembered conversation, as its last persona."
        ),
    ] = False,
) -> None:
    """Talk to SYNTHIA in the terminal, through the daemon.

    The daemon is started in the background if it is not running, and keeps
    running after the chat ends (``synthia stop`` stops it). Every turn stays
    on this machine unless remote is switched on. Ctrl+C stops an answer;
    Ctrl+C at the prompt, Ctrl+D or /exit leaves.
    """
    settings = _settings()
    use_remote = settings.brain is Brain.REMOTE if remote is None else remote
    try:
        if remote is None and use_remote and _refused(settings):
            typer.echo(_brain_refused(settings.remote))
            use_remote = False
        _require_a_model(settings)
        with logging_to(settings, settings.home / CHAT_LOG):
            daemon = asyncio.run(current_daemon(settings.home))
            run_chat(
                daemon, Console(), persona=persona, use_remote=use_remote, resume=resume
            )
    except (ConfigError, PersonaError, DaemonError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None


@app.command()
def serve() -> None:
    """Run SYNTHIA as a daemon until it is stopped.

    It listens on 127.0.0.1 only, on a free port written with a token to
    SYNTHIA_HOME/daemon.json, and logs to SYNTHIA_HOME/logs/synthia.log.
    Ctrl+C stops it.
    """
    settings = _settings()
    try:
        with logging_to(settings, settings.home / CHAT_LOG):
            asyncio.run(run_daemon(settings, local_service(settings)))
    except (ConfigError, PersonaError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    except KeyboardInterrupt:
        typer.echo("stopped")


@app.command()
def stop() -> None:
    """Stop the daemon.

    Answers still running are cancelled, and the local model and the MCP
    servers are stopped with it. The next chat starts it again.
    """
    try:
        stopped = asyncio.run(stop_running(_settings(stored_keys=False).home))
    except DaemonError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    typer.echo("daemon stopped" if stopped else "no daemon running")


@app.command()
def status() -> None:
    """Show whether the daemon runs, since when, and how its local model is."""
    found = asyncio.run(daemon_status(_settings().home))
    if found is None:
        typer.echo("no daemon running")
        return
    info = found.info
    since = info.started.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    typer.echo(
        f"daemon running: version {info.version}, since {since}, "
        f"pid {info.pid}, port {info.port}"
    )
    typer.echo(f"other conversations open: {found.others}")
    typer.echo(f"local model: {found.local}")


@memory_app.command("consolidate")
def memory_consolidate() -> None:
    """Merge the facts held about the person that the local model calls one.

    The daemon does it, since it holds the local model; it also does it on
    its own after ten minutes without a turn, at most once an hour. Each pair
    of close facts is asked about once; the older of a pair that is one fact
    is kept as replaced by the newer. Exits 2 when it could not run or the
    local model failed.
    """
    try:
        done = asyncio.run(consolidate_memory(_settings().home))
    except RpcError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    except KeyboardInterrupt:
        typer.echo("stopped; facts merged so far stay merged")
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    if done is None:
        typer.echo(
            "error: no daemon running; it holds the local model: "
            "start it with `synthia serve` or `synthia chat`",
            err=True,
        )
        raise typer.Exit(doctor_checks.Status.FAIL)
    pairs = "pair" if done.asked == 1 else "pairs"
    typer.echo(
        f"{done.asked} {pairs} of facts asked about, {done.merged} merged, "
        f"in {done.seconds:.1f} s"
    )
    if done.error is not None:
        typer.echo(f"error: {done.error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL)


@knowledge_app.command("add")
def knowledge_add(
    path: Annotated[Path, typer.Argument(help="The folder to read.")],
    *,
    private: Annotated[
        bool | None,
        typer.Option(
            "--private/--public",
            help="Mark it private, or not. "
            "Default: as before; a new folder is not private.",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            help="A folder or file inside it to leave out, relative to it; "
            "repeat for more. Replaces what it left out before.",
        ),
    ] = None,
) -> None:
    """Add a folder for SYNTHIA to read, or change how it reads one added before.

    A folder holding .obsidian is read as an Obsidian vault. The daemon
    reads it; it is started if it is not running. Exits 2 when the folder
    cannot be added.
    """
    params: Params = {"root": str(path.resolve()), "private": private}
    if exclude is not None:
        params["excludes"] = exclude
    source = source_from(_ask_knowledge("knowledge_add", params))
    typer.echo(_source_line(source))
    typer.echo(
        "run `synthia knowledge index` to read it now; the daemon also reads it "
        f"after {INDEX_IDLE_S / SECONDS_PER_MINUTE:g} minutes without a turn"
    )


@knowledge_app.command("remove")
def knowledge_remove(
    path: Annotated[Path, typer.Argument(help="The folder to stop reading.")],
) -> None:
    """Stop reading a folder, and forget everything read from it."""
    root = path.resolve()
    if removed_from(_ask_knowledge("knowledge_remove", {"root": str(root)})):
        typer.echo(f"removed {visible(str(root))}")
    else:
        typer.echo(f"nothing to remove: {visible(str(root))} was not added")


@knowledge_app.command("list")
def knowledge_list() -> None:
    """Show each folder SYNTHIA reads: name, kind, privacy, path, what it leaves out."""
    sources = sources_from(_ask_knowledge("knowledge_sources", {}))
    if not sources:
        typer.echo("no folders added; add one with `synthia knowledge add <folder>`")
    for source in sources:
        typer.echo(_source_line(source))


@knowledge_app.command("index")
def knowledge_index() -> None:
    """Read every folder now: new and changed files in, deleted ones out.

    The daemon also does it on its own after ten minutes without a turn, at
    most once an hour. Exits 2 when the pass failed.
    """
    try:
        done = indexed_from(_ask_knowledge("knowledge_index", {}))
    except KeyboardInterrupt:
        typer.echo("stopped; what was read so far is kept")
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    if done.report is None:
        typer.echo(f"error: {done.error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL)
    typer.echo(_index_line(done.report))
    if done.report.unavailable:
        names = ", ".join(visible(name) for name in done.report.unavailable)
        typer.echo(f"could not list {names}; what was read from it before is kept")


@knowledge_app.command("search")
def knowledge_search(
    question: Annotated[str, typer.Argument(help="What to look for.")],
    *,
    limit: Annotated[
        int,
        typer.Option(min=1, max=MAX_SEARCH_LIMIT, help="How many passages."),
    ] = SEARCH_LIMIT,
    variant: Annotated[
        str | None,
        typer.Option(
            help="How to rank: words, meaning, fused or fused:<word weight>, "
            "with +hop1x<weight> or +hop2x<weight> to follow links from the "
            "best 5 passages. Default: fused."
        ),
    ] = None,
    deep: Annotated[
        bool,
        typer.Option(
            help="First have the local model rewrite the question in the words "
            "files likely use and draft a passage that answers it, and search "
            "for those too: about 35 s on a laptop CPU."
        ),
    ] = False,
) -> None:
    """Find the passages of the folders read that best answer a question.

    Each shows where it is (name/path:start-end), its score and its first
    lines. Private folders are searched too: this is your own terminal.
    A deep search waits for the local model as long as it takes (up to
    136 s measured on a laptop CPU); Ctrl+C stops it.
    """
    params: Params = {"query": question, "limit": limit, "variant": variant}
    if deep:
        params["deep"] = True
    found = searched_from(_ask_knowledge("knowledge_search", params))
    if found.deep is not None:
        typer.echo(_deep_line(found.deep))
        if found.deep.note is not None:
            typer.echo(found.deep.note)
    if not found.passages:
        typer.echo("nothing found")
    for number, passage in enumerate(found.passages):
        if number:
            typer.echo("")
        for line in _passage_lines(passage):
            typer.echo(line)


def _ask_knowledge(method: str, params: Params) -> object:
    """Make one request of the daemon, or exit 2 saying why it could not be made."""
    settings = _settings()
    try:
        _require_a_model(settings)
        result = asyncio.run(ask_daemon(settings.home, method, params))
    except (ConfigError, DaemonError, RpcError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    return result


def _source_line(source: KnownSource) -> str:
    private = "private" if source.private else "not private"
    line = f"{source.name} | {source.kind.value} | {private} | {source.root}"
    if source.excludes:
        line += f" | leaves out {', '.join(source.excludes)}"
    return visible(line)


def _deep_line(deep: Expanded) -> str:
    count = len(deep.queries)
    rewrites = {0: "no rewrites", 1: "1 rewrite"}.get(count, f"{count} rewrites")
    drafted = "no drafted passage" if deep.passage is None else "a drafted passage"
    written = f"{rewrites} and {drafted}" if deep.written else "nothing usable written"
    return (
        f"deep: {written}, {deep.expand_s:.1f} s asking the local model, "
        f"{deep.search_s:.2f} s searching"
    )


def _index_line(report: IndexReport) -> str:
    return (
        f"files: {report.added} added, {report.changed} changed, "
        f"{report.unchanged} unchanged, {report.removed} removed, "
        f"{report.unreadable} unreadable, {report.held_back} held back, "
        f"{report.passed_over} passed over | chunks: {report.chunks} written, "
        f"{report.cut} cut, {report.embedded} embedded | {report.seconds:.1f} s"
    )


def _passage_lines(passage: FoundPassage) -> list[str]:
    private = " (private)" if passage.private else ""
    lines = [f"{visible(passage.citation)} | score {passage.score:.3f}{private}"]
    body = passage.body.splitlines()
    lines += [INDENT + visible(line) for line in body[:SHOWN_BODY_LINES]]
    hidden = len(body) - SHOWN_BODY_LINES
    if hidden > 0:
        more = "line" if hidden == 1 else "lines"
        lines.append(f"{INDENT}... {hidden} more {more}")
    return lines


@app.command()
def trace(
    session: Annotated[
        str | None,
        typer.Argument(help="The session to show. Default: the latest."),
    ] = None,
    *,
    list_all: Annotated[
        bool, typer.Option("--list", help="List every session instead.")
    ] = False,
) -> None:
    """Show what SYNTHIA did in a chat: each model step and tool call, as a tree.

    Exits 2 when there is no session by that name.
    """
    folder = _settings().home / TRACES
    found = sessions(folder)
    if not found:
        typer.echo(f"no traces yet; each chat writes one to {folder}")
        return
    if list_all:
        for path in found:
            typer.echo(summary(path))
        return
    by_name = {path.stem: path for path in found}
    path = found[-1] if session is None else by_name.get(session)
    if path is None:
        typer.echo(f"error: no trace named {session}", err=True)
        typer.echo("see: synthia trace --list", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL)
    records, unreadable = read_trace(path)
    Console().print(build_tree(path.stem, records, unreadable))


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

    Exits 1 when --check could not be completed, and 2 when the remote
    provider cannot be chosen.
    """
    try:
        report = asyncio.run(budget_report(_settings(), check=check))
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    for line in report.lines:
        typer.echo(line)
    if not report.complete:
        raise typer.Exit(doctor_checks.Status.WARN)


@app.command()
def providers() -> None:
    """List the remote providers: OpenRouter, built in, and SYNTHIA_HOME/providers.toml.

    The selected one is SYNTHIA_REMOTE, asked only after /remote on in a chat,
    with synthia chat --remote, or with SYNTHIA_BRAIN=remote. No part of a key
    is ever shown. Exits 2
    when providers.toml or SYNTHIA_REMOTE is invalid.
    """
    settings = _settings()
    try:
        selected = select_remote(settings)
        known = load_providers(settings.home)
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    for provider in known.values():
        chosen = "selected" if provider.name == selected.name else "not selected"
        key = "no key" if remote_key(settings, provider) is None else "key set"
        typer.echo(
            f"{provider.name}: {provider.base_url}, model {provider.model.id}, "
            f"{chosen}, {key}"
        )


@keys_app.command("set")
def keys_set(
    provider: Annotated[
        str, typer.Argument(help="The provider, as synthia providers names it.")
    ],
) -> None:
    """Store a provider's key in the system key store, asked without echo.

    The stored key wins over SYNTHIA_<NAME>_API_KEY in .env. Exits 2 when the
    provider is unknown, the key is not a key, or the store cannot be written.
    """
    settings = _settings(stored_keys=False)
    try:
        key_commands.require_provider(provider, load_providers(settings.home).keys())
        typed = typer.prompt(f"{provider} key", hide_input=True)
        said = key_commands.set_key(
            KeyStore(settings.home, SystemCredentials()), provider, typed
        )
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    typer.echo(said)


@keys_app.command("list")
def keys_list() -> None:
    """Say where each provider's key comes from, never any part of it."""
    settings = _settings(stored_keys=False)
    try:
        known = load_providers(settings.home)
        stored = KeyStore(settings.home, SystemCredentials()).load()
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    from_env = api_key_variables(DEFAULT_ENV_FILE)
    for line in key_commands.key_lines(known, stored.keys(), from_env):
        typer.echo(line)


@keys_app.command("remove")
def keys_remove(
    provider: Annotated[str, typer.Argument(help="The provider whose key to delete.")],
) -> None:
    """Delete a provider's key from the system key store.

    Exits 2 when the store cannot be written.
    """
    settings = _settings(stored_keys=False)
    try:
        said = key_commands.remove_key(
            KeyStore(settings.home, SystemCredentials()),
            provider,
            api_key_variables(DEFAULT_ENV_FILE),
        )
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    typer.echo(said)


@mcp_app.command("serve")
def mcp_serve(
    *,
    host: Annotated[
        str,
        typer.Option(
            help="The name the agent was installed under; SYNTHIA_HOME/mcp_hosts.toml "
            "and synthia consent decide what it may see."
        ),
    ],
) -> None:
    """Serve knowledge_search and memory_search to an agent over stdio (MCP).

    The agent starts this itself. Both tools only read, on this machine; a host
    sees the collections and memory mcp_hosts.toml lists for it, and private
    text only after synthia consent grant --host <host> --private. The daemon
    runs the searches and is started at the first one. Standard output carries
    only MCP messages; logs go to SYNTHIA_HOME/logs/mcp-serve.log.
    """
    settings = _settings(stored_keys=False)
    receiver = _host(host)
    stdout = sys.stdout.buffer

    def write(data: bytes) -> None:
        stdout.write(data)
        stdout.flush()

    with logging_to(settings, settings.home / MCP_LOG):
        asyncio.run(serve_host(settings.home, receiver, sys.stdin.buffer, write))


@consent_app.command("list")
def consent_list() -> None:
    """Say, for each provider and host, what may go to it and when that was decided.

    Exits 2 when the consent file or mcp_hosts.toml cannot be read or is damaged.
    """
    settings = _settings(stored_keys=False)
    try:
        lines = consent_commands.consent_lines(
            ConsentBook(settings.home),
            _receivers(settings),
            load_hosts(settings.home / HOSTS_FILE),
        )
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    for line in lines:
        typer.echo(line)


@consent_app.command("grant")
def consent_grant(
    provider: Annotated[
        str | None,
        typer.Argument(help="The provider, as synthia providers names it."),
    ] = None,
    *,
    host: Annotated[
        str | None,
        typer.Option(
            help="An agent running SYNTHIA as its MCP server, instead of a "
            "provider; needs --private."
        ),
    ] = None,
    private: Annotated[
        bool,
        typer.Option(
            "--private",
            help="Also let private memory and file passages go with the chat; "
            "for a host, let them go to it.",
        ),
    ] = False,
) -> None:
    """Let chats go to a provider once remote is on, or private text to a host.

    Exits 2 when the provider is unknown or the consent file cannot be written.
    """
    settings = _settings(stored_keys=False)
    book = ConsentBook(settings.home)
    chosen = _chosen(provider, host, private=private)
    try:
        if isinstance(chosen, Host):
            said = consent_commands.grant_host(book, chosen)
        else:
            said = consent_commands.grant(
                book, chosen, _receivers(settings), private=private
            )
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    typer.echo(said)


@consent_app.command("revoke")
def consent_revoke(
    provider: Annotated[
        str | None, typer.Argument(help="The provider to keep chats from.")
    ] = None,
    *,
    host: Annotated[
        str | None,
        typer.Option(help="An agent to keep private text from, instead of a provider."),
    ] = None,
    private: Annotated[
        bool,
        typer.Option(
            "--private", help="Keep only private passages away; chats still go."
        ),
    ] = False,
) -> None:
    """Keep every chat away from a provider, or private text from a host.

    Takes effect from the provider's next request or the host's next call.
    Exits 2 when the consent file cannot be written.
    """
    settings = _settings(stored_keys=False)
    chosen = _chosen(provider, host, private=True)
    book = ConsentBook(settings.home)
    try:
        receivers = _receivers(settings)
    except ConfigError:
        # A broken providers.toml must not stop a revoke; every address in
        # the consent file is still refused.
        receivers = {}
    try:
        said = (
            consent_commands.revoke_host(book, chosen)
            if isinstance(chosen, Host)
            else consent_commands.revoke(book, chosen, receivers, private=private)
        )
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None
    typer.echo(said)


def _refused(settings: Settings) -> bool:
    """Return whether the person said no to chats going to the selected provider.

    A damaged consent file is not a refusal here: the chat's own ``/remote on``
    reports it and stays local.

    Raises:
        ConfigError: If ``providers.toml`` or ``SYNTHIA_REMOTE`` is invalid.
    """
    receiver = select_remote(settings).receiver
    try:
        return ConsentBook(settings.home).decision(receiver, Scope.CHAT) is False
    except ConsentError:
        return False


def _brain_refused(provider: str) -> str:
    return (
        f"remote: stays off, you said no to chats going to {provider}; "
        f"/remote on asks again, synthia consent grant {provider} allows it"
    )


def _receivers(settings: Settings) -> dict[str, Provider]:
    return {
        name: provider.receiver
        for name, provider in load_providers(settings.home).items()
    }


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(doctor_checks.Status.FAIL)


def _host(name: str, *, private: bool = True) -> Host:
    """Return the host ``name``; exit 2 on a bad name, or without ``private``."""
    if not private:
        _fail(
            "a host is granted only private text, so add --private; what it may "
            f"see otherwise is set in SYNTHIA_HOME/{HOSTS_FILE}"
        )
    try:
        return Host(name=name)
    except ValidationError:
        _fail(
            "a host name is 1 to 32 lower-case letters, digits and hyphens, "
            "starting with a letter or digit"
        )


def _chosen(provider: str | None, host: str | None, *, private: bool) -> str | Host:
    """Return the provider named, or the host; exit 2 unless exactly one is named."""
    if host is not None and provider is None:
        return _host(host, private=private)
    if provider is not None and host is None:
        return provider
    _fail("name a provider, or a host with --host, not both")


def _settings(*, stored_keys: bool = True) -> Settings:
    unknown = unknown_variables()
    if unknown:
        typer.echo(
            f"warning: not a setting, ignored: {', '.join(unknown)} "
            "(.env.example lists every setting)",
            err=True,
        )
    try:
        return load_settings(stored_keys=stored_keys)
    except ConfigError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None


def _installer(settings: Settings) -> Installer:
    return Installer(settings.home, int(settings.disk_budget_gb * BYTES_PER_GB))


def _items(names: list[str]) -> tuple[Item, ...]:
    try:
        return model_commands.resolve(names)
    except KeyError as error:
        typer.echo(f"error: not in the catalogue: {error.args[0]}", err=True)
        typer.echo("see: synthia models list", err=True)
        raise typer.Exit(doctor_checks.Status.FAIL) from None


def local_service(
    settings: Settings, *, command: Callable[[Launch], list[str]] = Launch.command
) -> LocalService | None:
    """Return the installed local model's service, or None if none is installed.

    ``command`` builds the server's command line, so a caller may add flags
    such as a thread count.
    """
    setup = _local_setup(settings)
    if setup is None:
        return None
    return LocalService(setup, settings.home / SERVER_LOG, command=command)


def _local_setup(settings: Settings) -> LocalSetup | None:
    # The built-in remote's window, so moving a conversation from it never
    # shrinks what it holds; not the chosen remote's, whose window could need
    # more memory for the local model than this machine has.
    return find_local(
        settings,
        _installer(settings),
        model_commands.this_machine().target,
        context_limit=OPENROUTER_FREE.model.context_window,
    )


def _require_a_model(settings: Settings) -> None:
    """Refuse at once when there is no model at all, instead of a daemon that stops.

    Raises:
        ConfigError: If there is neither a remote key nor a local model, or
            the remote provider cannot be chosen.
    """
    key = remote_key(settings, select_remote(settings))
    if key is None and _local_setup(settings) is None:
        raise ConfigError(NO_MODEL)


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
            "error: no llama.cpp build suits this machine, so SYNTHIA cannot "
            "answer offline here",
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
