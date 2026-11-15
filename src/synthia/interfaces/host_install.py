"""What ``synthia install`` does: add SYNTHIA to another agent's MCP servers.

Each host keeps its MCP servers in its own config. Claude Desktop reads
``claude_desktop_config.json``, Codex reads ``~/.codex/config.toml`` and Claude
Code is changed through its own ``claude mcp`` command. A change is planned
first, so it can be shown and asked about; applying it backs up the old file
next to it and writes the new one in one step. Every other server the person
has is kept, and a second run changes nothing. Installing grants no access:
what a host may see is still decided in ``SYNTHIA_HOME/mcp_hosts.toml`` and by
``synthia consent``.
"""

from __future__ import annotations

import difflib
import json
import shlex
import subprocess
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from synthia.interfaces.mcp_hosts import HOSTS_FILE
from synthia.kernel.config import environment_without_own
from synthia.kernel.errors import ConfigError
from synthia.kernel.files import write_text_atomic

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import datetime

MAX_HOST_CONFIG_BYTES: Final = 16 * 1024 * 1024
SERVER_NAME: Final = "synthia"
BLOCK_START: Final = "# >>> synthia install >>>"
BLOCK_END: Final = "# <<< synthia install <<<"
COMMAND_TIMEOUT_S: Final = 60
MAX_SHOWN_ERROR: Final = 500
BACKUP_STAMP: Final = "%Y%m%dT%H%M%SZ"
MAX_BACKUP_TRIES: Final = 100
LEFT_AS_IT_IS: Final = "left as it is"
SECRET_KEYS: Final = frozenset({"env", "headers"})
HIDDEN: Final = "***"
# cmd.exe re-parses the arguments of a batch file, quotes and % included.
BATCH_SUFFIXES: Final = (".cmd", ".bat")


class InstallError(ConfigError):
    """A host's config cannot be changed safely, so nothing was changed."""


class HostTarget(StrEnum):
    """An agent SYNTHIA can be installed into; the value is its ``--host`` name."""

    CLAUDE_CODE = "claude-code"
    CLAUDE_DESKTOP = "claude-desktop"
    CODEX = "codex"

    @property
    def product(self) -> str:
        """Return the agent's name as people know it."""
        return _PRODUCTS[self]


_PRODUCTS: Final = {
    HostTarget.CLAUDE_CODE: "Claude Code",
    HostTarget.CLAUDE_DESKTOP: "Claude Desktop",
    HostTarget.CODEX: "Codex",
}


@dataclass(frozen=True, slots=True)
class ServerLaunch:
    """The command a host runs to start ``synthia mcp serve`` for itself."""

    command: str
    args: tuple[str, ...]

    @classmethod
    def of(cls, target: HostTarget, *, executable: str) -> ServerLaunch:
        """Return the launch through ``executable``, the running interpreter.

        The interpreter is used rather than the ``synthia`` script, whose path
        differs under ``uv run``, pipx and uvx.

        Raises:
            InstallError: If ``executable`` is empty, as for an embedded Python.
        """
        if not executable:
            message = "cannot tell which Python is running SYNTHIA, so nothing changed"
            raise InstallError(message)
        return cls(executable, ("-m", "synthia", "mcp", "serve", "--host", target))

    def entry(self) -> dict[str, object]:
        """Return the server entry as the hosts' configs hold it."""
        return {"command": self.command, "args": list(self.args)}


@dataclass(frozen=True, slots=True)
class HostPaths:
    """Where each host keeps its MCP servers, and the ``claude`` program if any.

    ``windows`` says how a command printed for the person is quoted.
    """

    desktop_config: Path
    codex_config: Path
    claude_json: Path
    claude_exe: str | None
    windows: bool = False

    @classmethod
    def find(
        cls,
        *,
        environ: Mapping[str, str],
        home: Path,
        platform: str,
        which: Callable[[str], str | None],
    ) -> HostPaths:
        """Return the paths each host uses on ``platform`` for the user at ``home``."""
        if platform == "win32":
            app_data = Path(environ.get("APPDATA") or home / "AppData" / "Roaming")
        elif platform == "darwin":
            app_data = home / "Library" / "Application Support"
        else:
            app_data = home / ".config"
        codex_home = environ.get("CODEX_HOME")
        return cls(
            desktop_config=app_data / "Claude" / "claude_desktop_config.json",
            codex_config=(Path(codex_home) if codex_home else home / ".codex")
            / "config.toml",
            claude_json=home / ".claude.json",
            claude_exe=which("claude"),
            windows=platform == "win32",
        )


@dataclass(frozen=True, slots=True)
class FileChange:
    """Replace ``path``'s text ``before`` (None: no file yet) with ``after``.

    ``preview`` is the diff the person is shown, without other servers' secrets.
    """

    path: Path
    before: str | None
    after: str
    preview: str


@dataclass(frozen=True, slots=True)
class CommandChange:
    """Run each argument list in turn, after backing up ``backup_of``."""

    commands: tuple[tuple[str, ...], ...]
    backup_of: Path


@dataclass(frozen=True, slots=True)
class NoChange:
    """Nothing to do, and why."""

    reason: str


type Change = FileChange | CommandChange | NoChange


def plan(
    target: HostTarget, launch: ServerLaunch, paths: HostPaths, *, uninstall: bool
) -> Change:
    """Return what installing ``launch`` into ``target``, or removing it, changes.

    Nothing is written here. For Claude Desktop and Claude Code, a ``synthia``
    server that differs is replaced, after the person sees the change; for
    Codex, only the block between SYNTHIA's marker lines is ever changed.

    Raises:
        InstallError: If the host's config cannot be read or is not what that
            host writes, if Codex has a ``synthia`` table outside the markers,
            or if ``claude`` is missing or a batch file.
    """
    match target:
        case HostTarget.CLAUDE_DESKTOP:
            return _plan_desktop(
                paths.desktop_config.resolve(), launch, uninstall=uninstall
            )
        case HostTarget.CODEX:
            return _plan_codex(
                paths.codex_config.resolve(), launch, uninstall=uninstall
            )
        case (
            HostTarget.CLAUDE_CODE
        ):  # pragma: no branch - pyright checks every case is matched
            return _plan_claude_code(paths, launch, uninstall=uninstall)


def _read_capped(path: Path) -> str | None:
    """Return ``path``'s text, or None if there is no such file.

    Raises:
        InstallError: If it cannot be read, is too large, or is not UTF-8.
    """
    try:
        with path.open("rb") as file:
            data = file.read(MAX_HOST_CONFIG_BYTES + 1)
    except FileNotFoundError:
        return None
    except OSError as error:
        message = f"cannot read {path}: {error.strerror}"
        raise InstallError(message) from None
    if len(data) > MAX_HOST_CONFIG_BYTES:
        message = (
            f"{path} is larger than {MAX_HOST_CONFIG_BYTES} bytes; {LEFT_AS_IT_IS}"
        )
        raise InstallError(message)
    try:
        return data.decode()
    except UnicodeDecodeError:
        message = f"{path} is not UTF-8 text; {LEFT_AS_IT_IS}"
        raise InstallError(message) from None


def _no_constant(name: str) -> object:
    message = f"{name} is not a JSON number"
    raise ValueError(message)


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    # A repeated key would silently lose all but its last value on rewrite.
    found = dict(pairs)
    if len(found) != len(pairs):
        message = "a JSON object repeats a key"
        raise ValueError(message)
    return found


def _json_config(path: Path, text: str | None) -> dict[str, object]:
    """Return the JSON object in ``text``; an empty one when there is no file.

    Raises:
        InstallError: If ``text`` is not a JSON object.
    """
    if text is None:
        return {}
    try:
        parsed: object = json.loads(
            text, parse_constant=_no_constant, object_pairs_hook=_unique_keys
        )
    except (ValueError, RecursionError) as error:
        message = f"{path} is not valid JSON ({error}); {LEFT_AS_IT_IS}"
        raise InstallError(message) from None
    if not isinstance(parsed, dict):
        message = f"{path} does not hold a JSON object; {LEFT_AS_IT_IS}"
        raise InstallError(message)
    return cast("dict[str, object]", parsed)


def _json_servers(path: Path, config: Mapping[str, object]) -> dict[str, object]:
    """Return the ``mcpServers`` object in ``config``; empty when it has none.

    Raises:
        InstallError: If ``mcpServers`` is there but not an object.
    """
    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        message = f"mcpServers in {path} is not a JSON object; {LEFT_AS_IT_IS}"
        raise InstallError(message)
    return cast("dict[str, object]", servers)


def _plan_desktop(path: Path, launch: ServerLaunch, *, uninstall: bool) -> Change:
    text = _read_capped(path)
    config = _json_config(path, text)
    servers = _json_servers(path, config)
    if uninstall:
        if SERVER_NAME not in servers:
            return NoChange(f"{path} has no {SERVER_NAME} server")
        kept = {name: v for name, v in servers.items() if name != SERVER_NAME}
    elif servers.get(SERVER_NAME) == launch.entry():
        return NoChange(f"{path} already starts SYNTHIA this way")
    else:
        kept = servers | {SERVER_NAME: launch.entry()}
    new_config = config | {"mcpServers": kept}
    after = _json_text(path, new_config)
    shown_before = "" if text is None else _json_text(path, _masked(config))
    preview = _diff(path, text, shown_before, _json_text(path, _masked(new_config)))
    return FileChange(path, text, after, preview)


def _json_text(path: Path, value: object) -> str:
    """Return ``value`` as the JSON written back.

    Raises:
        InstallError: If it holds a number JSON cannot write, such as 1e400.
    """
    try:
        return json.dumps(value, indent=2, allow_nan=False) + "\n"
    except ValueError:
        message = f"{path} holds a number too large for JSON; {LEFT_AS_IT_IS}"
        raise InstallError(message) from None


def _masked(value: object, *, secret: bool = False) -> object:
    """Return ``value`` with every value under an ``env`` or ``headers`` key hidden."""
    if isinstance(value, dict):
        fields = cast("dict[str, object]", value)
        return {
            key: _masked(item, secret=secret or key in SECRET_KEYS)
            for key, item in fields.items()
        }
    if isinstance(value, list):
        return [_masked(item, secret=secret) for item in cast("list[object]", value)]
    return HIDDEN if secret else value


def _diff(
    path: Path,
    before: str | None,
    shown_before: str,
    shown_after: str,
    context: int = 3,
) -> str:
    return "".join(
        difflib.unified_diff(
            shown_before.splitlines(keepends=True),
            shown_after.splitlines(keepends=True),
            fromfile=str(path) if before is not None else "/dev/null",
            tofile=str(path),
            n=context,
        )
    )


def _parse_toml(text: str) -> dict[str, object] | None:
    try:
        return tomllib.loads(text)
    except (tomllib.TOMLDecodeError, RecursionError):
        return None


def _split_block(path: Path, text: str) -> tuple[str, str | None, str]:
    """Return the text before SYNTHIA's block, the block (None if absent), and after.

    Raises:
        InstallError: If the block's marker lines do not pair up.
    """
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == BLOCK_START]
    ends = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == BLOCK_END]
    if not starts and not ends:
        return text, None, ""
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        message = (
            f"{path} has a damaged synthia block: its {BLOCK_START!r} and "
            f"{BLOCK_END!r} lines do not pair up; {LEFT_AS_IT_IS}"
        )
        raise InstallError(message)
    start, end = starts[0], ends[0] + 1
    return "".join(lines[:start]), "".join(lines[start:end]), "".join(lines[end:])


def _toml_block(launch: ServerLaunch, newline: str) -> str:
    # A JSON string is also a valid TOML basic string, escapes included.
    args = ", ".join(json.dumps(arg) for arg in launch.args)
    lines = (
        BLOCK_START,
        f"[mcp_servers.{SERVER_NAME}]",
        f"command = {json.dumps(launch.command)}",
        f"args = [{args}]",
        BLOCK_END,
    )
    return "".join(line + newline for line in lines)


def _with_server(
    config: Mapping[str, object], launch: ServerLaunch
) -> dict[str, object] | None:
    servers = config.get("mcp_servers", {})
    if not isinstance(servers, dict):
        return None
    known = cast("dict[str, object]", servers)
    return {**config, "mcp_servers": known | {SERVER_NAME: launch.entry()}}


def _with_block(
    current: str, parts: tuple[str, str | None, str], launch: ServerLaunch
) -> str:
    """Return ``current`` with SYNTHIA's block in place of the old one, or appended."""
    head, block, tail = parts
    newline = "\r\n" if "\r\n" in current else "\n"
    new_block = _toml_block(launch, newline)
    if block is not None:
        return head + new_block + tail
    if current and not current.endswith("\n"):
        return current + newline + new_block
    return current + new_block


def _without_server(config: Mapping[str, object]) -> dict[str, object]:
    """Return ``config`` without the synthia server, and without an emptied table."""
    servers = config.get("mcp_servers")
    if not isinstance(servers, dict):
        return dict(config)
    kept = {
        name: entry
        for name, entry in cast("dict[str, object]", servers).items()
        if name != SERVER_NAME
    }
    rest = {key: value for key, value in config.items() if key != "mcp_servers"}
    return rest | {"mcp_servers": kept} if kept else rest


def _codex_change(path: Path, text: str | None, after: str) -> FileChange:
    # No context lines: neighbouring servers' env values stay off the screen.
    preview = _diff(path, text, text or "", after, context=0)
    return FileChange(path, text, after, preview)


def _plan_codex(path: Path, launch: ServerLaunch, *, uninstall: bool) -> Change:
    text = _read_capped(path)
    current = text or ""
    head, block, tail = _split_block(path, current)
    others = _parse_toml(head + tail)
    parsed = _parse_toml(current)
    if parsed is None or others is None:
        message = f"{path} is not valid TOML; {LEFT_AS_IT_IS}"
        raise InstallError(message)
    if uninstall:
        if block is None:
            return NoChange(f"{path} has no synthia block")
        if _without_server(others) != _without_server(parsed):
            message = (
                f"removing the synthia block from {path} would change more than "
                f"the {SERVER_NAME} server; {LEFT_AS_IT_IS}"
            )
            raise InstallError(message)
        return _codex_change(path, text, head + tail)
    servers = others.get("mcp_servers")
    if isinstance(servers, dict) and SERVER_NAME in servers:
        message = (
            f"{path} already has an [mcp_servers.{SERVER_NAME}] table that "
            f"synthia install did not add; {LEFT_AS_IT_IS}"
        )
        raise InstallError(message)
    after = _with_block(current, (head, block, tail), launch)
    written = _parse_toml(after)
    if written is None or written != _with_server(others, launch):
        message = (
            f"the {SERVER_NAME} entry would not read back from {path} as written "
            f"(is mcp_servers an inline table?); {LEFT_AS_IT_IS}"
        )
        raise InstallError(message)
    if after == current:
        return NoChange(f"{path} already starts SYNTHIA this way")
    return _codex_change(path, text, after)


def _same_launch(entry: object, launch: ServerLaunch) -> bool:
    if not isinstance(entry, dict):
        return False
    fields = cast("dict[str, object]", entry)
    return fields.get("command") == launch.command and fields.get("args") == list(
        launch.args
    )


def _plan_claude_code(
    paths: HostPaths, launch: ServerLaunch, *, uninstall: bool
) -> Change:
    path = paths.claude_json
    entry = _json_servers(path, _json_config(path, _read_capped(path))).get(SERVER_NAME)
    remove = ("mcp", "remove", "--scope", "user", SERVER_NAME)
    add = (
        "mcp",
        "add",
        "--scope",
        "user",
        SERVER_NAME,
        "--",
        launch.command,
        *launch.args,
    )
    if uninstall:
        if entry is None:
            return NoChange(f"Claude Code has no user-scope {SERVER_NAME} server")
        steps = (remove,)
    elif _same_launch(entry, launch):
        return NoChange("Claude Code already starts SYNTHIA this way")
    else:
        steps = (add,) if entry is None else (remove, add)
    exe = paths.claude_exe
    if exe is None or exe.casefold().endswith(BATCH_SUFFIXES):
        quote = subprocess.list2cmdline if paths.windows else shlex.join
        commands = "\n".join(quote(("claude", *step)) for step in steps)
        why = (
            "claude is not on PATH"
            if exe is None
            else f"{exe} is a batch file, which cmd.exe would re-parse"
        )
        message = f"{why}, so nothing changed; run this yourself:\n{commands}"
        raise InstallError(message)
    return CommandChange(tuple((exe, *step) for step in steps), path)


def shown(change: Change) -> str:
    """Return what ``change`` does, as the person is shown it before saying yes."""
    match change:
        case FileChange(preview=preview):
            return preview
        case CommandChange(commands=commands):
            return "will run:\n" + "\n".join(shlex.join(argv) for argv in commands)
        case NoChange(
            reason=reason
        ):  # pragma: no branch - pyright checks every case is matched
            return reason


def _back_up(path: Path, text: str, now: datetime) -> None:
    """Keep ``text`` next to ``path``, stamped with ``now``, never over another file.

    Raises:
        InstallError: If every name tried is taken.
        OSError: If the backup cannot be written.
    """
    stamp = f"{now:{BACKUP_STAMP}}"
    for attempt in range(1, MAX_BACKUP_TRIES + 1):
        suffix = stamp if attempt == 1 else f"{stamp}-{attempt}"
        backup = path.with_name(f"{path.name}.{suffix}.synthia.bak")
        try:
            # Creating the name first claims it, so the replace below lands
            # on our own empty file and never on someone else's.
            backup.open("xb").close()
        except FileExistsError:
            continue
        write_text_atomic(backup, text)
        return
    message = f"{MAX_BACKUP_TRIES} backup names for {path} are taken; nothing changed"
    raise InstallError(message)


def apply(
    change: Change, *, run: Callable[[Sequence[str]], None], now: datetime
) -> None:
    """Make ``change``: back up the old file, then write or run.

    ``run`` executes one argument list with no shell; ``now`` is a UTC time.

    Raises:
        InstallError: If the file changed since it was planned, or ``run`` fails.
        OSError: If the backup or the new file cannot be written.
    """
    match change:
        case FileChange(path=path, before=before, after=after):
            if _read_capped(path) != before:
                message = f"{path} changed since it was read; run the install again"
                raise InstallError(message)
            if before is not None:
                _back_up(path, before, now)
            write_text_atomic(path, after)
        case CommandChange(commands=commands, backup_of=backup_of):
            before = _read_capped(backup_of)
            if before is not None:
                _back_up(backup_of, before, now)
            for argv in commands:
                run(argv)
        case NoChange():  # pragma: no branch - pyright checks every case is matched
            pass


def run_command(argv: Sequence[str]) -> None:
    """Run ``argv`` with no shell and no ``SYNTHIA_*`` variables, within a timeout.

    Raises:
        InstallError: If it cannot start, does not finish in time, or fails.
    """
    command = shlex.join(argv)
    try:
        subprocess.run(  # noqa: S603 - a resolved claude and SYNTHIA's own launch
            argv,
            check=True,
            timeout=COMMAND_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=environment_without_own({}),
        )
    except subprocess.TimeoutExpired:
        message = f"{command} did not finish within {COMMAND_TIMEOUT_S} s"
        raise InstallError(message) from None
    except subprocess.CalledProcessError as error:
        said = str(error.stderr).strip()[-MAX_SHOWN_ERROR:]
        message = f"{command} failed with exit code {error.returncode}: {said}"
        raise InstallError(message) from None
    except OSError as error:
        message = f"cannot run {command}: {error.strerror}"
        raise InstallError(message) from None


def next_steps(target: HostTarget, *, uninstall: bool) -> list[str]:
    """Return what the person does next; installing alone shares nothing."""
    product = target.product
    if uninstall:
        return [
            f"restart {product}, so it stops starting SYNTHIA",
            (
                f"[hosts.{target}] in SYNTHIA_HOME/{HOSTS_FILE} and any private grant "
                f"stay until you remove them (synthia consent revoke --host {target})"
            ),
        ]
    return [
        f"{product} can start SYNTHIA now, and sees nothing until you say what:",
        (
            f"1. add to SYNTHIA_HOME/{HOSTS_FILE} (collections: names from "
            "synthia knowledge list):"
        ),
        f"   [hosts.{target}]",
        "   collections = []",
        "   memory = false",
        (
            "2. only if it may also receive private passages: "
            f"synthia consent grant --host {target} --private"
        ),
        f"3. restart {product}",
    ]
