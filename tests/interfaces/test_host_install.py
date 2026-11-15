import json
import sys
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from synthia.interfaces import cli, host_install
from synthia.interfaces.cli import app
from synthia.interfaces.host_install import (
    MAX_HOST_CONFIG_BYTES,
    Change,
    CommandChange,
    FileChange,
    HostPaths,
    HostTarget,
    InstallError,
    NoChange,
    ServerLaunch,
    apply,
    next_steps,
    plan,
    run_command,
    shown,
)

NOW = datetime(2026, 11, 15, 8, 30, 0, tzinfo=UTC)
LATER = datetime(2026, 11, 15, 9, 0, 0, tzinfo=UTC)
WINDOWS_PYTHON = 'C:\\Program Files\\Py "3"\\python.exe'
FAKE_CLAUDE = "C:\\tools\\claude.exe"
FOREIGN = {"command": "npx", "args": ["-y", "other-server"]}
runner = CliRunner()


class Runs:
    def __init__(self) -> None:
        self.argvs: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str]) -> None:
        self.argvs.append(tuple(argv))


def paths_in(
    root: Path, *, claude_exe: str | None = FAKE_CLAUDE, windows: bool = False
) -> HostPaths:
    return HostPaths(
        desktop_config=root / "Claude" / "claude_desktop_config.json",
        codex_config=root / ".codex" / "config.toml",
        claude_json=root / ".claude.json",
        claude_exe=claude_exe,
        windows=windows,
    )


def desktop_plan(paths: HostPaths) -> Change:
    return plan(
        HostTarget.CLAUDE_DESKTOP,
        launch_for(HostTarget.CLAUDE_DESKTOP),
        paths,
        uninstall=False,
    )


def launch_for(target: HostTarget, executable: str = WINDOWS_PYTHON) -> ServerLaunch:
    return ServerLaunch.of(target, executable=executable)


def install(
    target: HostTarget,
    paths: HostPaths,
    *,
    uninstall: bool = False,
    executable: str = WINDOWS_PYTHON,
    now: datetime = NOW,
) -> Change:
    change = plan(target, launch_for(target, executable), paths, uninstall=uninstall)
    apply(change, run=Runs(), now=now)
    return change


def backups(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.synthia.bak"))


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_launch_runs_this_python_with_the_host_name() -> None:
    launch = launch_for(HostTarget.CODEX)

    assert launch.command == WINDOWS_PYTHON
    assert launch.args == ("-m", "synthia", "mcp", "serve", "--host", "codex")


def test_launch_without_a_known_python_is_refused() -> None:
    with pytest.raises(InstallError, match="which Python"):
        ServerLaunch.of(HostTarget.CODEX, executable="")


@pytest.mark.parametrize(
    ("platform", "environ", "desktop"),
    [
        ("win32", {"APPDATA": "R"}, Path("R/Claude")),
        ("win32", {}, Path("H/AppData/Roaming/Claude")),
        ("darwin", {}, Path("H/Library/Application Support/Claude")),
        ("linux", {}, Path("H/.config/Claude")),
    ],
)
def test_paths_follow_each_platform(
    platform: str, environ: dict[str, str], desktop: Path
) -> None:
    found = HostPaths.find(
        environ=environ, home=Path("H"), platform=platform, which=lambda _: None
    )

    assert found.desktop_config == desktop / "claude_desktop_config.json"
    assert found.codex_config == Path("H/.codex/config.toml")
    assert found.claude_json == Path("H/.claude.json")
    assert found.claude_exe is None
    assert found.windows is (platform == "win32")


def test_codex_home_moves_the_codex_config() -> None:
    found = HostPaths.find(
        environ={"CODEX_HOME": "C"},
        home=Path("H"),
        platform="linux",
        which={"claude": FAKE_CLAUDE}.get,
    )

    assert found.codex_config == Path("C/config.toml")
    assert found.claude_exe == FAKE_CLAUDE


def test_desktop_install_keeps_every_other_server_and_key(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    old = json.dumps({"theme": "dark", "mcpServers": {"other": FOREIGN}}).encode()
    write(paths.desktop_config, old)

    install(HostTarget.CLAUDE_DESKTOP, paths)

    written = json.loads(paths.desktop_config.read_text(encoding="utf-8"))
    assert written == {
        "theme": "dark",
        "mcpServers": {
            "other": FOREIGN,
            "synthia": launch_for(HostTarget.CLAUDE_DESKTOP).entry(),
        },
    }
    [backup] = backups(paths.desktop_config.parent)
    assert backup.name == "claude_desktop_config.json.20261115T083000Z.synthia.bak"
    assert backup.read_bytes() == old


def test_desktop_install_twice_writes_and_backs_up_once(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.desktop_config, b"{}")
    install(HostTarget.CLAUDE_DESKTOP, paths)
    first = paths.desktop_config.read_bytes()

    again = install(HostTarget.CLAUDE_DESKTOP, paths, now=LATER)

    assert isinstance(again, NoChange)
    assert paths.desktop_config.read_bytes() == first
    assert len(backups(paths.desktop_config.parent)) == 1


def test_desktop_install_without_a_config_makes_one_and_no_backup(
    tmp_path: Path,
) -> None:
    paths = paths_in(tmp_path)

    change = install(HostTarget.CLAUDE_DESKTOP, paths)

    assert isinstance(change, FileChange)
    assert change.before is None
    assert json.loads(paths.desktop_config.read_text(encoding="utf-8")) == {
        "mcpServers": {"synthia": launch_for(HostTarget.CLAUDE_DESKTOP).entry()}
    }
    assert backups(paths.desktop_config.parent) == []


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"{", "not valid JSON"),
        (b'{"a": NaN}', "not valid JSON"),
        (b"[" * 100_000, "not valid JSON"),
        (b"[]", "does not hold a JSON object"),
        (b'{"mcpServers": []}', "mcpServers in .* is not a JSON object"),
        (b'{"a": "\xff"}', "not UTF-8"),
    ],
    ids=["cut", "nan", "deep", "array", "servers", "bytes"],
)
def test_a_damaged_desktop_config_is_refused_and_left_as_it_is(
    tmp_path: Path, data: bytes, reason: str
) -> None:
    paths = paths_in(tmp_path)
    write(paths.desktop_config, data)

    with pytest.raises(InstallError, match=reason):
        plan(
            HostTarget.CLAUDE_DESKTOP,
            launch_for(HostTarget.CLAUDE_DESKTOP),
            paths,
            uninstall=False,
        )

    assert paths.desktop_config.read_bytes() == data


def test_a_config_too_large_to_read_is_refused(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.desktop_config, b" " * (MAX_HOST_CONFIG_BYTES + 1))

    with pytest.raises(InstallError, match="larger than"):
        plan(
            HostTarget.CLAUDE_DESKTOP,
            launch_for(HostTarget.CLAUDE_DESKTOP),
            paths,
            uninstall=False,
        )


def test_a_config_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    paths.desktop_config.mkdir(parents=True)

    with pytest.raises(InstallError, match="cannot read"):
        plan(
            HostTarget.CLAUDE_DESKTOP,
            launch_for(HostTarget.CLAUDE_DESKTOP),
            paths,
            uninstall=False,
        )


def test_desktop_uninstall_keeps_the_other_servers(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.desktop_config, json.dumps({"mcpServers": {"other": FOREIGN}}).encode())
    install(HostTarget.CLAUDE_DESKTOP, paths)

    install(HostTarget.CLAUDE_DESKTOP, paths, uninstall=True, now=LATER)

    assert json.loads(paths.desktop_config.read_text(encoding="utf-8")) == {
        "mcpServers": {"other": FOREIGN}
    }
    assert len(backups(paths.desktop_config.parent)) == 2


def test_desktop_uninstall_with_nothing_installed_changes_nothing(
    tmp_path: Path,
) -> None:
    paths = paths_in(tmp_path)

    change = install(HostTarget.CLAUDE_DESKTOP, paths, uninstall=True)

    assert isinstance(change, NoChange)
    assert not paths.desktop_config.exists()


START = host_install.BLOCK_START + "\n"
END = host_install.BLOCK_END + "\n"
CODEX_OTHER = '[mcp_servers.other]\ncommand = "npx"\nargs = ["-y", "other-server"]\n'


def codex(paths: HostPaths) -> dict[str, object]:
    return tomllib.loads(paths.codex_config.read_text(encoding="utf-8"))


def test_codex_install_keeps_the_other_server_and_reads_back(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.codex_config, CODEX_OTHER.encode())

    install(HostTarget.CODEX, paths)

    assert codex(paths) == {
        "mcp_servers": {
            "other": {"command": "npx", "args": ["-y", "other-server"]},
            "synthia": launch_for(HostTarget.CODEX).entry(),
        }
    }
    [backup] = backups(paths.codex_config.parent)
    assert backup.read_bytes() == CODEX_OTHER.encode()


def test_codex_install_twice_leaves_one_block_and_one_backup(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.codex_config, CODEX_OTHER.encode())
    install(HostTarget.CODEX, paths)
    first = paths.codex_config.read_bytes()

    again = install(HostTarget.CODEX, paths, now=LATER)

    assert isinstance(again, NoChange)
    assert paths.codex_config.read_bytes() == first
    assert first.count(host_install.BLOCK_START.encode()) == 1
    assert len(backups(paths.codex_config.parent)) == 1


def test_codex_install_with_another_python_replaces_the_block(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.codex_config, CODEX_OTHER.encode())
    install(HostTarget.CODEX, paths)

    install(HostTarget.CODEX, paths, executable="/usr/bin/python3", now=LATER)

    text = paths.codex_config.read_text(encoding="utf-8")
    assert text.count(host_install.BLOCK_START) == 1
    assert codex(paths)["mcp_servers"] == {
        "other": {"command": "npx", "args": ["-y", "other-server"]},
        "synthia": launch_for(HostTarget.CODEX, "/usr/bin/python3").entry(),
    }


@pytest.mark.parametrize(
    "original",
    [
        CODEX_OTHER,
        CODEX_OTHER.replace("\n", "\r\n"),
        'model = "o3"',
        "",
        "[mcp_servers]\n",
    ],
)
def test_codex_uninstall_gives_back_the_file_as_it_was(
    tmp_path: Path, original: str
) -> None:
    paths = paths_in(tmp_path)
    write(paths.codex_config, original.encode())

    install(HostTarget.CODEX, paths)
    install(HostTarget.CODEX, paths, uninstall=True, now=LATER)

    expected = original if original.endswith("\n") or not original else original + "\n"
    assert paths.codex_config.read_bytes() == expected.encode()


def test_codex_block_follows_a_crlf_file(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.codex_config, CODEX_OTHER.replace("\n", "\r\n").encode())

    install(HostTarget.CODEX, paths)

    data = paths.codex_config.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n")


def test_codex_install_without_a_config_makes_one(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)

    install(HostTarget.CODEX, paths)

    assert codex(paths) == {
        "mcp_servers": {"synthia": launch_for(HostTarget.CODEX).entry()}
    }


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ('[mcp_servers.other\ncommand = "x"\n', "not valid TOML"),
        ('[mcp_servers.synthia]\ncommand = "mine"\n', "did not add"),
        (START + START + END, "damaged synthia block"),
        (END + START, "damaged synthia block"),
        (START, "damaged synthia block"),
        (START + "[mcp_servers.synthia\n" + END, "not valid TOML"),
        (START + END + '[mcp_servers.synthia.env]\nA = "1"\n', "did not add"),
        ('mcp_servers = { other = { command = "x" } }\n', "would not read back"),
        ("mcp_servers = 3\n", "would not read back"),
        ('[[mcp_servers]]\nname = "x"\n', "would not read back"),
    ],
)
def test_a_codex_config_synthia_cannot_own_is_refused_and_left_as_it_is(
    tmp_path: Path, text: str, reason: str
) -> None:
    paths = paths_in(tmp_path)
    write(paths.codex_config, text.encode())

    with pytest.raises(InstallError, match=reason):
        plan(HostTarget.CODEX, launch_for(HostTarget.CODEX), paths, uninstall=False)

    assert paths.codex_config.read_bytes() == text.encode()


def test_codex_uninstall_without_a_block_changes_nothing(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.codex_config, CODEX_OTHER.encode())

    change = install(HostTarget.CODEX, paths, uninstall=True)

    assert isinstance(change, NoChange)
    assert backups(paths.codex_config.parent) == []


def claude_json(paths: HostPaths, servers: dict[str, object]) -> bytes:
    data = json.dumps({"numStartups": 3, "mcpServers": servers}).encode()
    write(paths.claude_json, data)
    return data


def code_plan(paths: HostPaths, *, uninstall: bool = False) -> Change:
    return plan(
        HostTarget.CLAUDE_CODE,
        launch_for(HostTarget.CLAUDE_CODE),
        paths,
        uninstall=uninstall,
    )


ADD = (
    FAKE_CLAUDE,
    "mcp",
    "add",
    "--scope",
    "user",
    "synthia",
    "--",
    WINDOWS_PYTHON,
    *launch_for(HostTarget.CLAUDE_CODE).args,
)
REMOVE = (FAKE_CLAUDE, "mcp", "remove", "--scope", "user", "synthia")


def test_claude_code_install_backs_up_then_runs_claude_mcp_add(
    tmp_path: Path,
) -> None:
    paths = paths_in(tmp_path)
    old = claude_json(paths, {"other": FOREIGN})
    runs = Runs()

    apply(code_plan(paths), run=runs, now=NOW)

    assert runs.argvs == [ADD]
    [backup] = backups(tmp_path)
    assert backup.name == ".claude.json.20261115T083000Z.synthia.bak"
    assert backup.read_bytes() == old
    assert paths.claude_json.read_bytes() == old


def test_claude_code_install_with_no_claude_json_backs_up_nothing(
    tmp_path: Path,
) -> None:
    paths = paths_in(tmp_path)
    runs = Runs()

    apply(code_plan(paths), run=runs, now=NOW)

    assert runs.argvs == [ADD]
    assert backups(tmp_path) == []


def test_claude_code_already_installed_changes_nothing(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    stdio: dict[str, object] = {"type": "stdio", "env": {}}
    entry = stdio | launch_for(HostTarget.CLAUDE_CODE).entry()
    claude_json(paths, {"synthia": entry})

    assert isinstance(code_plan(paths), NoChange)


@pytest.mark.parametrize("entry", [{"command": "old", "args": []}, "not an object"])
def test_claude_code_with_another_synthia_removes_then_adds(
    tmp_path: Path, entry: object
) -> None:
    paths = paths_in(tmp_path)
    claude_json(paths, {"synthia": entry})

    change = code_plan(paths)

    assert change == CommandChange((REMOVE, ADD), paths.claude_json)


def test_claude_code_uninstall_runs_claude_mcp_remove(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    claude_json(paths, {"synthia": launch_for(HostTarget.CLAUDE_CODE).entry()})

    assert code_plan(paths, uninstall=True) == CommandChange(
        (REMOVE,), paths.claude_json
    )


def test_claude_code_uninstall_with_nothing_installed_changes_nothing(
    tmp_path: Path,
) -> None:
    paths = paths_in(tmp_path)
    claude_json(paths, {"other": FOREIGN})

    assert isinstance(code_plan(paths, uninstall=True), NoChange)


def test_claude_code_without_claude_is_refused_with_the_command(
    tmp_path: Path,
) -> None:
    paths = paths_in(tmp_path, claude_exe=None)

    with pytest.raises(InstallError, match="not on PATH") as caught:
        code_plan(paths)

    assert "claude mcp add --scope user synthia --" in str(caught.value)
    assert not paths.claude_json.exists()


def test_claude_code_with_a_damaged_claude_json_is_refused(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.claude_json, b"{")

    with pytest.raises(InstallError, match="not valid JSON"):
        code_plan(paths)


def test_a_file_changed_after_it_was_shown_is_not_written(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.desktop_config, b"{}")
    change = plan(
        HostTarget.CLAUDE_DESKTOP,
        launch_for(HostTarget.CLAUDE_DESKTOP),
        paths,
        uninstall=False,
    )
    write(paths.desktop_config, b'{"new": 1}')

    with pytest.raises(InstallError, match="changed since"):
        apply(change, run=Runs(), now=NOW)

    assert paths.desktop_config.read_bytes() == b'{"new": 1}'
    assert backups(paths.desktop_config.parent) == []


def test_shown_is_a_diff_a_command_or_the_reason(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    created = shown(desktop_plan(paths))
    write(paths.desktop_config, b'{"theme": "dark"}')
    edited = shown(desktop_plan(paths))

    assert created.startswith("--- /dev/null\n")
    assert edited.startswith(f"--- {paths.desktop_config.resolve()}\n")
    assert '-  "theme": "dark"\n+  "theme": "dark",\n' in edited
    assert shown(CommandChange((("claude", "mcp", "x y"),), tmp_path)) == (
        "will run:\nclaude mcp 'x y'"
    )
    assert shown(NoChange("already")) == "already"


def test_run_command_runs_without_synthia_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTHIA_CANARY", "1")

    run_command(
        [
            sys.executable,
            "-c",
            "import os, sys; sys.exit('SYNTHIA_CANARY' in os.environ)",
        ]
    )


def test_run_command_reports_a_failure_with_its_output() -> None:
    with pytest.raises(InstallError, match="exit code 3: boom"):
        run_command(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
        )


def test_run_command_stops_a_command_that_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(host_install, "COMMAND_TIMEOUT_S", 0.5)

    with pytest.raises(InstallError, match="did not finish"):
        run_command([sys.executable, "-c", "import time; time.sleep(30)"])


def test_run_command_reports_a_program_that_cannot_start(tmp_path: Path) -> None:
    with pytest.raises(InstallError, match="cannot run"):
        run_command([str(tmp_path / "no-such-program")])


def test_next_steps_grant_nothing_and_name_the_host() -> None:
    steps = "\n".join(next_steps(HostTarget.CODEX, uninstall=False))
    removed = "\n".join(next_steps(HostTarget.CODEX, uninstall=True))

    assert "[hosts.codex]" in steps
    assert "memory = false" in steps
    assert "synthia consent grant --host codex --private" in steps
    assert "restart Codex" in steps
    assert "restart Codex" in removed
    assert "synthia consent revoke --host codex" in removed


@pytest.fixture
def cli_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HostPaths:
    paths = paths_in(tmp_path)
    monkeypatch.setattr(cli, "host_paths", lambda: paths)
    return paths


def test_cli_install_asks_and_a_no_changes_nothing(cli_paths: HostPaths) -> None:
    result = runner.invoke(app, ["install", "--for", "codex"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "+[mcp_servers.synthia]" in result.output
    assert "nothing changed" in result.output
    assert not cli_paths.codex_config.exists()


def test_cli_install_with_yes_writes_and_prints_the_next_step(
    cli_paths: HostPaths,
) -> None:
    result = runner.invoke(app, ["install", "--for", "codex", "--yes"])
    again = runner.invoke(app, ["install", "--for", "codex", "--yes"])

    assert result.exit_code == 0, result.output
    assert "[hosts.codex]" in result.output
    assert codex(cli_paths)["mcp_servers"] == {
        "synthia": launch_for(HostTarget.CODEX, sys.executable).entry()
    }
    assert again.exit_code == 0, again.output
    assert "already starts SYNTHIA" in again.output


def test_cli_refuses_a_damaged_config_with_exit_2(cli_paths: HostPaths) -> None:
    write(cli_paths.desktop_config, b"{")

    result = runner.invoke(app, ["install", "--for", "claude-desktop", "--yes"])

    assert result.exit_code == 2
    assert "not valid JSON" in result.output
    assert cli_paths.desktop_config.read_bytes() == b"{"


@pytest.mark.usefixtures("cli_paths")
def test_cli_reports_a_failed_command_with_exit_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(argv: Sequence[str]) -> None:
        message = f"{argv[0]} failed"
        raise InstallError(message)

    monkeypatch.setattr(host_install, "run_command", fail)

    result = runner.invoke(app, ["install", "--for", "claude-code", "--yes"])

    assert result.exit_code == 2
    assert f"{FAKE_CLAUDE} failed" in result.output


@pytest.mark.parametrize("exe", ["C:\\npm\\claude.CMD", "C:\\npm\\claude.bat"])
def test_a_claude_batch_file_is_never_run_and_the_command_is_printed(
    tmp_path: Path, exe: str
) -> None:
    paths = paths_in(tmp_path, claude_exe=exe, windows=True)
    claude_json(paths, {"other": FOREIGN})

    with pytest.raises(InstallError, match="batch file") as caught:
        code_plan(paths)

    assert '"C:\\Program Files\\Py \\"3\\"\\python.exe"' in str(caught.value)
    assert backups(tmp_path) == []


def test_the_printed_command_is_quoted_for_this_system(tmp_path: Path) -> None:
    with pytest.raises(InstallError) as windows:
        code_plan(paths_in(tmp_path, claude_exe=None, windows=True))
    with pytest.raises(InstallError) as posix:
        code_plan(paths_in(tmp_path, claude_exe=None))

    assert '"C:\\Program Files\\Py \\"3\\"\\python.exe"' in str(windows.value)
    assert "'C:\\Program Files\\Py \"3\"\\python.exe'" in str(posix.value)


@pytest.mark.parametrize(
    "data",
    [
        b'{"mcpServers": {"other": {"command": "x", "n": 1e400}}}',
        b'{"mcpServers": {"other": {"env": {"K": "1", "K": "2"}}}}',
        b'{"a": 1, "a": 1}',
    ],
    ids=["infinite", "nested-repeat", "repeat"],
)
def test_json_that_cannot_be_written_back_as_read_is_refused(
    tmp_path: Path, data: bytes
) -> None:
    paths = paths_in(tmp_path)
    write(paths.desktop_config, data)

    with pytest.raises(InstallError, match=r"too large|not valid JSON"):
        desktop_plan(paths)

    assert paths.desktop_config.read_bytes() == data


CANARY = "canary-value-9f2c"


def test_desktop_diff_hides_other_servers_env_and_headers(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    other = FOREIGN | {"env": {"TOKEN": CANARY}, "headers": [{"auth": CANARY}]}
    write(paths.desktop_config, json.dumps({"mcpServers": {"other": other}}).encode())

    change = desktop_plan(paths)
    apply(change, run=Runs(), now=NOW)

    assert CANARY not in shown(change)
    assert '+    "synthia": {' in shown(change)
    assert '"auth": "***"' in shown(change)
    written = json.loads(paths.desktop_config.read_text(encoding="utf-8"))
    assert written["mcpServers"]["other"] == other


def test_codex_diff_shows_only_the_synthia_lines(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(
        paths.codex_config,
        f'{CODEX_OTHER}[mcp_servers.other.env]\nTOKEN = "{CANARY}"\n'.encode(),
    )

    change = plan(
        HostTarget.CODEX, launch_for(HostTarget.CODEX), paths, uninstall=False
    )

    assert CANARY not in shown(change)
    assert "+[mcp_servers.synthia]\n" in shown(change)


def test_a_backup_never_replaces_a_file_already_there(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.desktop_config, b"{}")
    taken = paths.desktop_config.with_name(
        "claude_desktop_config.json.20261115T083000Z.synthia.bak"
    )
    taken.write_bytes(b"mine")

    install(HostTarget.CLAUDE_DESKTOP, paths)

    assert taken.read_bytes() == b"mine"
    second = taken.with_name(
        "claude_desktop_config.json.20261115T083000Z-2.synthia.bak"
    )
    assert second.read_bytes() == b"{}"


def test_with_every_backup_name_taken_nothing_is_written(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    write(paths.desktop_config, b"{}")
    stamp = "claude_desktop_config.json.20261115T083000Z"
    for suffix in ["", *(f"-{n}" for n in range(2, host_install.MAX_BACKUP_TRIES + 1))]:
        paths.desktop_config.with_name(f"{stamp}{suffix}.synthia.bak").write_bytes(b"")

    with pytest.raises(InstallError, match="backup names"):
        install(HostTarget.CLAUDE_DESKTOP, paths)

    assert paths.desktop_config.read_bytes() == b"{}"


def test_codex_uninstall_that_would_drop_more_than_synthia_is_refused(
    tmp_path: Path,
) -> None:
    paths = paths_in(tmp_path)
    text = START + '[mcp_servers.synthia]\ncommand = "x"\n[kept]\nk = 1\n' + END
    write(paths.codex_config, text.encode())

    with pytest.raises(InstallError, match="more than"):
        plan(HostTarget.CODEX, launch_for(HostTarget.CODEX), paths, uninstall=True)

    assert paths.codex_config.read_bytes() == text.encode()
