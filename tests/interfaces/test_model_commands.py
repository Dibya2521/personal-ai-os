import io
import platform
from pathlib import Path

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from synthia.interfaces import model_commands
from synthia.interfaces.cli import app
from synthia.interfaces.model_commands import (
    Machine,
    configured,
    install,
    list_table,
    remove,
    resolve,
    suggested,
    this_machine,
)
from synthia.kernel.config import DEFAULT_LOCAL_MODEL
from synthia.models.catalogue import (
    MODELS,
    Backend,
    Target,
    find,
    target_of,
)
from synthia.models.download import part_path
from synthia.models.install import (
    BYTES_PER_GB,
    COMPLETE_MARKER,
    Installer,
    size_text,
)
from tests.models.fakes import CPU, MODEL, Files

runner = CliRunner()
WINDOWS = Machine(Target.WINDOWS_X64, nvidia=False)
WINDOWS_CPU = "llama.cpp-b11130-windows-x64-cpu"


def console() -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    return Console(file=out, width=120, color_system=None), out


def never(question: str) -> bool:
    raise AssertionError(question)


def test_this_machine_reads_the_platform() -> None:
    assert this_machine().target == target_of(platform.system(), platform.machine())


def test_suggested_is_the_builds_for_here_and_the_configured_model() -> None:
    picks = suggested(WINDOWS, MODEL)

    assert [getattr(p, "backend", None) for p in picks] == [
        Backend.VULKAN,
        Backend.CPU,
        None,
    ]
    assert picks[-1] is MODEL
    assert suggested(Machine(None, nvidia=True), MODEL) == ()


def test_the_configured_model_must_be_a_catalogue_model() -> None:
    assert configured(DEFAULT_LOCAL_MODEL) is find(DEFAULT_LOCAL_MODEL)
    with pytest.raises(KeyError, match="gpt-5"):
        configured("gpt-5")
    with pytest.raises(KeyError):
        configured(WINDOWS_CPU)


def test_resolve_keeps_order_drops_repeats_and_names_every_unknown() -> None:
    assert resolve([DEFAULT_LOCAL_MODEL, DEFAULT_LOCAL_MODEL]) == (
        find(DEFAULT_LOCAL_MODEL),
    )
    with pytest.raises(KeyError, match="gpt-5, llama-9"):
        resolve(["gpt-5", DEFAULT_LOCAL_MODEL, "llama-9"])


def test_the_list_shows_each_state_and_marks_the_picks(tmp_path: Path) -> None:
    models = Installer(tmp_path, BYTES_PER_GB)
    cpu = models.path_of(find(WINDOWS_CPU))
    cpu.mkdir(parents=True)
    (cpu / COMPLETE_MARKER).write_text("b11130\n", encoding="utf-8")
    model = MODELS[0]
    assert model.id == DEFAULT_LOCAL_MODEL
    models.path_of(model).mkdir(parents=True)
    part_path(model.files[0], models.path_of(model)).write_bytes(b"x" * 10)
    out_console, out = console()

    out_console.print(list_table(models, WINDOWS, DEFAULT_LOCAL_MODEL))

    # A row reads: [mark] name size unit state.
    rows = {line.split()[-4]: line.split() for line in out.getvalue().splitlines()[1:]}
    assert rows[WINDOWS_CPU] == ["*", WINDOWS_CPU, "17.7", "MB", "installed"]
    assert rows[DEFAULT_LOCAL_MODEL][0] == "*"
    assert rows[DEFAULT_LOCAL_MODEL][-1] == "partial"
    assert rows["llama.cpp-b11130-linux-x64-cpu"][0] != "*"
    assert rows["llama.cpp-b11130-linux-x64-cpu"][-1] == "-"


async def test_install_shows_the_plan_asks_and_installs(tmp_path: Path) -> None:
    files = Files()
    models = Installer(tmp_path, BYTES_PER_GB)
    out_console, out = console()
    asked: list[str] = []

    def yes(question: str) -> bool:
        asked.append(question)
        return True

    async with httpx.AsyncClient(transport=httpx.MockTransport(files)) as client:
        await install(models, client, [CPU, MODEL], out_console, yes)

    assert models.installed(CPU)
    assert models.installed(MODEL)
    assert asked == [f"download {size_text(CPU.size + MODEL.size)} and install?"]
    assert "tiny" in out.getvalue()


async def test_a_no_installs_nothing_and_fetches_nothing(tmp_path: Path) -> None:
    files = Files()
    out_console, out = console()

    async with httpx.AsyncClient(transport=httpx.MockTransport(files)) as client:
        await install(
            Installer(tmp_path, BYTES_PER_GB),
            client,
            [MODEL],
            out_console,
            lambda _: False,
        )

    assert files.requests == []
    assert out.getvalue().endswith("nothing installed\n")


async def test_nothing_to_do_does_not_ask(tmp_path: Path) -> None:
    files = Files()
    models = Installer(tmp_path, BYTES_PER_GB)
    out_console, out = console()
    async with httpx.AsyncClient(transport=httpx.MockTransport(files)) as client:
        await models.install(client, MODEL)
        await install(models, client, [MODEL], out_console, never)

    assert out.getvalue() == "everything asked for is installed\n"


async def test_remove_says_what_it_freed(tmp_path: Path) -> None:
    models = Installer(tmp_path, BYTES_PER_GB)
    async with httpx.AsyncClient(transport=httpx.MockTransport(Files())) as client:
        await models.install(client, MODEL)
    out_console, out = console()

    remove(models, [MODEL, CPU], out_console)

    assert out.getvalue() == (
        f"removed tiny, {size_text(MODEL.size)} freed\n{CPU.id} was not installed\n"
    )


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run from an empty directory, so no developer .env is ever read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.mark.usefixtures("home")
def test_models_list_prints_the_catalogue() -> None:
    result = runner.invoke(app, ["models", "list"])

    assert result.exit_code == 0
    assert DEFAULT_LOCAL_MODEL in result.stdout
    assert WINDOWS_CPU in result.stdout


@pytest.mark.usefixtures("home")
def test_unknown_names_exit_2_and_point_at_the_list() -> None:
    result = runner.invoke(app, ["models", "install", "gpt-5", "llama-9"])

    assert result.exit_code == 2
    assert "not in the catalogue: gpt-5, llama-9" in result.stderr
    assert "synthia models list" in result.stderr


@pytest.mark.usefixtures("home")
def test_install_asks_first_and_no_means_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_commands, "this_machine", lambda: WINDOWS)

    result = runner.invoke(app, ["models", "install"], input="n\n")

    assert result.exit_code == 0
    assert "download 3.23 GB and install? [y/N]" in result.stdout
    assert "nothing installed" in result.stdout


def test_install_over_the_budget_exits_1_before_any_request(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    monkeypatch.setenv("SYNTHIA_DISK_BUDGET_GB", "0.001")

    result = runner.invoke(app, ["models", "install", "--yes", DEFAULT_LOCAL_MODEL])

    assert result.exit_code == 1
    assert "raise SYNTHIA_DISK_BUDGET_GB" in result.stderr
    assert not home.exists()


@pytest.mark.usefixtures("home")
def test_install_with_an_unknown_local_model_exits_2_and_the_list_still_shows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTHIA_LOCAL_MODEL", "gpt-5")

    install_result = runner.invoke(app, ["models", "install"])
    list_result = runner.invoke(app, ["models", "list"])

    assert install_result.exit_code == 2
    assert "SYNTHIA_LOCAL_MODEL=gpt-5 is not a model" in install_result.stderr
    assert "synthia models list" in install_result.stderr
    assert list_result.exit_code == 0
    assert DEFAULT_LOCAL_MODEL in list_result.stdout


def test_the_list_marks_nothing_where_no_build_suits(tmp_path: Path) -> None:
    out_console, out = console()

    out_console.print(
        list_table(
            Installer(tmp_path, BYTES_PER_GB),
            Machine(None, nvidia=False),
            DEFAULT_LOCAL_MODEL,
        )
    )

    assert "*" not in out.getvalue()


@pytest.mark.usefixtures("home")
def test_install_with_no_build_for_this_machine_exits_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_commands, "this_machine", lambda: Machine(None, nvidia=False)
    )

    result = runner.invoke(app, ["models", "install"])

    assert result.exit_code == 2
    assert "SYNTHIA runs remote only" in result.stderr


@pytest.mark.usefixtures("home")
def test_remove_of_what_is_not_installed_says_so() -> None:
    result = runner.invoke(app, ["models", "remove", DEFAULT_LOCAL_MODEL])

    assert result.exit_code == 0
    assert f"{DEFAULT_LOCAL_MODEL} was not installed" in result.stdout


@pytest.mark.usefixtures("home")
def test_a_bad_configuration_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTHIA_DISK_BUDGET_GB", "-1")

    result = runner.invoke(app, ["models", "list"])

    assert result.exit_code == 2
    assert "SYNTHIA_DISK_BUDGET_GB" in result.stderr
