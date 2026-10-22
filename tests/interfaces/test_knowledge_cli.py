import json
import re
from pathlib import Path

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from synthia.interfaces import cli, doctor
from synthia.interfaces.cli import app
from synthia.kernel.config import Settings
from synthia.kernel.jsonrpc import Params
from synthia.knowledge.deep import NO_REWRITES
from synthia.server.api import NOT_INDEXING
from tests.agent.scripted import says
from tests.interfaces.daemons import private_daemon, toy_embedder
from tests.interfaces.test_chat import KEY
from tests.knowledge.corpus import NOTES, planted
from tests.server.test_confinement import Serving
from tests.server.test_daemon import Local

runner = CliRunner()
FAIL = doctor.Status.FAIL
LOG = "".join(
    [
        "kestrel \x1b[2J wiped the screen\n",
        *(f"{word}\n" for word in ("a", "the", "of", "my", "is", "in", "who", "for")),
        "which\n",
    ]
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    monkeypatch.setenv("SYNTHIA_OPENROUTER_API_KEY", KEY)
    planted(tmp_path / "notes", NOTES)
    return tmp_path


def keyed(home: Path) -> Settings:
    return Settings(home=home, openrouter_api_key=SecretStr(KEY))


def knowledge(*arguments: str) -> tuple[int, str, str]:
    result = runner.invoke(app, ["knowledge", *arguments])
    return result.exit_code, result.stdout, result.stderr


def test_a_folder_is_added_indexed_searched_and_removed(home: Path) -> None:
    root = (home / "notes").resolve()
    with private_daemon(keyed(home), embedder=toy_embedder):
        none_yet = knowledge("list")
        added = knowledge("add", "notes", "--private", "--exclude", "drafts")
        listed = knowledge("list")
        indexed = knowledge("index")
        found = knowledge("search", "kestrel", "--variant", "words")
        two = knowledge("search", "miso kestrel", "--variant", "words")
        public = knowledge("add", "notes", "--public")
        removed = knowledge("remove", "notes")
        again = knowledge("remove", "notes")
        nothing = knowledge("search", "kestrel")

    source = f"notes | files | private | {root} | leaves out drafts"
    assert none_yet == (
        0,
        "no folders added; add one with `synthia knowledge add <folder>`\n",
        "",
    )
    assert added == (
        0,
        (
            f"{source}\nrun `synthia knowledge index` to read it now; "
            "the daemon also reads it after 10 minutes without a turn\n"
        ),
        "",
    )
    assert listed == (0, f"{source}\n", "")
    assert indexed[0] == 0
    assert re.fullmatch(
        r"files: 3 added, 0 changed, 0 unchanged, 0 removed, 0 unreadable, "
        r"0 held back, 0 passed over \| chunks: 3 written, 0 cut, 3 embedded "
        r"\| \d+\.\d s\n",
        indexed[1],
    )
    assert found[0] == 0
    assert re.fullmatch(
        r"notes/server\.txt:1 \| score \d+\.\d{3} \(private\)\n"
        r"    the database server host kestrel\n",
        found[1],
    )
    assert sorted(shown.split(" | ")[0] for shown in two[1].split("\n\n")) == [
        "notes/cat.txt:1",
        "notes/server.txt:1",
    ]
    assert public[1].startswith(f"notes | files | not private | {root} | leaves out")
    assert removed == (0, f"removed {root}\n", "")
    assert again == (0, f"nothing to remove: {root} was not added\n", "")
    assert nothing == (0, "nothing found\n", "")


def test_a_long_passage_shows_its_first_lines_with_control_characters_escaped(
    home: Path,
) -> None:
    planted(home / "logs", {"log.txt": LOG, "other.txt": "my cat\n"})
    with private_daemon(keyed(home), embedder=toy_embedder):
        knowledge("add", "logs")
        knowledge("index")
        code, shown, _ = knowledge("search", "kestrel", "--variant", "words")

    lines = shown.splitlines()
    assert code == 0
    assert re.fullmatch(r"logs/log\.txt:1-10 \| score \d+\.\d{3}", lines[0])
    assert lines[1] == "    kestrel \\x1b[2J wiped the screen"
    assert lines[2:9] == [f"    {word}" for word in ("a", "the", "of", "my", "is")] + [
        "    in",
        "    who",
    ]
    assert lines[9:] == ["    ... 2 more lines"]
    assert "\x1b" not in shown


def deep_searched(home: Path, *replies: str) -> tuple[int, str, str]:
    model = Local(*(says(text) for text in replies))
    with private_daemon(keyed(home), local=Serving(model), embedder=toy_embedder):
        knowledge("add", "notes")
        knowledge("index")
        return knowledge("search", "kestrel", "--variant", "words", "--deep")


SECONDS = r"\d+\.\d s asking the local model, \d+\.\d\d s searching"


@pytest.mark.parametrize(
    ("queries", "passage", "said"),
    [
        (["miso", "pune"], "a cat", "2 rewrites and a drafted passage"),
        (["miso"], "", "1 rewrite and no drafted passage"),
        ([], "a cat", "no rewrites and a drafted passage"),
    ],
)
def test_a_deep_search_says_what_the_model_wrote_before_the_passages(
    home: Path, queries: list[str], passage: str, said: str
) -> None:
    reply = json.dumps({"queries": queries, "passage": passage})

    code, shown, error = deep_searched(home, reply)

    lines = shown.splitlines()
    assert (code, error) == (0, "")
    assert re.fullmatch(f"deep: {said}, {SECONDS}", lines[0])
    assert re.fullmatch(r"notes/\w+\.txt:1 \| score \d+\.\d{3}", lines[1])


def test_a_deep_search_that_fell_back_says_so(home: Path) -> None:
    code, shown, _ = deep_searched(home, "no", "still no", "never")

    lines = shown.splitlines()
    assert code == 0
    assert re.fullmatch(f"deep: nothing usable written, {SECONDS}", lines[0])
    assert lines[1] == NO_REWRITES
    assert lines[2].startswith("notes/server.txt:1 | score ")


@pytest.mark.parametrize(
    ("arguments", "problem"),
    [
        (("add", "missing"), r"^error: .*missing cannot be added: "),
        (("search", "cat", "--variant", "loud"), r"^error: variant: 'loud' is not"),
    ],
    ids=["missing-folder", "unknown-variant"],
)
def test_what_the_daemon_refuses_is_an_error_and_exit_2(
    home: Path, arguments: tuple[str, ...], problem: str
) -> None:
    with private_daemon(keyed(home), embedder=toy_embedder):
        code, shown, error = knowledge(*arguments)

    assert (code, shown) == (FAIL, "")
    assert re.match(problem, error)


def test_a_daemon_with_no_embedding_model_says_how_to_get_one(home: Path) -> None:
    with private_daemon(keyed(home)):
        refused = knowledge("list")

    assert refused == (FAIL, "", f"error: {NOT_INDEXING}\n")


@pytest.mark.usefixtures("home")
def test_with_no_model_at_all_no_daemon_is_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYNTHIA_OPENROUTER_API_KEY")

    code, shown, error = knowledge("list")

    assert (code, shown) == (FAIL, "")
    assert error.startswith("error: no model")


def answering(result: object) -> object:
    async def ask(_home: Path, _method: str, _params: Params) -> object:
        return result

    return ask


@pytest.mark.usefixtures("home")
def test_an_index_pass_that_failed_says_why_and_exits_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "ask_daemon", answering({"error": "disk I/O error"}))

    assert knowledge("index") == (FAIL, "", "error: disk I/O error\n")


@pytest.mark.usefixtures("home")
def test_an_index_pass_names_the_folders_it_could_not_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = dict.fromkeys(
        (
            "added",
            "changed",
            "unchanged",
            "removed",
            "unreadable",
            "held_back",
            "passed_over",
            "chunks",
            "cut",
            "embedded",
        ),
        0,
    )
    result = counts | {"unavailable": ["share", "usb"], "seconds": 0.25, "error": None}
    monkeypatch.setattr(cli, "ask_daemon", answering(result))

    code, shown, _ = knowledge("index")

    assert code == 0
    assert shown.splitlines()[1] == (
        "could not list share, usb; what was read from it before is kept"
    )


@pytest.mark.usefixtures("home")
def test_an_index_pass_stopped_with_ctrl_c_says_what_stays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def interrupted(_home: Path, _method: str, _params: Params) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "ask_daemon", interrupted)

    assert knowledge("index") == (FAIL, "stopped; what was read so far is kept\n", "")
