import json
import logging
from pathlib import Path

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import SecretStr
from typer.testing import CliRunner

from synthia.gateway.assemble import build_gateway
from synthia.gateway.errors import RemoteUnavailableError
from synthia.gateway.openai_compat import Dialect
from synthia.gateway.protocol import collect
from synthia.gateway.providers import (
    DEFAULT_DAILY_CAP,
    DEFAULT_REQUESTS_PER_MINUTE,
    HIDDEN_NAME,
    MAX_INT32,
    MAX_PROVIDERS_FILE_BYTES,
    OPENROUTER,
    OPENROUTER_FREE,
    PROVIDER_NAME,
    PROVIDERS_FILE,
    key_variable,
    load_providers,
    remote_key,
    select_remote,
)
from synthia.gateway.types import ChatRequest, Message, ModelInfo, Reasoning
from synthia.interfaces.cli import app
from synthia.kernel.bus import Event
from synthia.kernel.config import Settings, load_settings
from synthia.kernel.errors import ConfigError
from tests.models.test_service import service_at

PLANTED = "sk-planted-0123456789abcdef"  # pragma: allowlist secret
LONG_KEY = PLANTED + "x" * 45
AT_MOST_INT32 = f"Input should be less than or equal to {MAX_INT32}"
REMOTE_KEY = "sk-remote-test-key-000"  # pragma: allowlist secret
OPENROUTER_KEY = "sk-or-v1-providers-test-key-000"  # pragma: allowlist secret
OPENAI_URL = "https://api.openai.com/v1"
VALID = f"""
[providers.openai]
base_url = "{OPENAI_URL}"
model = "<model id from the vendor>"
context_window = 128000
tools = true

[providers.home-server]
base_url = "http://127.0.0.1:8080/v1"
model = "local-model"
context_window = 8192
requests_per_minute = 60
daily_cap = 5000
dialect = "llama.cpp"
"""
KEY_A = "sk-vendor-a-test-key-000"  # pragma: allowlist secret
KEY_B = "sk-vendor-b-test-key-000"  # pragma: allowlist secret
VENDORS = """
[providers.vendor-a]
base_url = "https://a.example/v1"
model = "a-model"
context_window = 8192

[providers.vendor-b]
base_url = "https://b.example/v1"
model = "b-model"
context_window = 8192
"""
HELLO = ChatRequest((Message.user("hello"),), reasoning=Reasoning.OFF, use_remote=True)


def written(home: Path, text: str) -> Path:
    (home / PROVIDERS_FILE).write_text(text, encoding="utf-8")
    return home


def table(base_url: str = OPENAI_URL, **fields: str) -> str:
    lines = [f'base_url = "{base_url}"', 'model = "m"', "context_window = 8"]
    lines += [f"{name} = {value}" for name, value in fields.items()]
    return "[providers.x]\n" + "\n".join(lines) + "\n"


def test_load_providers_adds_each_table_after_the_built_in(tmp_path: Path) -> None:
    found = load_providers(written(tmp_path, VALID))

    assert list(found) == [OPENROUTER, "openai", "home-server"]
    assert found[OPENROUTER] is OPENROUTER_FREE
    openai, home = found["openai"], found["home-server"]
    assert openai.model == ModelInfo(
        "<model id from the vendor>", 128_000, vision=False, tools=True
    )
    assert (openai.base_url, openai.dialect, dict(openai.headers)) == (
        OPENAI_URL,
        Dialect.OPENAI,
        {},
    )
    assert (openai.requests_per_minute, openai.daily_cap) == (
        DEFAULT_REQUESTS_PER_MINUTE,
        DEFAULT_DAILY_CAP,
    )
    assert not openai.reports_daily_cap
    assert (home.dialect, home.requests_per_minute, home.daily_cap) == (
        Dialect.LLAMA_CPP,
        60,
        5000,
    )


def test_load_providers_without_a_file_is_only_the_built_in(tmp_path: Path) -> None:
    assert load_providers(tmp_path) == {OPENROUTER: OPENROUTER_FREE}


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:8080/v1",
        "http://localhost:11434/v1",
        "http://[::1]:8080/v1",
    ],
)
def test_http_is_accepted_for_this_machine(tmp_path: Path, base_url: str) -> None:
    found = load_providers(written(tmp_path, table(base_url)))

    assert found["x"].base_url == base_url


@pytest.mark.parametrize(
    ("text", "named"),
    [
        (
            table(api_key=f'"{PLANTED}"'),
            "[providers.x] api_key: Extra inputs are not permitted",
        ),
        (
            f'[providers.x]\nbase_url = "{OPENAI_URL}"\nmodel = "{PLANTED}"\n',
            "[providers.x] context_window: Field required",
        ),
        (
            table(context_window=f'"{PLANTED}"').replace("context_window = 8\n", ""),
            "[providers.x] context_window: Input should be a valid integer",
        ),
        (
            table(f"http://api.example.com/v1/{PLANTED}"),
            "[providers.x] base_url: base_url must start with https://",
        ),
        (
            table(f"https://user:{PLANTED}@api.example.com/v1"),
            "[providers.x] base_url: base_url must not carry a user name or password",
        ),
        (
            table(f"https://api.example.com/v1?key={PLANTED}"),
            "[providers.x] base_url: base_url must not have a query",
        ),
        (
            table(f"https://api.example.com/{PLANTED}").replace("x]", "My_AI]"),
            "[providers.My_AI]: a provider name is 1 to 32 lowercase letters",
        ),
        (
            table(f"https://evil.example/{PLANTED}").replace("x]", "openrouter]"),
            "[providers.openrouter] is built in and cannot be redefined",
        ),
        (f'[providers.x]\nmodel = "{PLANTED}\n', "is not valid TOML at line 2"),
        (
            table(f"https:///{PLANTED}"),
            "[providers.x] base_url: base_url needs a host",
        ),
        (
            table(f"https://api.example.com:99999/{PLANTED}"),
            "[providers.x] base_url: base_url has a port that is not a number",
        ),
        (
            f'[other]\nbase_url = "https://evil.example/{PLANTED}"\n',
            "other: Extra inputs are not permitted",
        ),
        (
            table(f"http://127.9.9.9/v1/{PLANTED}"),
            "base_url must start with https://, or with http:// only for this",
        ),
        (
            table(f"https://api.example.com/v1 {PLANTED}"),
            "[providers.x] base_url: base_url must not contain spaces or control",
        ),
        (
            table(f"https://api.example.com/v1\\u0007{PLANTED}"),
            "[providers.x] base_url: base_url must not contain spaces or control",
        ),
        (
            table(f"https://api.example.com/{PLANTED}").replace("x]", f"{LONG_KEY}]"),
            f"[providers.{HIDDEN_NAME}]: a provider name is 1 to 32 lowercase",
        ),
        (
            f"{LONG_KEY} = 1\n",
            f"{HIDDEN_NAME}: Extra inputs are not permitted",
        ),
        (
            table(**{LONG_KEY: "1"}),
            f"[providers.x] {HIDDEN_NAME}: Extra inputs are not permitted",
        ),
        (
            f"{PLANTED} = {'[' * 30_000}\n",
            "cannot be read: it nests too deeply or holds a number",
        ),
        (
            f"{PLANTED} = {'9' * 5_000}\n",
            "cannot be read: it nests too deeply or holds a number",
        ),
        (
            table(daily_cap=str(2**70)),
            f"[providers.x] daily_cap: {AT_MOST_INT32}",
        ),
    ],
    ids=[
        "unknown field",
        "no context window",
        "quoted number",
        "http to a public host",
        "user and password",
        "query",
        "bad name",
        "redefines openrouter",
        "not toml",
        "no host",
        "bad port",
        "unknown top-level table",
        "http to another loopback address",
        "space in the url",
        "control character in the url",
        "key as a table name",
        "key as a top-level name",
        "key as a field name",
        "deep nesting",
        "too many digits",
        "number past 32 bits",
    ],
)
def test_a_bad_file_names_the_problem_and_never_a_value(
    tmp_path: Path, text: str, named: str
) -> None:
    with pytest.raises(ConfigError) as raised:
        load_providers(written(tmp_path, text))

    message = str(raised.value)
    assert str(tmp_path / PROVIDERS_FILE) in message
    assert named in message
    assert PLANTED not in message


def test_an_oversized_file_is_refused_before_it_is_parsed(tmp_path: Path) -> None:
    padding = "#" * MAX_PROVIDERS_FILE_BYTES
    written(tmp_path, f'[providers.x]\nmodel = "{PLANTED}"\n{padding}\n')

    with pytest.raises(ConfigError, match=r"larger than 65,536 bytes") as raised:
        load_providers(tmp_path)

    assert PLANTED not in str(raised.value)


def test_a_file_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    (tmp_path / PROVIDERS_FILE).mkdir()

    with pytest.raises(ConfigError, match=r"could not be read"):
        load_providers(tmp_path)


def test_a_file_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    (tmp_path / PROVIDERS_FILE).write_bytes(b"[providers.x]\nmodel = '\xff'\n")

    with pytest.raises(ConfigError, match=r"is not UTF-8 text"):
        load_providers(tmp_path)


def test_select_remote_defaults_to_the_built_in_openrouter(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path)

    assert settings.remote == OPENROUTER
    assert select_remote(settings) is OPENROUTER_FREE


def test_select_remote_never_repeats_a_key_pasted_as_the_name(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path, remote=LONG_KEY)

    with pytest.raises(ConfigError) as raised:
        select_remote(settings)

    assert f"SYNTHIA_REMOTE={HIDDEN_NAME} is not a provider" in str(raised.value)
    assert PLANTED not in str(raised.value)


def test_select_remote_names_the_known_providers_for_an_unknown_one(
    tmp_path: Path,
) -> None:
    settings = Settings(home=written(tmp_path, VALID), remote="gemini")

    with pytest.raises(
        ConfigError,
        match=r"SYNTHIA_REMOTE=gemini is not a provider; "
        r"the providers are openrouter, openai, home-server",
    ):
        select_remote(settings)


@pytest.mark.parametrize(
    ("name", "variable"),
    [
        (OPENROUTER, "SYNTHIA_OPENROUTER_API_KEY"),
        ("deepseek", "SYNTHIA_DEEPSEEK_API_KEY"),
        ("my-local", "SYNTHIA_MY_LOCAL_API_KEY"),
    ],
)
def test_each_provider_has_its_own_key_variable(name: str, variable: str) -> None:
    assert key_variable(name) == variable


@given(st.lists(st.from_regex(PROVIDER_NAME, fullmatch=True), unique=True))
def test_no_two_provider_names_share_a_key_variable(names: list[str]) -> None:
    assert len({key_variable(name) for name in names}) == len(names)


@pytest.mark.parametrize("name", ["My-Local", "my_local"])
def test_a_name_that_could_share_a_variable_is_refused(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(ConfigError, match=r"a provider name is 1 to 32 lowercase"):
        load_providers(written(tmp_path, table().replace("x]", f"{name}]")))


def test_remote_key_reads_only_the_providers_own_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = load_providers(written(tmp_path, VALID))
    monkeypatch.setenv("SYNTHIA_OPENAI_API_KEY", REMOTE_KEY)
    monkeypatch.delenv("SYNTHIA_HOME_SERVER_API_KEY", raising=False)
    settings = Settings(home=tmp_path, openrouter_api_key=SecretStr(OPENROUTER_KEY))

    assert remote_key(settings, OPENROUTER_FREE) == SecretStr(OPENROUTER_KEY)
    assert remote_key(settings, found["openai"]) == SecretStr(REMOTE_KEY)
    assert remote_key(settings, found["home-server"]) is None


def recording(sent: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        events = [{"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, content=(body + "data: [DONE]\n\n").encode())

    return httpx.MockTransport(handler)


async def test_a_request_goes_to_the_chosen_provider_with_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[httpx.Request] = []
    monkeypatch.setenv("SYNTHIA_OPENAI_API_KEY", REMOTE_KEY)
    settings = Settings(
        home=written(tmp_path, VALID),
        remote="openai",
        openrouter_api_key=SecretStr(OPENROUTER_KEY),
    )
    async with httpx.AsyncClient(transport=recording(sent)) as client:
        gateway = build_gateway(settings, client, _ignore)
        reply = await collect(gateway.model.stream(HELLO))

    (request,) = sent
    body = json.loads(request.content)
    assert reply.text == "hi"
    assert str(request.url) == f"{OPENAI_URL}/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {REMOTE_KEY}"
    assert {"HTTP-Referer", "X-Title"}.isdisjoint(request.headers)
    assert body["model"] == "<model id from the vendor>"
    assert {"reasoning", "chat_template_kwargs", "return_progress"}.isdisjoint(body)


async def test_switching_the_remote_never_sends_one_providers_key_to_another(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    home = written(tmp_path, VENDORS)
    monkeypatch.setenv("SYNTHIA_VENDOR_A_API_KEY", KEY_A)
    monkeypatch.setenv("SYNTHIA_VENDOR_B_API_KEY", KEY_B)
    sent: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=recording(sent)) as client:
        for remote in ("vendor-a", "vendor-b"):
            chosen = Settings(home=home, remote=remote)
            await collect(build_gateway(chosen, client, _ignore).model.stream(HELLO))
        monkeypatch.delenv("SYNTHIA_VENDOR_B_API_KEY")
        local = service_at(tmp_path).model(client)
        unkeyed = build_gateway(
            Settings(home=home, remote="vendor-b"), client, _ignore, local
        )
        with pytest.raises(RemoteUnavailableError):
            await collect(unkeyed.model.stream(HELLO))

    to_a, to_b = sent
    assert (to_a.url.host, to_b.url.host) == ("a.example", "b.example")
    assert to_a.headers["Authorization"] == f"Bearer {KEY_A}"
    assert to_b.headers["Authorization"] == f"Bearer {KEY_B}"
    assert KEY_A not in str(to_b.headers) + to_b.content.decode() + str(to_b.url)
    assert unkeyed.router.remote is None
    assert KEY_A not in caplog.text
    assert KEY_B not in caplog.text


async def test_a_key_in_the_env_file_is_sent_to_its_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"SYNTHIA_REMOTE=vendor-b\nSYNTHIA_VENDOR_B_API_KEY={KEY_B}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SYNTHIA_HOME", str(written(tmp_path, VENDORS)))
    monkeypatch.delenv("SYNTHIA_VENDOR_B_API_KEY", raising=False)
    monkeypatch.delenv("SYNTHIA_REMOTE", raising=False)
    sent: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=recording(sent)) as client:
        gateway = build_gateway(load_settings(env_file), client, _ignore)
        await collect(gateway.model.stream(HELLO))

    (request,) = sent
    assert request.url.host == "b.example"
    assert request.headers["Authorization"] == f"Bearer {KEY_B}"


def test_the_providers_command_lists_each_without_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNTHIA_HOME", str(written(tmp_path, VALID)))
    monkeypatch.setenv("SYNTHIA_REMOTE", "openai")
    monkeypatch.setenv("SYNTHIA_OPENAI_API_KEY", REMOTE_KEY)
    monkeypatch.delenv("SYNTHIA_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SYNTHIA_HOME_SERVER_API_KEY", raising=False)

    result = CliRunner().invoke(app, ["providers"])

    assert result.exit_code == 0
    assert result.stdout.splitlines() == [
        (
            "openrouter: https://openrouter.ai/api/v1, model openrouter/free, "
            "not selected, no key"
        ),
        f"openai: {OPENAI_URL}, model <model id from the vendor>, selected, key set",
        (
            "home-server: http://127.0.0.1:8080/v1, model local-model, "
            "not selected, no key"
        ),
    ]
    assert REMOTE_KEY not in result.stdout + result.stderr


def test_the_providers_command_exits_2_on_a_bad_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNTHIA_HOME", str(written(tmp_path, "not toml\n")))

    result = CliRunner().invoke(app, ["providers"])

    assert result.exit_code == 2
    assert result.stderr.startswith("error: ")
    assert "is not valid TOML at line 1" in result.stderr


async def _ignore(_: Event) -> None:
    return None
