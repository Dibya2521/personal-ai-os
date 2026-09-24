import socket
from pathlib import Path

import pytest
from pydantic import SecretStr

from synthia.kernel.config import DEFAULT_LOCAL_MODEL, LocalBackend
from synthia.models.catalogue import Backend, Download, Model, find
from synthia.models.server import (
    API_KEY_VARIABLE,
    LOOPBACK,
    Launch,
    ServerError,
    find_server,
    free_port,
    new_key,
)

KEY = SecretStr("launch-test-key")
SECRET = KEY.get_secret_value()
TEXT_ONLY = Model(
    "tiny",
    "example/tiny",
    "0" * 40,
    "mit",
    Download("tiny.gguf", "https://example.com/tiny.gguf", 1, "0" * 64),
)


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.mark.parametrize("name", ["llama-server.exe", "llama-server"])
def test_the_server_is_found_at_the_top_of_a_flat_archive(
    tmp_path: Path, name: str
) -> None:
    touch(tmp_path / "llama-cli.exe")
    server = touch(tmp_path / name)

    assert find_server(tmp_path) == server


def test_the_shallowest_server_wins_in_a_nested_archive(tmp_path: Path) -> None:
    touch(tmp_path / "build" / "bin" / "extra" / "llama-server")
    server = touch(tmp_path / "build" / "bin" / "llama-server")

    assert find_server(tmp_path) == server


def test_a_similar_name_is_not_the_server(tmp_path: Path) -> None:
    touch(tmp_path / "llama-server-helper.exe")
    touch(tmp_path / "llama-server.pdb")

    with pytest.raises(ServerError, match="no llama-server"):
        find_server(tmp_path)


def test_a_free_port_can_be_bound_on_loopback() -> None:
    port = free_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK, port))


def test_each_launch_gets_its_own_key() -> None:
    assert new_key().get_secret_value() != new_key().get_secret_value()


def launch_of(tmp_path: Path, model: Model) -> Launch:
    touch(tmp_path / "runtime" / "llama-server")
    return Launch.of(
        Backend.VULKAN,
        tmp_path / "runtime",
        model,
        tmp_path / "model",
        port=41234,
        context=8192,
        key=KEY,
    )


def test_the_command_binds_loopback_offline_without_the_web_ui(tmp_path: Path) -> None:
    model = find(DEFAULT_LOCAL_MODEL)
    assert isinstance(model, Model)
    assert model.projector is not None
    launch = launch_of(tmp_path, model)

    command = launch.command()

    assert command[0] == str(tmp_path / "runtime" / "llama-server")
    pairs = dict(zip(command[1::2], command[2::2], strict=False))
    assert pairs["--model"] == str(tmp_path / "model" / model.weights.name)
    assert pairs["--mmproj"] == str(tmp_path / "model" / model.projector.name)
    assert (pairs["--host"], pairs["--port"]) == (LOOPBACK, "41234")
    assert pairs["--ctx-size"] == "8192"
    assert {"--no-webui", "--offline"} <= set(command)
    assert launch.base_url == f"http://{LOOPBACK}:41234/v1"
    assert launch.health_url == f"http://{LOOPBACK}:41234/health"


def test_a_text_only_model_is_launched_without_a_projector(tmp_path: Path) -> None:
    command = launch_of(tmp_path, TEXT_ONLY).command()

    assert "--mmproj" not in command


def test_the_key_goes_in_the_environment_never_on_the_command_line(
    tmp_path: Path,
) -> None:
    launch = launch_of(tmp_path, TEXT_ONLY)

    environment = launch.environment({"PATH": "somewhere"})

    assert environment == {"PATH": "somewhere", API_KEY_VARIABLE: SECRET}
    assert all(SECRET not in arg for arg in launch.command())
    assert SECRET not in repr(launch)


def test_every_local_backend_but_auto_names_a_catalogue_backend() -> None:
    names = {b.value for b in LocalBackend} - {LocalBackend.AUTO.value}

    assert names == {b.value for b in Backend}


def test_the_default_local_model_is_in_the_catalogue() -> None:
    assert isinstance(find(DEFAULT_LOCAL_MODEL), Model)
