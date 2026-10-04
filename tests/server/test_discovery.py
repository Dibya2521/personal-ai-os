import logging
import os
import stat
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import psutil
import pytest

from synthia.server import discovery
from synthia.server.discovery import (
    DaemonInfo,
    is_running,
    new_token,
    patiently,
    read_info,
    remove_info,
    started_now,
    write_info,
)


def info(pid: int, started: datetime | None = None) -> DaemonInfo:
    return DaemonInfo(
        port=45678,
        pid=pid,
        version="0.2.0",
        started=started or started_now(),
        token=new_token(),
    )


def test_the_file_reads_back_as_written(tmp_path: Path) -> None:
    path = tmp_path / "home" / "daemon.json"
    written = info(os.getpid())

    write_info(path, written)

    assert read_info(path) == written
    assert written.base_url == "http://127.0.0.1:45678"
    assert written.socket_url == "ws://127.0.0.1:45678/ws"
    assert written.token not in repr(written)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_only_its_owner_can_read_the_file(tmp_path: Path) -> None:
    path = tmp_path / "daemon.json"
    write_info(path, info(os.getpid()))

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_no_file_or_a_broken_one_is_no_daemon(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")

    with caplog.at_level(logging.WARNING):
        found = (read_info(tmp_path / "missing.json"), read_info(broken))

    assert found == (None, None)
    assert "ignored an unreadable broken.json" in caplog.text


def test_a_daemon_is_running_only_while_its_process_is_the_one_that_wrote_it() -> None:
    here = psutil.Process()
    born = datetime.fromtimestamp(here.create_time(), UTC)
    unused = max(psutil.pids()) + 100_000

    assert is_running(info(here.pid))
    assert not is_running(info(here.pid, started=born.replace(year=born.year - 1)))
    assert not is_running(info(unused))


def test_the_file_is_removed_only_by_the_daemon_it_describes(tmp_path: Path) -> None:
    path = tmp_path / "daemon.json"
    write_info(path, info(4242))

    remove_info(path, 999)
    kept = path.exists()
    remove_info(path, 4242)
    remove_info(path, 4242)

    assert (kept, path.exists()) == (True, False)


def test_a_client_reading_all_the_while_never_stops_a_write_or_a_removal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "daemon.json"
    written = info(os.getpid())
    done = threading.Event()
    seen: list[DaemonInfo | None] = []

    def keep_reading() -> None:
        while not done.is_set():
            seen.append(read_info(path))

    reader = threading.Thread(target=keep_reading)
    reader.start()
    try:
        for _ in range(100):
            write_info(path, written)
            remove_info(path, written.pid)
            assert not path.exists()
    finally:
        done.set()
        reader.join()

    assert all(found in (written, None) for found in seen)


def test_a_refusal_that_outlasts_the_wait_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(discovery, "SHARING_WAIT_S", 0.05)
    attempts: list[int] = []

    def refused() -> None:
        attempts.append(1)
        raise PermissionError(13, "in use")

    with pytest.raises(PermissionError, match="in use"):
        patiently(refused)

    assert len(attempts) > 1
