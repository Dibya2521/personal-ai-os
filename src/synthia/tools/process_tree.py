"""Start a process whose whole tree can be killed, on Windows and POSIX.

Killing a process leaves the processes it started running. On Windows the
child is put in a Job Object, which ``TerminateJobObject`` ends with every
process started inside it. On POSIX the child leads a new session, and
``killpg`` ends its process group. The child receives its input only after it
is contained, so nothing it starts can be outside.

A descendant can still leave: on POSIX by calling ``setsid``, and on Windows by
starting a process with ``CREATE_BREAKAWAY_FROM_JOB``, which the job refuses
unless it allows breakaway (this one does not).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING, Final, cast, override

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

KILLED_EXIT_CODE: Final = 1
DRAIN_TIMEOUT_S: Final = 5.0
_STDOUT: Final = 1

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.CreateJobObjectW.restype = wintypes.HANDLE
    _KERNEL32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _KERNEL32.SetInformationJobObject.restype = wintypes.BOOL
    _KERNEL32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _KERNEL32.OpenProcess.restype = wintypes.HANDLE
    _KERNEL32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _KERNEL32.AssignProcessToJobObject.restype = wintypes.BOOL
    _KERNEL32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _KERNEL32.TerminateJobObject.restype = wintypes.BOOL
    _KERNEL32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _KERNEL32.CloseHandle.restype = wintypes.BOOL
    _KERNEL32.CloseHandle.argtypes = (wintypes.HANDLE,)

    _PROCESS_TERMINATE: Final = 0x0001
    _PROCESS_SET_QUOTA: Final = 0x0100
    _NOT_INHERITED: Final = 0
    _KILL_ON_JOB_CLOSE: Final = 0x2000
    _EXTENDED_LIMIT_INFORMATION: Final = 9

    class _BasicLimits(ctypes.Structure):
        _fields_ = (
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _IoCounters(ctypes.Structure):
        _fields_ = tuple(
            (name, ctypes.c_uint64)
            for name in (
                "Reads",
                "Writes",
                "Others",
                "ReadBytes",
                "WriteBytes",
                "OtherBytes",
            )
        )

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = (
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )

    def _checked(ok: object) -> None:
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    class _Containment:
        """A Job Object; closing its last handle also ends every process in it."""

        def __init__(self) -> None:
            job = _KERNEL32.CreateJobObjectW(None, None)
            _checked(job)
            self._job: int | None = job
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
            try:
                _checked(
                    _KERNEL32.SetInformationJobObject(
                        job,
                        _EXTENDED_LIMIT_INFORMATION,
                        ctypes.byref(limits),
                        ctypes.sizeof(limits),
                    )
                )
            except OSError:
                self.close()
                raise

        def hold(self, pid: int) -> None:
            process = _KERNEL32.OpenProcess(
                _PROCESS_TERMINATE | _PROCESS_SET_QUOTA, _NOT_INHERITED, pid
            )
            _checked(process)
            try:
                _checked(_KERNEL32.AssignProcessToJobObject(self._job, process))
            finally:
                _KERNEL32.CloseHandle(process)

        def kill(self) -> None:
            if self._job is not None:
                _KERNEL32.TerminateJobObject(self._job, KILLED_EXIT_CODE)

        def close(self) -> None:
            # Once closed, the handle's number may be reused for something else.
            if self._job is not None:
                _KERNEL32.CloseHandle(self._job)
                self._job = None

else:

    class _Containment:
        """A new session; its process group is killed as one."""

        def __init__(self) -> None:
            self._group: int | None = None

        def hold(self, pid: int) -> None:
            self._group = pid

        def kill(self) -> None:
            if self._group is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._group, signal.SIGKILL)

        def close(self) -> None:
            self.kill()
            self._group = None


class _Collector(asyncio.SubprocessProtocol):
    """Keeps the output, and tells apart the process exiting and its output ending.

    ``asyncio.subprocess.Process.wait`` returns only once every pipe is closed,
    so a process the child left running, holding the output open, would make
    the child look alive; ``process_exited`` comes when the child itself exits.
    """

    def __init__(self, limit: int) -> None:
        self.output = bytearray()
        self.limit = limit
        self.exited = asyncio.Event()
        self.overflowed = asyncio.Event()
        self.output_closed = asyncio.Event()

    @override
    def pipe_data_received(self, fd: int, data: bytes) -> None:
        self.output += data
        if len(self.output) > self.limit:
            self.overflowed.set()

    @override
    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        if fd == _STDOUT:
            self.output_closed.set()

    @override
    def process_exited(self) -> None:
        self.exited.set()


class ProcessTree:
    """A running process and everything it starts, ended together.

    Output and errors arrive together in ``output``, at most a little past
    ``limit`` bytes before ``overflowed`` is set.
    """

    def __init__(
        self,
        transport: asyncio.SubprocessTransport,
        collector: _Collector,
        containment: _Containment,
    ) -> None:
        self._transport = transport
        self._collector = collector
        self._containment = containment
        self.output = collector.output
        self.exited = collector.exited
        self.overflowed = collector.overflowed
        self.output_closed = collector.output_closed

    @classmethod
    async def start(
        cls,
        argv: Sequence[str],
        *,
        stdin: bytes,
        cwd: Path,
        env: Mapping[str, str],
        limit: int,
    ) -> ProcessTree:
        """Start ``argv`` contained, then send it ``stdin`` and close its input."""
        containment = _Containment()
        loop = asyncio.get_running_loop()
        try:
            transport, collector = await loop.subprocess_exec(
                lambda: _Collector(limit),
                *argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=dict(env),
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                start_new_session=True,
            )
        except BaseException:
            containment.close()
            raise
        tree = cls(transport, collector, containment)
        try:
            containment.hold(transport.get_pid())
            pipe = cast("asyncio.WriteTransport", transport.get_pipe_transport(0))
            pipe.write(stdin)
            # Closing still sends what was written, unless the child exits first.
            pipe.close()
        except BaseException:
            await tree.end()
            raise
        return tree

    @property
    def exit_code(self) -> int | None:
        """Return the process's exit code, or None while it runs."""
        return self._transport.get_returncode()

    async def end(self) -> None:
        """End the process and everything it started; safe to call again."""
        self._containment.kill()
        if self.exit_code is None:
            # Not contained if start failed between spawning and holding it.
            with contextlib.suppress(ProcessLookupError):
                self._transport.kill()
        await self.exited.wait()
        self._containment.close()
        # Once every writer is gone the output ends by itself, with nothing lost;
        # only a descendant that left the tree could keep it open.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.output_closed.wait(), DRAIN_TIMEOUT_S)
        self._transport.close()
