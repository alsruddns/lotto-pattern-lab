import os

import pytest

from lotto_pattern_lab.windows_process import (
    add_no_console_flag,
    enable_hidden_multiprocessing_children,
)


def test_only_multiprocessing_spawn_gets_no_console_flag():
    calls = []

    def fake_create_process(*args, **kwargs):
        calls.append((args, kwargs))
        return "process-created"

    args = (
        "python.exe",
        'python.exe -c "from multiprocessing.spawn import spawn_main" --multiprocessing-fork',
        None,
        None,
        False,
        0x00000200,
        None,
        None,
        object(),
    )

    result = add_no_console_flag(fake_create_process, *args)

    assert result == "process-created"
    assert calls[0][0][5] == 0x00000200 | 0x08000000
    assert calls[0][0][:5] == args[:5]
    assert calls[0][0][6:] == args[6:]


def test_regular_subprocess_creation_flags_remain_unchanged():
    calls = []

    def fake_create_process(*args, **kwargs):
        calls.append((args, kwargs))
        return 4321

    args = ("julia.exe", "julia.exe --project", None, None, False, 0x00000200)
    result = add_no_console_flag(fake_create_process, *args)

    assert result == 4321
    assert calls == [(args, {})]


@pytest.mark.skipif(os.name != "nt", reason="Windows multiprocessing spawn only")
def test_windows_spawn_hook_installs_idempotently_and_preserves_process_result(monkeypatch):
    import multiprocessing.popen_spawn_win32 as spawn_win32

    calls = []

    def fake_create_process(*args, **kwargs):
        calls.append((args, kwargs))
        return ("handle", "thread", 2468, 9)

    monkeypatch.setattr(spawn_win32._winapi, "CreateProcess", fake_create_process)

    assert enable_hidden_multiprocessing_children() is True
    assert enable_hidden_multiprocessing_children() is True
    result = spawn_win32._winapi.CreateProcess(
        "python.exe",
        'python.exe -c "from multiprocessing.spawn import spawn_main" --multiprocessing-fork',
        None,
        None,
        False,
        0,
        None,
        None,
        object(),
    )

    assert result == ("handle", "thread", 2468, 9)
    assert calls[0][0][5] == 0x08000000
