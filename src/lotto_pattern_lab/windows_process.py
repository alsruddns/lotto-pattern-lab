"""Windows process creation adjustments for research workers."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any

_MULTIPROCESSING_ARGUMENT = "--multiprocessing-fork"
_CREATE_NO_WINDOW = 0x08000000
_PATCH_MARKER = "_lotto_pattern_lab_hidden_spawn_console"


def add_no_console_flag(
    create_process: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Call CreateProcess without a console for multiprocessing spawn children only."""
    command_line = args[1] if len(args) > 1 else kwargs.get("command_line")
    if not isinstance(command_line, str) or _MULTIPROCESSING_ARGUMENT not in command_line:
        return create_process(*args, **kwargs)

    positional = list(args)
    if len(positional) > 5:
        positional[5] = int(positional[5]) | _CREATE_NO_WINDOW
    elif "creation_flags" in kwargs:
        kwargs = {**kwargs, "creation_flags": int(kwargs["creation_flags"]) | _CREATE_NO_WINDOW}
    else:
        return create_process(*args, **kwargs)
    return create_process(*positional, **kwargs)


def enable_hidden_multiprocessing_children() -> bool:
    """Hide Windows spawn-worker consoles while preserving normal process creation."""
    if os.name != "nt":
        return False

    try:
        import multiprocessing.popen_spawn_win32 as spawn_win32
    except ImportError:
        return False

    winapi = spawn_win32._winapi
    create_process = winapi.CreateProcess
    if getattr(create_process, _PATCH_MARKER, False):
        return True

    @wraps(create_process)
    def create_process_with_hidden_spawn_console(*args: Any, **kwargs: Any) -> Any:
        return add_no_console_flag(create_process, *args, **kwargs)

    setattr(create_process_with_hidden_spawn_console, _PATCH_MARKER, True)
    winapi.CreateProcess = create_process_with_hidden_spawn_console
    return True
