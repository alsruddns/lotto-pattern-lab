"""Optional, lazy PySR diagnostics; the core application never imports Julia at startup."""

from importlib import metadata, util
from typing import Any


def optional_backend_status() -> dict[str, Any]:
    """Inspect installation metadata without importing PySR or starting Julia."""
    if util.find_spec("pysr") is None:
        return {
            "available": False,
            "installed": False,
            "python_version_requirement": ">=3.10 (declared by current PySR project metadata)",
            "detail": "Install the optional symbolic extra in a separate environment step.",
            "install_command": ".venv\\Scripts\\python -m pip install -e .[symbolic]",
            "runtime_verified": False,
        }
    try:
        version = metadata.version("pysr")
    except metadata.PackageNotFoundError:
        version = "installed-without-package-metadata"
    return {
        "available": True,
        "installed": True,
        "version": version,
        "python_version_requirement": ">=3.10 (declared by current PySR project metadata)",
        "detail": "PySR import and Julia runtime have not been initialized by the core app.",
        "runtime_verified": False,
    }


def initialize_optional_backend() -> tuple[Any | None, dict[str, Any]]:
    """Explicitly load PySR only when requested, returning failures as data."""
    status = optional_backend_status()
    if not status["installed"]:
        return None, status
    try:
        from pysr import PySRRegressor
    except Exception as error:  # Julia setup can fail independently of the core application.
        return None, {
            **status,
            "available": False,
            "runtime_verified": True,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return PySRRegressor, {**status, "runtime_verified": True, "error": None}
