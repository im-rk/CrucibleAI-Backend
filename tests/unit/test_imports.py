"""Every module in the package must import cleanly (catches syntax/import errors early)."""

from __future__ import annotations

import importlib
import pkgutil

import matsci_agent


def test_every_module_imports() -> None:
    failures = []
    for module in pkgutil.walk_packages(matsci_agent.__path__, prefix="matsci_agent."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:  # noqa: BLE001 - report every broken module, not just the first
            failures.append(f"{module.name}: {type(exc).__name__}: {exc}")
    assert not failures, "\n".join(failures)
