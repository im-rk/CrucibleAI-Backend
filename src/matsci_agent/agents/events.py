"""Progress events emitted by agent nodes for live streaming to the UI."""

from __future__ import annotations

from typing import Any


async def emit(config: Any, event: str, **data: Any) -> None:
    """Send `event` to the run's listener (config['configurable']['emit']), if one is attached."""
    listener = ((config or {}).get("configurable") or {}).get("emit")
    if listener is not None:
        await listener({"event": event, **data})
