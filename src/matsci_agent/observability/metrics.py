"""Process-wide metrics in the Prometheus text exposition format (no client library needed).

Counters only go up; observations keep a running count and sum, enough for rates and mean latency.
"""

from __future__ import annotations

import threading

PREFIX = "crucible_"


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    body = ",".join(f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"' for k, v in sorted(labels.items()))
    return "{" + body + "}"


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple], float] = {}
        self._sums: dict[tuple[str, tuple], list[float]] = {}  # [count, sum]
        self._help: dict[str, str] = {}

    def describe(self, name: str, text: str) -> None:
        self._help[name] = text

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            entry = self._sums.setdefault(key, [0.0, 0.0])
            entry[0] += 1
            entry[1] += value

    def render(self) -> str:
        with self._lock:
            counters = dict(self._counters)
            sums = {k: list(v) for k, v in self._sums.items()}
        lines: list[str] = []
        for name in sorted({n for n, _ in counters}):
            lines.append(f"# HELP {PREFIX}{name} {self._help.get(name, name)}")
            lines.append(f"# TYPE {PREFIX}{name} counter")
            lines += [f"{PREFIX}{n}{_labels(dict(lab))} {v:g}" for (n, lab), v in sorted(counters.items()) if n == name]
        for name in sorted({n for n, _ in sums}):
            lines.append(f"# HELP {PREFIX}{name} {self._help.get(name, name)}")
            lines.append(f"# TYPE {PREFIX}{name} summary")
            for (n, lab), (count, total) in sorted(sums.items()):
                if n == name:
                    lines.append(f"{PREFIX}{n}_count{_labels(dict(lab))} {count:g}")
                    lines.append(f"{PREFIX}{n}_sum{_labels(dict(lab))} {total:g}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._sums.clear()


METRICS = Metrics()
METRICS.describe("runs_total", "Research runs by outcome")
METRICS.describe("llm_calls_total", "LLM calls by model, agent and outcome (ok, error, cache_hit)")
METRICS.describe("llm_tokens_total", "LLM tokens by model and direction")
METRICS.describe("llm_call_seconds", "LLM call latency in seconds")
METRICS.describe("run_seconds", "End-to-end research run duration in seconds")
