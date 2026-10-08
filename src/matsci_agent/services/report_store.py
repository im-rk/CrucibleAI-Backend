"""Recent finished reports, kept in memory so follow-up questions do not need a database."""

from __future__ import annotations

import threading
from collections import OrderedDict


class ReportStore:
    """A small LRU of session id -> report."""

    def __init__(self, capacity: int = 25) -> None:
        self.capacity = capacity
        self._reports: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, session_id: str, report: dict) -> None:
        with self._lock:
            self._reports[session_id] = report
            self._reports.move_to_end(session_id)
            while len(self._reports) > self.capacity:
                self._reports.popitem(last=False)

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            report = self._reports.get(session_id)
            if report is not None:
                self._reports.move_to_end(session_id)
            return report
