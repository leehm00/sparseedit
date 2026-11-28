# SPDX-License-Identifier: Apache-2.0
"""Request-local CUDA event pairs for the Track A anchor."""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

_CURRENT = ContextVar("track_a_device_ledger", default=None)


class DeviceLedger:
    def __init__(self, event_factory):
        self.event_factory = event_factory
        self.intervals = defaultdict(list)

    def __enter__(self):
        self.token = _CURRENT.set(self)
        return self

    def __exit__(self, *_):
        _CURRENT.reset(self.token)

    @contextmanager
    def measure(self, name):
        start = self.event_factory()
        end = self.event_factory()
        start.record()
        yield
        end.record()
        self.intervals[name].append((start, end))

    def seconds(self, name):
        intervals = self.intervals[name]
        total_ms = 0.0
        for start, end in intervals:
            end.synchronize()
            total_ms += start.elapsed_time(end)
        return total_ms / 1000.0, len(intervals)


def copy_ledger():
    ledger = _CURRENT.get()
    return ledger.measure("graph_output_copy") if ledger is not None else nullcontext()
