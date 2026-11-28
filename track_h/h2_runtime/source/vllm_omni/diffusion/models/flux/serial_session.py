# SPDX-License-Identifier: Apache-2.0
"""Process-local cache ownership for one serial FLUX worker."""

from collections import OrderedDict
import hashlib
import os
import time


TAIL_PREWARM_MODES = ("l1_l2", "l1_l2_l3", "idle_prewarm")


def session_mode():
    mode = os.environ.get("SPARSEEDIT_TRACK_A_SESSION", "off")
    if mode not in ("off", "singleton", "demand", "eager", *TAIL_PREWARM_MODES):
        raise ValueError("unknown Track A serial session mode")
    return mode


class PostResponsePrewarm:
    """One-use work item published in a completed native FLUX response."""

    def __init__(self, capture):
        self.capture = capture
        self.ticket = time.perf_counter_ns()


def run_post_response_prewarm(pipeline, ticket, response_returned_ns):
    if session_mode() != "idle_prewarm":
        raise ValueError("post-response prewarm requires the idle_prewarm session")
    pending = getattr(pipeline, "_track_a_idle_prewarm", None)
    if pending is None or pending.ticket != ticket:
        raise ValueError("no pending post-response prewarm for this ticket")
    started = time.perf_counter_ns()
    if not ticket <= response_returned_ns <= started:
        raise ValueError("invalid client response return time for post-response prewarm")
    # The serialized worker RPC consumes the lease before it can perform work.
    pipeline._track_a_idle_prewarm = None
    capture = pending.capture()
    return {
        **capture,
        "ticket": ticket,
        "response_returned_ns": response_returned_ns,
        "capture_started_ns": started,
        "capture_finished_ns": time.perf_counter_ns(),
        "worker_pid": os.getpid(),
        "boundary": "after_native_client_response",
        "charged_to_previous_request": False,
    }


class _ProjectionCache(OrderedDict):
    def __init__(self):
        super().__init__()
        self.pinned = None

    def pin(self, key):
        if key not in self:
            raise KeyError("negative projections must be populated before pinning")
        self.pinned = key

    def clear(self):
        self.pinned = None
        super().clear()

    def popitem(self, last=True):
        order = reversed(self) if last else iter(self)
        key = next((key for key in order if key != self.pinned), None)
        if key is None:
            raise KeyError("no evictable condition projections")
        value = self.pop(key)
        return key, value


class UniversalNegativeBank:
    def __init__(self):
        self.key = None
        self.embeddings = None
        self.projection_cache = _ProjectionCache()

    def encode(self, key, factory):
        hit = self.embeddings is not None and key == self.key
        if not hit:
            self.embeddings = factory()
            self.key = key
        return self.embeddings, hit


class SourceImageSessionCache:
    def __init__(self, capacity=8):
        if capacity < 1:
            raise ValueError("positive source cache capacity is required")
        self.capacity = capacity
        self.entries = OrderedDict()

    @staticmethod
    def key(image, height, width, dtype):
        source = image.convert("RGB")
        pixels = hashlib.sha256()
        pixels.update(f"{source.width}x{source.height}\n".encode())
        pixels.update(source.tobytes())
        return pixels.hexdigest(), height, width, str(dtype)

    def get(self, key):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, key, value):
        self.entries[key] = tuple(tensor.detach() for tensor in value)
        self.entries.move_to_end(key)
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


def tail_menu(active_tokens):
    return tuple(ratio for count, ratio in ((460, .2), (921, .4)) if count != active_tokens)
