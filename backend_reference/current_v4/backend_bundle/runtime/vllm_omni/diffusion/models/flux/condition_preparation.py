# SPDX-License-Identifier: Apache-2.0
"""Request-scoped condition preparation launched after warmup computation is queued."""

from __future__ import annotations

import time

import torch


class _SynchronousQueue:
    def launch(self, function, *, producer_start):
        function()

    def record_consumer(self, plan):
        pass

    def wait(self):
        pass

    def join(self):
        pass

    def metadata(self, *, denoise_start):
        return {
            "backend": "cpu_synchronous",
            "device_seconds": None,
            "consumer_wait_device_seconds": None,
        }


class _CudaQueue:
    def __init__(self, plan, device, overlap):
        self.consumer = torch.cuda.current_stream(device)
        self.overlap = overlap
        self.stream = (
            getattr(plan, "_warmup_preparation_stream", None)
            if overlap
            else self.consumer
        )
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device, priority=0)
            plan._warmup_preparation_stream = self.stream
        self.inputs_ready = torch.cuda.Event()
        # Only request input producers precede this event. Later warmup kernels
        # must not become dependencies of the auxiliary preparation stream.
        self.inputs_ready.record(self.consumer)
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.producer_end = torch.cuda.Event(enable_timing=True)
        self.wait_begin = torch.cuda.Event(enable_timing=True)
        self.wait_end = torch.cuda.Event(enable_timing=True)
        self.producer_start = None
        self.launched = self.waited = self.end_recorded = False

    def launch(self, function, *, producer_start):
        self.producer_start = producer_start
        self.producer_end.record(self.consumer)
        self.launched = True
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.inputs_ready)
            self.begin.record()
            try:
                function()
            finally:
                self.end.record()
                self.end_recorded = True

    def record_consumer(self, plan):
        for name in (
            "current_temb",
            "current_condition",
            "current_projected",
            "temb_values",
            "condition_values",
            "projected_values",
        ):
            tensor = getattr(plan, name, None)
            if tensor is not None and tensor.is_cuda:
                tensor.record_stream(self.consumer)

    def wait(self):
        self.wait_begin.record(self.consumer)
        self.consumer.wait_event(self.end)
        self.wait_end.record(self.consumer)
        self.waited = True

    def join(self):
        if self.end_recorded:
            self.end.synchronize()
        elif self.launched:
            self.stream.synchronize()

    def metadata(self, *, denoise_start):
        result = {
            "backend": "cuda_auxiliary_stream"
            if self.overlap
            else "cuda_default_stream",
            "stream_id": self.stream.cuda_stream,
            "device_seconds": None,
            "consumer_wait_device_seconds": None,
        }
        if not self.launched:
            return result
        result["device_seconds"] = self.begin.elapsed_time(self.end) / 1000
        if self.waited:
            result["consumer_wait_device_seconds"] = (
                self.wait_begin.elapsed_time(self.wait_end) / 1000
            )
        if denoise_start is not None:
            start = denoise_start.elapsed_time(self.begin) / 1000
            end = denoise_start.elapsed_time(self.end) / 1000
            result.update(start_offset_seconds=start, end_offset_seconds=end)
            if self.producer_start is not None:
                producer_start = denoise_start.elapsed_time(self.producer_start) / 1000
                producer_end = denoise_start.elapsed_time(self.producer_end) / 1000
                result.update(
                    producer_start_offset_seconds=producer_start,
                    producer_end_offset_seconds=producer_end,
                    overlap_with_enqueued_window_seconds=max(
                        0.0, min(end, producer_end) - max(start, producer_start)
                    ),
                )
        return result


def _make_queue(plan, device, overlap):
    return (
        _CudaQueue(plan, device, overlap)
        if device.type == "cuda"
        else _SynchronousQueue()
    )


class WarmupConditionPreparation:
    """Hold inputs through completion; retain the plan's stable graph consumers.

    The owner records input readiness before denoise, calls start after the
    two dense forward enqueues, waits before consuming a prepared row, and
    joins in its request finally block before template cleanup on every path.
    """

    def __init__(self, plan, bind, *, device, overlap=True):
        self.plan = plan
        self._bind = bind
        self.queue = _make_queue(plan, device, overlap)
        self.started = self.failed = self.waited = self.joined = False
        self.enqueue_wall_seconds = 0.0
        self.enqueued_forwards = self.launch_forwards = 0
        self.window_start = None

    def after_forward_enqueue(self, *, producer_start):
        if self.started:
            return
        if self.enqueued_forwards == 0:
            self.window_start = producer_start
        self.enqueued_forwards += 1
        if self.enqueued_forwards == 2:
            self.start(producer_start=self.window_start, enqueued_forwards=2)

    def start(self, *, producer_start, enqueued_forwards=1):
        if self.started or self.joined:
            raise RuntimeError("condition preparation has already started or joined")
        self.started = True
        self.launch_forwards = enqueued_forwards
        started = time.perf_counter()
        try:
            self.queue.launch(self._bind, producer_start=producer_start)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.enqueue_wall_seconds = time.perf_counter() - started
            self.queue.record_consumer(self.plan)

    def wait(self):
        if self.failed:
            raise RuntimeError("condition preparation failed")
        if not self.started:
            raise RuntimeError("condition preparation has not started")
        if not self.waited:
            self.queue.wait()
            self.waited = True

    def join(self):
        if not self.joined:
            try:
                if self.started:
                    self.queue.join()
            except BaseException:
                self.failed = True
                raise
            finally:
                self.joined = True
                self._bind = None

    def metadata(self, *, denoise_start):
        if not self.joined:
            raise RuntimeError(
                "join condition preparation before reading timing metadata"
            )
        timing = self.queue.metadata(denoise_start=denoise_start)
        graph = getattr(self.plan, "preparation_graph", None)
        graph_metadata = graph.metadata() if graph is not None else {}
        if graph_metadata.get("cold_capture_synchronizes"):
            timing["observed_interval_intersection_seconds"] = timing.get(
                "overlap_with_enqueued_window_seconds"
            )
            timing["overlap_with_enqueued_window_seconds"] = None
            timing["overlap_attribution"] = (
                "cold_capture_global_sync; raw_interval_intersection_is_not_work_overlap"
            )
        elif graph_metadata.get("backend") == "cuda_graph":
            timing["overlap_attribution"] = "warm_replay_event_interval_intersection"
        return {
            **timing,
            "launches": int(self.started),
            "consumer_waits": int(self.waited),
            "failed": self.failed,
            "enqueue_wall_seconds": self.enqueue_wall_seconds,
            "launch_scope": "after_dense_forward_enqueues_during_warmup",
            "enqueued_forwards_before_launch": self.launch_forwards,
            "device_scope": "inside_denoise_not_additive_to_denoise_time",
        }
