"""Bounded per-group GPU staging and transfer timing for integration."""
from __future__ import annotations

from time import perf_counter

import numpy as np


class TileSession:
    """Cache weights and overlap one CUDA upload with the current reduction.

    Two persistent pinned/device slots bound the look-ahead memory. Other
    backends retain weight caching but use synchronous staging. The caller
    must finish a tile before reusing its slot, and close on every exit.
    """

    @classmethod
    def create(cls, weights, shape, log, *, allow_pipeline=True):
        from .torch_rejection import _get_torch, _device

        torch = _get_torch()
        dev = _device()
        if dev is None or dev.type == "cpu":
            return None
        return cls(torch, dev, weights, shape, log, allow_pipeline=allow_pipeline)

    def __init__(self, torch, dev, weights, shape, log, *, allow_pipeline=True):
        self.torch = torch
        self.dev = dev
        self.log = log
        self.weights = torch.from_numpy(np.asarray(weights, np.float32)).reshape(-1, 1, 1, 1).to(dev)
        self.pipelined = False
        self.host = []
        self.device = []
        self.events = []
        self.outputs = {}
        self.stream = None
        self.times = dict(upload=0.0, reduce=0.0, download=0.0)
        if dev.type == "cuda" and allow_pipeline:
            import psutil

            size = int(np.prod(shape)) * 4
            free, _ = torch.cuda.mem_get_info(dev)
            # Leave room for rejection temporaries as well as two input slots.
            if free > size * 14 + 64 * 1024**2 and psutil.virtual_memory().available > size * 4:
                try:
                    self.stream = torch.cuda.Stream(device=dev)
                    self.host = [torch.empty(shape, dtype=torch.float32, pin_memory=True) for _ in range(2)]
                    self.device = [torch.empty(shape, dtype=torch.float32, device=dev) for _ in range(2)]
                    self.events = [None, None]
                    self.pipelined = True
                except (RuntimeError, MemoryError) as exc:
                    self.close()
                    log(f"GPU upload look-ahead unavailable: {exc}; using one tile at a time.")
        log(f"GPU transfer mode: {'two-slot pinned CUDA pipeline' if self.pipelined else 'single tile'}, cached frame weights.")

    def prepare(self, array, slot=0):
        t0 = perf_counter()
        if not self.pipelined:
            result = self.torch.from_numpy(array).to(self.dev, dtype=self.torch.float32)
        else:
            if self.events[slot] is not None:
                self.events[slot].synchronize()
            slices = tuple(slice(0, n) for n in array.shape)
            host = self.host[slot][slices]
            np.copyto(host.numpy(), array)
            result = self.device[slot][slices]
            with self.torch.cuda.stream(self.stream):
                result.copy_(host, non_blocking=True)
                event = self.torch.cuda.Event()
                event.record(self.stream)
            self.events[slot] = event
        self.times['upload'] += perf_counter() - t0
        return result, slot

    def ready(self, prepared):
        tensor, slot = prepared
        if self.pipelined:
            self.torch.cuda.current_stream(self.dev).wait_event(self.events[slot])
        return tensor

    def download(self, result):
        t0 = perf_counter()
        out, maps = result
        tensors = (out, *maps) if isinstance(maps, tuple) else (out, maps)
        arrays = []
        for i, tensor in enumerate(tensors):
            if self.pipelined:
                key = (i, tuple(tensor.shape), tensor.dtype)
                host = self.outputs.get(key)
                if host is None:
                    host = self.torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                    self.outputs[key] = host
                host.copy_(tensor, non_blocking=True)
                arrays.append(host.numpy())
            else:
                arrays.append(tensor.cpu().numpy())
        if self.pipelined:
            event = self.torch.cuda.Event()
            event.record(self.torch.cuda.current_stream(self.dev))
            event.synchronize()
        self.times['download'] += perf_counter() - t0
        # Pinned outputs are reused next pass; consumers must accumulate/copy
        # them before the next download (the integration loop does so).
        return arrays[0], tuple(arrays[1:]) if isinstance(maps, tuple) else arrays[1]

    def reduce(self, reducer, array, weights, prepared, **kwargs):
        tensor = self.ready(prepared)
        start = end = None
        if self.dev.type == "cuda":
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record(self.torch.cuda.current_stream(self.dev))
        t0 = perf_counter()
        result = reducer(array, weights, device_tile=tensor,
                         device_weights=self.weights, return_device=True, **kwargs)
        if end is not None:
            end.record(self.torch.cuda.current_stream(self.dev))
        result = self.download(result)
        self.times['reduce'] += (start.elapsed_time(end) / 1000.0 if end is not None
                                 else perf_counter() - t0)
        return result

    def close(self):
        if self.stream is not None:
            self.stream.synchronize()
        self.host.clear()
        self.device.clear()
        self.events.clear()
        self.outputs.clear()
        self.stream = None
        self.pipelined = False
