"""Synthetic integration benchmark; never installs or changes GPU packages.

Example: python ops/benchmark_stacking_gpu.py --backend directml --baseline-ref dev
Run in SAS Pro's Python environment or expose its runtime site-packages first.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'directml'), default='directml')
    parser.add_argument('--frames', type=int, default=128)
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--baseline-ref')
    parser.add_argument('--algorithm', default='Weighted Windsorized Sigma Clipping')
    args = parser.parse_args()
    if min(args.frames, args.rows, args.width, args.repeat) < 1:
        parser.error('Dimensions and repeat must be positive')
    from setiastro.saspro.runtime_torch import add_runtime_to_sys_path
    add_runtime_to_sys_path(status_cb=lambda *_: None)
    import torch
    from setiastro.saspro import torch_rejection as tr
    from setiastro.saspro.stacking_gpu_pipeline import TileSession

    if args.backend == 'directml':
        import torch_directml
        dev = torch_directml.device()
        print(torch_directml.device_name(0).rstrip('\0'))
    else:
        dev = torch.device(args.backend)
    tr._TORCH, tr._DEVICE = torch, dev
    torch.set_num_threads(min(8, torch.get_num_threads()))
    rng = np.random.default_rng(73)
    stack = rng.normal(.5, .04, (args.frames, args.rows, args.width, 3)).astype(np.float32)
    stack[0, 0, 0] = 5.
    stack[:, 1 % args.rows, 1 % args.width] = 0.
    weights = np.linspace(.5, 1.5, args.frames, dtype=np.float32)
    kw = dict(algo_name=args.algorithm, reduce_rej_maps=True)

    def measure(name, fn):
        result = fn()  # warm-up, excluded
        samples = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            result = fn()  # includes the final host download/synchronization
            samples.append(time.perf_counter() - start)
        print(f'{name}: median={statistics.median(samples):.4f}s, samples={samples}')
        return result, statistics.median(samples)

    tr._DEVICE = torch.device('cpu')
    reference = tr.torch_reduce_tile(stack, weights, **kw)
    tr._DEVICE = dev
    baseline_seconds = None
    if args.baseline_ref:
        source = subprocess.check_output(['git', 'show',
            f'{args.baseline_ref}:src/setiastro/saspro/torch_rejection.py'], cwd=ROOT, text=True, encoding='utf-8')
        baseline = types.ModuleType('setiastro.saspro._benchmark_baseline')
        baseline.__package__ = 'setiastro.saspro'
        exec(compile(source, '<baseline torch_rejection>', 'exec'), baseline.__dict__)
        baseline._TORCH, baseline._DEVICE = torch, dev
        old, baseline_seconds = measure('baseline', lambda: baseline.torch_reduce_tile(stack, weights, **kw))
        print('baseline/reference max difference:', np.nanmax(np.abs(old[0] - reference[0])))
    session = TileSession.create(weights, stack.shape, print)
    try:
        def optimized():
            if session is None:
                return tr.torch_reduce_tile(stack, weights, **kw)
            return session.reduce(tr.torch_reduce_tile, stack, weights, session.prepare(stack), **kw)
        actual, seconds = measure('optimized', optimized)
        np.testing.assert_allclose(actual[0], reference[0], rtol=2e-5, atol=2e-5, equal_nan=True)
        for a, b in zip(actual[1], reference[1]):
            np.testing.assert_array_equal(a, b)
        print('CPU reference parity: passed')
        if baseline_seconds is not None:
            print(f'Synthetic reducer speedup: {baseline_seconds / seconds:.2f}x')
        print('This excludes disk reads, calibration, registration and drizzle.')
    finally:
        if session is not None:
            session.close()


if __name__ == '__main__':
    main()
