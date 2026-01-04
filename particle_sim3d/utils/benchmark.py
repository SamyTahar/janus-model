#!/usr/bin/env python3
"""
Performance benchmark for Janus 3D simulation.

Compares different force calculation backends:
- Barnes-Hut (CPU, O(N log N))
- Direct CPU (NumPy, O(N²))
- Metal GPU (macOS, O(N²))

Usage:
    python -m particle_sim3d.benchmark [--particles 1000] [--iterations 10]
"""

from __future__ import annotations

import argparse
import time
import sys
import math
import random
from typing import Callable

try:
    import numpy as np
except ImportError:
    np = None

from particle_sim3d.physics.forces import BarnesHutSolver, DirectCPUSolver

# Try Metal backend
try:
    from particle_sim3d.physics.metal_backend import MetalNBody
    METAL_AVAILABLE = True
except Exception:
    METAL_AVAILABLE = False


def generate_particles(n: int, seed: int = 42) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Generate random particle data."""
    rng = random.Random(seed)
    xs = [rng.uniform(-500, 500) for _ in range(n)]
    ys = [rng.uniform(-500, 500) for _ in range(n)]
    zs = [rng.uniform(-500, 500) for _ in range(n)]
    signs = [1.0 if rng.random() > 0.3 else -1.0 for _ in range(n)]
    charges = [s * rng.uniform(0.5, 1.5) for s in signs]
    return xs, ys, zs, charges, signs


def benchmark_barnes_hut(xs, ys, zs, charges, signs, g, eps2, theta=0.5, iterations=10) -> tuple[float, float]:
    """Benchmark Barnes-Hut solver."""
    solver = BarnesHutSolver(theta=theta)
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        solver.compute(xs, ys, zs, charges, signs, g, eps2)
        times.append(time.perf_counter() - t0)
    
    mean_ms = (sum(times) / len(times)) * 1000
    std_ms = (sum((t - mean_ms/1000)**2 for t in times) / len(times))**0.5 * 1000
    return mean_ms, std_ms


def benchmark_direct_cpu(xs, ys, zs, charges, signs, g, eps2, tile_size=256, iterations=10) -> tuple[float, float] | None:
    """Benchmark Direct CPU solver (NumPy)."""
    if np is None:
        return None
    
    solver = DirectCPUSolver(tile_size=tile_size)
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = solver.compute(xs, ys, zs, charges, signs, g, eps2)
        if result is None:
            return None
        times.append(time.perf_counter() - t0)
    
    mean_ms = (sum(times) / len(times)) * 1000
    std_ms = (sum((t - mean_ms/1000)**2 for t in times) / len(times))**0.5 * 1000
    return mean_ms, std_ms


def benchmark_metal(xs, ys, zs, charges, signs, g, eps2, tile_size=256, iterations=10) -> tuple[float, float] | None:
    """Benchmark Metal GPU solver."""
    if not METAL_AVAILABLE or np is None:
        return None
    
    try:
        metal = MetalNBody(tile_size=tile_size)
    except Exception as e:
        print(f"Metal init failed: {e}", file=sys.stderr)
        return None
    
    # Prepare numpy arrays
    n = len(xs)
    pos = np.empty((n, 4), dtype=np.float32)
    pos[:, 0] = xs
    pos[:, 1] = ys
    pos[:, 2] = zs
    pos[:, 3] = 0.0
    charges_np = np.array(charges, dtype=np.float32)
    signs_np = np.array(signs, dtype=np.float32)
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        metal.compute_accel(pos, charges_np, signs_np, g=g, eps2=eps2)
        times.append(time.perf_counter() - t0)
    
    mean_ms = (sum(times) / len(times)) * 1000
    std_ms = (sum((t - mean_ms/1000)**2 for t in times) / len(times))**0.5 * 1000
    return mean_ms, std_ms


def run_benchmark(n_particles: int, iterations: int) -> dict:
    """Run full benchmark suite."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_particles} particles, {iterations} iterations")
    print(f"{'='*60}")
    
    g = 1.0
    eps2 = 10.0  # softening²
    
    # Generate particles
    print("Generating particles...", end=" ", flush=True)
    xs, ys, zs, charges, signs = generate_particles(n_particles)
    print("done")
    
    results = {}
    
    # Barnes-Hut (theta=0.5)
    print("Barnes-Hut (θ=0.5)...", end=" ", flush=True)
    bh_time, bh_std = benchmark_barnes_hut(xs, ys, zs, charges, signs, g, eps2, theta=0.5, iterations=iterations)
    print(f"{bh_time:.2f} ± {bh_std:.2f} ms")
    results["barnes_hut_0.5"] = bh_time
    
    # Barnes-Hut (theta=0.7 - faster)
    print("Barnes-Hut (θ=0.7)...", end=" ", flush=True)
    bh7_time, bh7_std = benchmark_barnes_hut(xs, ys, zs, charges, signs, g, eps2, theta=0.7, iterations=iterations)
    print(f"{bh7_time:.2f} ± {bh7_std:.2f} ms")
    results["barnes_hut_0.7"] = bh7_time
    
    # Direct CPU
    print("Direct CPU (NumPy)...", end=" ", flush=True)
    cpu_result = benchmark_direct_cpu(xs, ys, zs, charges, signs, g, eps2, iterations=iterations)
    if cpu_result:
        cpu_time, cpu_std = cpu_result
        print(f"{cpu_time:.2f} ± {cpu_std:.2f} ms")
        results["direct_cpu"] = cpu_time
    else:
        print("not available")
    
    # Metal GPU
    print("Metal GPU...", end=" ", flush=True)
    metal_result = benchmark_metal(xs, ys, zs, charges, signs, g, eps2, iterations=iterations)
    if metal_result:
        metal_time, metal_std = metal_result
        print(f"{metal_time:.2f} ± {metal_std:.2f} ms")
        results["metal_gpu"] = metal_time
    else:
        print("not available")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    if "barnes_hut_0.5" in results:
        print(f"  Barnes-Hut (θ=0.5): {results['barnes_hut_0.5']:.2f} ms")
    if "barnes_hut_0.7" in results:
        print(f"  Barnes-Hut (θ=0.7): {results['barnes_hut_0.7']:.2f} ms")
    if "direct_cpu" in results:
        speedup = results["direct_cpu"] / results["barnes_hut_0.5"]
        print(f"  Direct CPU: {results['direct_cpu']:.2f} ms ({speedup:.1f}x slower than B-H)")
    if "metal_gpu" in results:
        speedup = results["direct_cpu"] / results["metal_gpu"] if "direct_cpu" in results else 1.0
        print(f"  Metal GPU: {results['metal_gpu']:.2f} ms ({speedup:.1f}x faster than CPU direct)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Janus 3D force calculations")
    parser.add_argument("--particles", "-n", type=int, default=1000, help="Number of particles")
    parser.add_argument("--iterations", "-i", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--sweep", action="store_true", help="Run sweep over particle counts")
    args = parser.parse_args()
    
    print("Janus 3D Performance Benchmark")
    print(f"Platform: {sys.platform}")
    print(f"NumPy: {'available' if np is not None else 'not available'}")
    print(f"Metal: {'available' if METAL_AVAILABLE else 'not available'}")
    
    if args.sweep:
        # Sweep over different particle counts
        counts = [100, 500, 1000, 2000, 5000]
        for n in counts:
            run_benchmark(n, args.iterations)
    else:
        run_benchmark(args.particles, args.iterations)


if __name__ == "__main__":
    main()
