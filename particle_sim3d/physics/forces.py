"""
Physics solvers for N-body force calculations.

This module provides different backends for computing gravitational forces:
- Barnes-Hut (octree): O(N log N) approximation
- Direct CPU: O(N²) exact calculation on CPU
- Metal GPU: O(N²) parallel calculation on macOS GPU

Example:
    >>> from particle_sim3d.physics import compute_janus_forces
    >>> forces = compute_janus_forces(particles, params)
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particle_sim3d.physics.octree import OctreeNode
    from particle_sim3d.params import Sim3DParams

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None


def compute_janus_force_pair(
    si: int,
    sj: int,
    mi: float,
    mj: float,
    xi: float, yi: float, zi: float,
    xj: float, yj: float, zj: float,
    g: float,
    eps2: float,
) -> tuple[float, float, float]:
    """
    Compute the gravitational force on particle i due to particle j.
    
    Uses the Janus force law where same-sign particles attract
    and opposite-sign particles repel.
    
    Args:
        si, sj: Signs of particles i and j (+1 or -1)
        mi, mj: Masses of particles i and j (always positive)
        xi, yi, zi: Position of particle i
        xj, yj, zj: Position of particle j
        g: Gravitational constant
        eps2: Softening parameter squared (avoids singularities)
    
    Returns:
        (ax, ay, az): Acceleration on particle i due to particle j
    
    Note:
        The Janus force law is:
        a_i = G * s_i * q_j * (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)
        where q_j = s_j * m_j (signed charge)
    """
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    
    r2 = dx*dx + dy*dy + dz*dz + eps2
    inv_r = 1.0 / math.sqrt(r2)
    inv_r3 = inv_r * inv_r * inv_r
    
    # q_j = s_j * m_j (signed charge)
    qj = float(sj) * mj
    
    # Force factor: G * s_i * q_j / r³
    f = g * float(si) * qj * inv_r3
    
    return dx * f, dy * f, dz * f


def compute_forces_direct(
    xs: list[float],
    ys: list[float],
    zs: list[float],
    charges: list[float],
    signs: list[float],
    g: float,
    eps2: float,
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute forces using direct O(N²) summation (pure Python).
    
    This is the reference implementation for testing.
    
    Args:
        xs, ys, zs: Particle positions
        charges: Signed charges (s * m for each particle)
        signs: Particle signs (as floats: 1.0 or -1.0)
        g: Gravitational constant
        eps2: Softening parameter squared
    
    Returns:
        (ax_list, ay_list, az_list): Accelerations for all particles
    """
    n = len(xs)
    ax = [0.0] * n
    ay = [0.0] * n
    az = [0.0] * n
    
    for i in range(n):
        xi, yi, zi = xs[i], ys[i], zs[i]
        si = signs[i]
        axi = ayi = azi = 0.0
        
        for j in range(n):
            if i == j:
                continue
            dx = xs[j] - xi
            dy = ys[j] - yi
            dz = zs[j] - zi
            
            r2 = dx*dx + dy*dy + dz*dz + eps2
            inv_r = 1.0 / math.sqrt(r2)
            inv_r3 = inv_r * inv_r * inv_r
            
            f = g * si * charges[j] * inv_r3
            axi += dx * f
            ayi += dy * f
            azi += dz * f
        
        ax[i] = axi
        ay[i] = ayi
        az[i] = azi
    
    return ax, ay, az


class ForceSolver:
    """
    Abstract base interface for force solvers.
    
    Subclasses implement different algorithms for computing N-body forces.
    """
    
    def compute(
        self,
        xs: list[float],
        ys: list[float],
        zs: list[float],
        charges: list[float],
        signs: list[float],
        g: float,
        eps2: float,
        **kwargs,
    ) -> tuple[list[float], list[float], list[float]] | None:
        """
        Compute accelerations for all particles.
        
        Returns:
            (ax, ay, az) lists or None if computation failed.
        """
        raise NotImplementedError


class BarnesHutSolver(ForceSolver):
    """
    Barnes-Hut tree-based force solver.
    
    Approximates distant particle groups as single masses,
    achieving O(N log N) complexity.
    
    Attributes:
        theta: Opening angle parameter (0 = exact, higher = faster but less accurate)
    """
    
    def __init__(self, theta: float = 0.5):
        """
        Initialize the Barnes-Hut solver.
        
        Args:
            theta: Opening angle parameter. Typical values:
                   0.0 = exact (same as direct)
                   0.3 = accurate
                   0.5 = balanced
                   1.0 = fast but approximate
        """
        self.theta = theta
        self.last_build_time_ms: float | None = None
        self.last_traverse_time_ms: float | None = None
    
    def compute(
        self,
        xs: list[float],
        ys: list[float],
        zs: list[float],
        charges: list[float],
        signs: list[float],
        g: float,
        eps2: float,
        *,
        particle_signs: list[int] | None = None,
        cached_octree: object | None = None,  # OctreeNode
        **kwargs,
    ) -> tuple[list[float], list[float], list[float]] | tuple[list[float], list[float], list[float], object]:
        """
        Compute accelerations using Barnes-Hut algorithm.
        
        Args:
            xs, ys, zs: Particle positions
            charges: Signed charges (s * m)
            signs: Particle signs as floats
            g: Gravitational constant
            eps2: Softening squared
            particle_signs: Integer signs for each particle (required)
            cached_octree: Pre-built octree to reuse (optional, for performance)
        
        Returns:
            If cached_octree is None: (ax, ay, az) lists
            Otherwise: (ax, ay, az, new_octree) tuple with the built/reused octree
        """
        from particle_sim3d.physics.octree import build_octree
        
        n = len(xs)
        if n == 0:
            return [], [], []
        
        # Build or reuse tree
        if cached_octree is None:
            # Determine bounds
            space_half = max(
                1.0,
                max(abs(x) for x in xs),
                max(abs(y) for y in ys),
                max(abs(z) for z in zs),
            ) * 1.01
            
            # Build tree
            t0 = time.perf_counter()
            root = build_octree(xs, ys, zs, half=space_half)
            root.accumulate_charge(charges, xs, ys, zs)
            self.last_build_time_ms = (time.perf_counter() - t0) * 1000.0
            return_octree = root
        else:
            # Reuse cached octree (faster!)
            root = cached_octree
            # Still need to update charges as masses may have changed
            root.accumulate_charge(charges, xs, ys, zs)
            self.last_build_time_ms = 0.0  # No build time
            return_octree = root
        
        # Traverse tree
        t0 = time.perf_counter()
        ax = [0.0] * n
        ay = [0.0] * n
        az = [0.0] * n
        
        if particle_signs is None:
            # Derive from signs floats
            particle_signs = [1 if s > 0 else -1 for s in signs]
        
        for i in range(n):
            axi, ayi, azi = root.accel_on(
                idx=i,
                si=particle_signs[i],
                xi=xs[i],
                yi=ys[i],
                zi=zs[i],
                g=g,
                eps2=eps2,
                theta=self.theta,
                xs=xs,
                ys=ys,
                zs=zs,
                charges=charges,
            )
            ax[i] = axi
            ay[i] = ayi
            az[i] = azi
        
        self.last_traverse_time_ms = (time.perf_counter() - t0) * 1000.0
        
        # Return with octree if caller wants to cache it
        if cached_octree is not None or 'return_octree' in kwargs:
            return ax, ay, az, return_octree
        return ax, ay, az


class DirectCPUSolver(ForceSolver):
    """
    Direct O(N²) force solver using NumPy on CPU.
    
    More accurate than Barnes-Hut but slower for large N.
    Uses tiled computation to optimize cache usage.
    """
    
    def __init__(self, tile_size: int = 256):
        self.tile_size = tile_size
    
    def compute(
        self,
        xs: list[float],
        ys: list[float],
        zs: list[float],
        charges: list[float],
        signs: list[float],
        g: float,
        eps2: float,
        **kwargs,
    ) -> tuple[list[float], list[float], list[float]] | None:
        """Compute accelerations using tiled direct summation."""
        if np is None:
            return None
        
        n = len(xs)
        if n == 0:
            return [], [], []
        
        pos = np.empty((n, 3), dtype=np.float32)
        pos[:, 0] = xs
        pos[:, 1] = ys
        pos[:, 2] = zs
        charges_np = np.array(charges, dtype=np.float32)
        signs_np = np.array(signs, dtype=np.float32)
        
        acc = np.zeros((n, 3), dtype=np.float32)
        g32 = np.float32(g)
        eps2_32 = np.float32(eps2)
        tile = max(1, self.tile_size)
        
        for i0 in range(0, n, tile):
            i1 = min(n, i0 + tile)
            pi = pos[i0:i1]
            si = signs_np[i0:i1].reshape(-1, 1)
            acc_i = np.zeros((i1 - i0, 3), dtype=np.float32)
            
            for j0 in range(0, n, tile):
                j1 = min(n, j0 + tile)
                pj = pos[j0:j1]
                cj = charges_np[j0:j1].reshape(1, -1)
                
                d = pj[None, :, :] - pi[:, None, :]
                r2 = np.sum(d * d, axis=2, dtype=np.float32)
                r2 += eps2_32
                
                if i0 == j0:
                    diag = np.arange(min(i1 - i0, j1 - j0), dtype=np.int64)
                    r2[diag, diag] = np.inf
                
                inv_r = 1.0 / np.sqrt(r2)
                inv_r3 = inv_r * inv_r * inv_r
                f = g32 * si * (cj * inv_r3)
                acc_i += np.sum(d * f[:, :, None], axis=1, dtype=np.float32)
            
            acc[i0:i1] = acc_i
        
        return acc[:, 0].tolist(), acc[:, 1].tolist(), acc[:, 2].tolist()
