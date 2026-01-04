"""
Barnes-Hut Octree implementation for N-body force calculation.

This module provides an octree data structure for approximating gravitational
forces in O(N log N) time instead of the naive O(N²) direct summation.

The algorithm works by:
1. Building a tree that recursively subdivides 3D space into octants
2. Computing the total charge and center of charge for each node
3. For each particle, traversing the tree and using the multipole
   approximation for distant nodes

Constants:
    MAX_DEPTH: Maximum tree depth to prevent infinite recursion
    BUCKET_CAPACITY: Maximum particles per leaf before splitting
    MIN_HALF: Minimum node size to prevent excessive subdivision
    MIN_ABS_CHARGE: Threshold for considering a node's charge as zero

Example:
    >>> from particle_sim3d.octree import build_octree
    >>> xs, ys, zs = [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    >>> root = build_octree(xs, ys, zs, half=10.0)
    >>> root.accumulate_charge([1.0, 1.0, -1.0], xs, ys, zs)
    >>> ax, ay, az = root.accel_on(idx=0, si=1, xi=0.0, yi=0.0, zi=0.0,
    ...                            g=1.0, eps2=0.01, theta=0.5,
    ...                            xs=xs, ys=ys, zs=zs, charges=[1.0, 1.0, -1.0])
"""

from __future__ import annotations

import math
from dataclasses import dataclass


MAX_DEPTH = 24
BUCKET_CAPACITY = 16
MIN_HALF = 1e-3
MIN_ABS_CHARGE = 1e-12


@dataclass(slots=True)
class OctreeNode:
    """
    A node in the Barnes-Hut octree.
    
    Each node represents a cubic region of space and can either be:
    - A leaf node containing particle indices
    - An internal node with 8 children (one per octant)
    
    Attributes:
        cx, cy, cz: Center coordinates of this node's region
        half: Half-size of the cubic region (full size = 2 * half)
        child: List of 8 child nodes if internal, None if leaf
        indices: List of particle indices if leaf, None if internal
        q: Total charge in this node (sum of all particle charges)
        qx, qy, qz: Charge-weighted position components (for center of charge)
    
    The center of charge is computed as: (qx/q, qy/q, qz/q)
    """
    cx: float
    cy: float
    cz: float
    half: float
    child: list["OctreeNode"] | None = None
    indices: list[int] | None = None

    q: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0

    def is_leaf(self) -> bool:
        """Return True if this is a leaf node (no children)."""
        return self.child is None

    def _octant(self, x: float, y: float, z: float) -> int:
        ox = 1 if x >= self.cx else 0
        oy = 1 if y >= self.cy else 0
        oz = 1 if z >= self.cz else 0
        return ox | (oy << 1) | (oz << 2)

    def _ensure_children(self) -> None:
        if self.child is not None:
            return
        h = self.half * 0.5
        children: list[OctreeNode] = []
        for o in range(8):
            dx = h if (o & 1) else -h
            dy = h if (o & 2) else -h
            dz = h if (o & 4) else -h
            children.append(OctreeNode(self.cx + dx, self.cy + dy, self.cz + dz, h))
        self.child = children

    def insert(self, idx: int, xs: list[float], ys: list[float], zs: list[float], *, depth: int = 0) -> None:
        # Barnes-Hut octree: stop splitting when max depth is reached (or node is too small)
        # and keep a small bucket of indices to avoid infinite recursion on coincident points.
        if self.child is None:
            if self.indices is None:
                self.indices = [idx]
                return

            if depth >= MAX_DEPTH or self.half <= MIN_HALF or len(self.indices) < BUCKET_CAPACITY:
                self.indices.append(idx)
                return

            old = self.indices
            self.indices = None
            self._ensure_children()
            for j in old:
                o = self._octant(xs[j], ys[j], zs[j])
                assert self.child is not None
                self.child[o].insert(j, xs, ys, zs, depth=depth + 1)

        o = self._octant(xs[idx], ys[idx], zs[idx])
        assert self.child is not None
        self.child[o].insert(idx, xs, ys, zs, depth=depth + 1)

    def accumulate_charge(self, charges: list[float], xs: list[float], ys: list[float], zs: list[float]) -> None:
        if self.child is None:
            if not self.indices:
                self.q = 0.0
                self.qx = self.qy = self.qz = 0.0
                return
            q = 0.0
            qx = qy = qz = 0.0
            for i in self.indices:
                qi = float(charges[i])
                q += qi
                qx += xs[i] * qi
                qy += ys[i] * qi
                qz += zs[i] * qi
            self.q = q
            self.qx = qx
            self.qy = qy
            self.qz = qz
            return

        q = 0.0
        qx = qy = qz = 0.0
        assert self.child is not None
        for ch in self.child:
            ch.accumulate_charge(charges, xs, ys, zs)
            q += ch.q
            qx += ch.qx
            qy += ch.qy
            qz += ch.qz
        self.q = q
        self.qx = qx
        self.qy = qy
        self.qz = qz

    def approx_center(self) -> tuple[float, float, float] | None:
        if abs(self.q) < MIN_ABS_CHARGE:
            return None
        inv = 1.0 / float(self.q)
        return self.qx * inv, self.qy * inv, self.qz * inv

    def accel_on(
        self,
        *,
        idx: int,
        si: int,
        xi: float,
        yi: float,
        zi: float,
        g: float,
        eps2: float,
        theta: float,
        xs: list[float],
        ys: list[float],
        zs: list[float],
        charges: list[float],
    ) -> tuple[float, float, float]:
        if self.child is None:
            if not self.indices:
                return 0.0, 0.0, 0.0
            ax = ay = az = 0.0
            for j in self.indices:
                if j == idx:
                    continue
                dx = xs[j] - xi
                dy = ys[j] - yi
                dz = zs[j] - zi
                r2 = (dx * dx) + (dy * dy) + (dz * dz) + eps2
                inv_r = 1.0 / math.sqrt(r2)
                inv_r3 = inv_r * inv_r * inv_r
                f = g * float(si) * float(charges[j]) * inv_r3
                ax += dx * f
                ay += dy * f
                az += dz * f
            return ax, ay, az

        center = self.approx_center()
        if center is not None:
            cx, cy, cz = center
            dx = cx - xi
            dy = cy - yi
            dz = cz - zi
            d2 = (dx * dx) + (dy * dy) + (dz * dz)
            d = math.sqrt(d2) if d2 > 0.0 else 0.0
            size = self.half * 2.0
            if d > 0.0 and (size / d) < theta:
                r2 = d2 + eps2
                inv_r = 1.0 / math.sqrt(r2)
                inv_r3 = inv_r * inv_r * inv_r
                # Approximation monopole: a_i += G * s_i * Q_cell * (r_cell - r_i) / r^3
                f = g * float(si) * float(self.q) * inv_r3
                return dx * f, dy * f, dz * f

        ax = ay = az = 0.0
        assert self.child is not None
        for ch in self.child:
            cx, cy, cz = ch.accel_on(
                idx=idx,
                si=si,
                xi=xi,
                yi=yi,
                zi=zi,
                g=g,
                eps2=eps2,
                theta=theta,
                xs=xs,
                ys=ys,
                zs=zs,
                charges=charges,
            )
            ax += cx
            ay += cy
            az += cz
        return ax, ay, az


def build_octree(xs: list[float], ys: list[float], zs: list[float], half: float) -> OctreeNode:
    root = OctreeNode(0.0, 0.0, 0.0, half)
    for i in range(len(xs)):
        root.insert(i, xs, ys, zs, depth=0)
    return root
