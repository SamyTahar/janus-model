from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

from particle_sim3d.physics.octree import build_octree
from particle_sim3d.params import Sim3DParams
from particle_sim3d.physics.forces import BarnesHutSolver, DirectCPUSolver
from particle_sim3d.core.init_conditions import create_random_distribution, create_janus_galaxy, InitialParticle


@dataclass(slots=True)
class Particle3D:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    s: int
    m: float
    blob: bool = False


class ParticleSim3D:
    def __init__(self, params: Sim3DParams) -> None:
        self.params = params
        self._rng = random.Random(params.seed)
        self.particles: list[Particle3D] = []
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._zs: list[float] = []
        self._charges: list[float] = []
        self._signs: list[float] = []
        self._ax: list[float] = []
        self._ay: list[float] = []
        self._az: list[float] = []
        self.accel_mag: list[float] = []
        self._metal = None
        self._metal_error: str | None = None
        self._metal_tile_size: int | None = None
        self._pos_np = None
        self._pos4_np = None
        self._charges_np = None
        self._signs_np = None
        self.last_force_ms: float | None = None
        self.last_force_backend: str = "off"
        
        # Force solvers (initialized lazily based on params)
        self._barnes_hut_solver: BarnesHutSolver | None = None
        self._direct_cpu_solver: DirectCPUSolver | None = None
        
        # Octree cache for performance optimization
        self._octree_cache: object | None = None  # OctreeNode
        self._octree_valid: bool = False
        self._octree_positions: list[tuple[float, float, float]] | None = None
        
        self.reset()

    def reset(self) -> None:
        p = self.params
        self._rng = random.Random(p.seed)
        self.particles = []
        
        # Invalidate octree cache
        self._octree_valid = False
        self._octree_cache = None
        self._octree_positions = None
        
        # Get population counts
        n_pos, n_neg = self._population_counts()
        
        # Generate initial conditions using init_conditions module
        if p.init_mode == "random":
            initial_particles = create_random_distribution(p, self._rng, n_pos, n_neg)
        else:
            initial_particles = create_janus_galaxy(p, self._rng, n_pos, n_neg)
        
        # Convert InitialParticle to Particle3D
        for ip in initial_particles:
            self.particles.append(Particle3D(
                x=ip.x, y=ip.y, z=ip.z,
                vx=ip.vx, vy=ip.vy, vz=ip.vz,
                s=ip.s, m=ip.m,
            ))

    def _population_counts(self) -> tuple[int, int]:
        p = self.params
        pos = int(p.positive_count)
        neg = int(p.negative_count)
        mode = str(getattr(p, "population_mode", "total")).strip().lower()

        def counts_from_total() -> tuple[int, int]:
            n_total = int(p.particle_count)
            n_neg = int(round(n_total * float(p.negative_fraction)))
            if n_total > 1:
                n_neg = max(1, min(n_total - 1, n_neg))
            else:
                n_neg = max(0, min(n_total, n_neg))
            n_pos = n_total - n_neg
            return n_pos, n_neg

        if mode == "explicit":
            if pos > 0 and neg > 0:
                return pos, neg
            return counts_from_total()
        if mode == "total":
            return counts_from_total()
        if pos > 0 and neg > 0:
            return pos, neg
        return counts_from_total()

    def _reset_random(self) -> None:
        p = self.params
        m_pos = float(p.mass_positive)
        m_neg = float(p.mass_negative)
        n_pos, n_neg = self._population_counts()
        total = n_pos + n_neg
        signs = [1] * n_pos + [-1] * n_neg
        self._rng.shuffle(signs)
        if p.bound_mode == "box":
            b = float(p.bounds)
            for i in range(total):
                x = self._rng.uniform(-b, b)
                y = self._rng.uniform(-b, b)
                z = self._rng.uniform(-b, b)

                u = self._rng.random()
                v = self._rng.random()
                theta = 2.0 * math.pi * u
                phi = math.acos(2.0 * v - 1.0)
                speed = self._rng.random() * p.initial_speed
                vx = math.cos(theta) * math.sin(phi) * speed
                vy = math.sin(theta) * math.sin(phi) * speed
                vz = math.cos(phi) * speed

                s = signs[i]
                m = m_neg if s < 0 else m_pos
                self.particles.append(Particle3D(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, s=s, m=m))
            return

        r = float(p.bound_sphere_radius)
        inv_fz = 1.0 / max(1e-9, float(p.bound_sphere_flatten_z))
        zmax = r / inv_fz
        r2_max = r * r

        for i in range(total):
            # rejection sampling inside the (possibly flattened) sphere:
            # x^2 + y^2 + (z^2)/(fz^2) <= R^2  <=> x^2 + y^2 + (z*inv_fz)^2 <= R^2
            while True:
                x = self._rng.uniform(-r, r)
                y = self._rng.uniform(-r, r)
                z = self._rng.uniform(-zmax, zmax)
                if (x * x) + (y * y) + ((z * inv_fz) * (z * inv_fz)) <= r2_max:
                    break

            u = self._rng.random()
            v = self._rng.random()
            theta = 2.0 * math.pi * u
            phi = math.acos(2.0 * v - 1.0)
            speed = self._rng.random() * p.initial_speed
            vx = math.cos(theta) * math.sin(phi) * speed
            vy = math.sin(theta) * math.sin(phi) * speed
            vz = math.cos(phi) * speed

            s = signs[i]
            m = m_neg if s < 0 else m_pos
            self.particles.append(Particle3D(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, s=s, m=m))

    def _reset_janus_galaxy(self) -> None:
        p = self.params
        bound_mode = str(p.bound_mode)
        domain = float(p.bounds) if bound_mode == "box" else float(p.bound_sphere_radius)
        fz = max(1e-9, float(p.bound_sphere_flatten_z))
        m_pos = float(p.mass_positive)
        m_neg = float(p.mass_negative)

        n_pos, n_neg = self._population_counts()
        static_shell = bool(p.bound_mode == "sphere" and p.negative_on_boundary and p.negative_static_on_boundary)
        neg_vphi_scale = float(p.negative_vphi_scale)

        def negative_rotation_velocity(x: float, y: float) -> tuple[float, float]:
            if abs(neg_vphi_scale) <= 1e-9:
                return 0.0, 0.0
            r = math.hypot(x, y)
            if r <= 1e-6:
                return 0.0, 0.0
            vphi = p.galaxy_vmax * (1.0 - math.exp(-r / p.galaxy_turnover)) * neg_vphi_scale
            tx = -y / r
            ty = x / r
            return tx * vphi, ty * vphi

        # M+ galaxy: exponential disk in XY with small thickness in Z, inside a void.
        for _ in range(n_pos):
            r = self._sample_exponential_radius(scale=p.galaxy_scale_length, r_max=p.galaxy_radius)
            a = self._rng.random() * (2.0 * math.pi)
            x = r * math.cos(a)
            y = r * math.sin(a)
            z = self._rng.gauss(0.0, p.galaxy_thickness)

            if r > 1e-6:
                vphi = p.galaxy_vmax * (1.0 - math.exp(-r / p.galaxy_turnover))
                tx = -y / r
                ty = x / r
                vx = (tx * vphi) + self._rng.gauss(0.0, p.galaxy_sigma_v)
                vy = (ty * vphi) + self._rng.gauss(0.0, p.galaxy_sigma_v)
            else:
                vx = self._rng.gauss(0.0, p.galaxy_sigma_v)
                vy = self._rng.gauss(0.0, p.galaxy_sigma_v)
            vz = self._rng.gauss(0.0, p.galaxy_sigma_v * 0.5)

            self.particles.append(Particle3D(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, s=1, m=m_pos))

        # M- environment: clumpy distribution outside the void, acting as a potential barrier.
        clump_centers: list[tuple[float, float, float]] = []
        outer = domain * 0.95
        inner = max(p.void_radius + (2.0 * p.negative_clump_sigma), p.void_radius + 5.0)
        # If the shell is too thin (or void too large), fall back to something valid.
        if inner >= outer:
            inner = max(p.void_radius, outer * 0.7)
        if inner >= outer:
            inner = outer * 0.5

        if p.bound_mode == "sphere" and p.negative_on_boundary:
            r = float(p.bound_sphere_radius)
            fz = max(1e-9, float(p.bound_sphere_flatten_z))
            for _ in range(n_neg):
                dx, dy, dz = self._sample_unit_vector()
                x = r * dx
                y = r * dy
                z = r * fz * dz
                if static_shell:
                    vx = vy = vz = 0.0
                else:
                    vx, vy = negative_rotation_velocity(x, y)
                    vx += self._rng.gauss(0.0, p.negative_sigma_v)
                    vy += self._rng.gauss(0.0, p.negative_sigma_v)
                    vz = self._rng.gauss(0.0, p.negative_sigma_v)
                self.particles.append(Particle3D(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, s=-1, m=m_neg))
            return

        for _ in range(int(p.negative_clump_count)):
            # biased toward the inner radius to build a "barrier" around the void
            u = self._rng.random()
            rr = inner + (outer - inner) * (u * u)
            dx, dy, dz = self._sample_unit_vector()
            if bound_mode == "sphere":
                # sample clump center on the oblate/prolate spheroid in "scaled space"
                # (x, y, z/fz) has spherical radius rr.
                clump_centers.append((dx * rr, dy * rr, dz * rr * fz))
            else:
                clump_centers.append((dx * rr, dy * rr, dz * rr))

        for _ in range(n_neg):
            cx, cy, cz = clump_centers[self._rng.randrange(len(clump_centers))]
            x = cx + self._rng.gauss(0.0, p.negative_clump_sigma)
            y = cy + self._rng.gauss(0.0, p.negative_clump_sigma)
            z = cz + self._rng.gauss(0.0, p.negative_clump_sigma)

            # enforce the void: push particles outside void_radius
            rr = math.sqrt((x * x) + (y * y) + (z * z))
            if rr < p.void_radius:
                dx, dy, dz = self._sample_unit_vector()
                rr2 = p.void_radius + abs(self._rng.gauss(0.0, p.negative_clump_sigma))
                x = dx * rr2
                y = dy * rr2
                z = dz * rr2

            if bound_mode == "sphere":
                x, y, z = self._project_into_spheroid(x, y, z)
            else:
                x = max(-domain, min(domain, x))
                y = max(-domain, min(domain, y))
                z = max(-domain, min(domain, z))

            vx, vy = negative_rotation_velocity(x, y)
            vx += self._rng.gauss(0.0, p.negative_sigma_v)
            vy += self._rng.gauss(0.0, p.negative_sigma_v)
            vz = self._rng.gauss(0.0, p.negative_sigma_v)

            self.particles.append(Particle3D(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, s=-1, m=m_neg))

    def _sample_unit_vector(self) -> tuple[float, float, float]:
        u = self._rng.random()
        v = self._rng.random()
        theta = 2.0 * math.pi * u
        phi = math.acos(2.0 * v - 1.0)
        return (
            math.cos(theta) * math.sin(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(phi),
        )

    def _sample_exponential_radius(self, *, scale: float, r_max: float) -> float:
        # Exponential disk: p(r) ~ r * exp(-r/scale) (Gamma(k=2, theta=scale)), truncated at r_max.
        scale = max(1e-6, float(scale))
        r_max = max(1e-6, float(r_max))
        while True:
            u1 = max(1e-12, self._rng.random())
            u2 = max(1e-12, self._rng.random())
            r = -scale * math.log(u1 * u2)
            if r <= r_max:
                return r

    def _project_into_spheroid(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        p = self.params
        r = float(p.bound_sphere_radius)
        inv_fz = 1.0 / max(1e-9, float(p.bound_sphere_flatten_z))
        r2 = (x * x) + (y * y) + ((z * inv_fz) * (z * inv_fz))
        r2_max = r * r
        if r2 <= r2_max:
            return x, y, z
        rr = math.sqrt(r2)
        if rr <= 0.0:
            return 0.0, 0.0, 0.0
        t = r / rr
        return x * t, y * t, z * t

    def _ensure_metal(self, *, tile_size: int) -> None:
        if self._metal is not None and self._metal_tile_size == tile_size:
            return
        if self._metal_error is not None and self._metal_tile_size == tile_size:
            return
        self._metal = None
        self._metal_error = None
        if np is None:
            self._metal_error = "numpy not available"
            print("[metal] numpy not available, falling back to CPU.", file=sys.stderr)
            return
        try:
            from particle_sim3d.physics.metal_backend import MetalNBody

            self._metal = MetalNBody(tile_size=tile_size)
            self._metal_tile_size = tile_size
        except Exception as exc:  # pragma: no cover - depends on platform
            self._metal_error = str(exc)
            print(f"[metal] init failed, falling back to CPU: {exc}", file=sys.stderr)

    def _accel_metal(
        self,
        *,
        xs: list[float],
        ys: list[float],
        zs: list[float],
        charges: list[float],
        signs: list[float],
        g: float,
        eps2: float,
        debug: bool = False,
    ):
        if self._metal is None or np is None:
            return None
        n = len(xs)
        if n == 0:
            return None
        if self._pos_np is None or self._pos_np.shape[0] != n:
            self._pos_np = np.empty((n, 3), dtype=np.float32)
            self._pos4_np = None
            self._charges_np = np.empty(n, dtype=np.float32)
            self._signs_np = np.empty(n, dtype=np.float32)
        if self._pos4_np is None or self._pos4_np.shape[0] != n:
            self._pos4_np = np.empty((n, 4), dtype=np.float32)
        pos = self._pos_np
        pos4 = self._pos4_np
        charges_np = self._charges_np
        signs_np = self._signs_np
        if pos is None or pos4 is None or charges_np is None or signs_np is None:
            return None
        pos[:, 0] = xs
        pos[:, 1] = ys
        pos[:, 2] = zs
        pos4[:, :3] = pos
        pos4[:, 3] = 0.0
        charges_np[:] = charges
        signs_np[:] = signs
        try:
            accel = self._metal.compute_accel(
                pos4, charges_np, signs_np, g=g, eps2=eps2, debug=debug
            )
            if np is not None and not np.isfinite(accel).all():
                if debug:
                    self._debug_force_output(
                        backend="metal",
                        positions=pos4,
                        charges=charges_np,
                        signs=signs_np,
                        accel=accel,
                        g=g,
                        eps2=eps2,
                    )
                raise ValueError("non-finite accel from Metal")
            return accel
        except Exception as exc:  # pragma: no cover - backend failure
            self._metal_error = str(exc)
            self._metal = None
            print(f"[metal] compute failed, falling back to CPU: {exc}", file=sys.stderr)
            if debug and np is not None:
                if not np.isfinite(pos4).all():
                    print("[metal][debug] non-finite input positions detected.", file=sys.stderr)
                if not np.isfinite(charges_np).all():
                    print("[metal][debug] non-finite input charges detected.", file=sys.stderr)
                if not np.isfinite(signs_np).all():
                    print("[metal][debug] non-finite input signs detected.", file=sys.stderr)
            return None

    def _accel_cpu_direct(
        self,
        *,
        xs: list[float],
        ys: list[float],
        zs: list[float],
        charges: list[float],
        signs: list[float],
        g: float,
        eps2: float,
        tile_size: int = 256,
    ):
        if np is None:
            print("[cpu_direct] numpy not available, falling back to CPU.", file=sys.stderr)
            return None
        n = len(xs)
        if n == 0:
            return None
        if self._pos_np is None or self._pos_np.shape[0] != n:
            self._pos_np = np.empty((n, 3), dtype=np.float32)
            self._charges_np = np.empty(n, dtype=np.float32)
            self._signs_np = np.empty(n, dtype=np.float32)
        pos = self._pos_np
        charges_np = self._charges_np
        signs_np = self._signs_np
        if pos is None or charges_np is None or signs_np is None:
            return None

        pos[:, 0] = xs
        pos[:, 1] = ys
        pos[:, 2] = zs
        charges_np[:] = charges
        signs_np[:] = signs

        acc = np.zeros((n, 3), dtype=np.float32)
        g32 = np.float32(g)
        eps2_32 = np.float32(eps2)
        tile = max(1, int(tile_size))

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

        return acc

    def _debug_force_output(
        self,
        *,
        backend: str,
        positions: "np.ndarray",
        charges: "np.ndarray",
        signs: "np.ndarray",
        accel: "np.ndarray",
        g: float,
        eps2: float,
    ) -> None:
        if np is None:
            return
        n = int(positions.shape[0])
        finite_acc = np.isfinite(accel)
        bad_mask = ~finite_acc
        bad_any = np.any(bad_mask, axis=1)
        bad_idx = np.where(bad_any)[0]

        print(f"[{backend}][debug] n={n} g={g:.6g} eps2={eps2:.6g}", file=sys.stderr)
        print(
            f"[{backend}][debug] pos min/max={positions[:, :3].min():.6g}/{positions[:, :3].max():.6g}",
            file=sys.stderr,
        )
        print(
            f"[{backend}][debug] charge min/max={charges.min():.6g}/{charges.max():.6g}",
            file=sys.stderr,
        )
        print(
            f"[{backend}][debug] sign min/max={signs.min():.6g}/{signs.max():.6g}",
            file=sys.stderr,
        )
        print(
            f"[{backend}][debug] accel non-finite count={bad_idx.size}",
            file=sys.stderr,
        )

        if bad_idx.size:
            sample = bad_idx[:8]
            for idx in sample:
                ax, ay, az = accel[idx]
                px, py, pz = positions[idx, :3]
                print(
                    f"[{backend}][debug] idx={int(idx)} pos=({px:.6g},{py:.6g},{pz:.6g}) "
                    f"acc=({ax},{ay},{az})",
                    file=sys.stderr,
                )
            # Compare against CPU direct for a small subset (float32 vs float64).
            for idx in sample[:3]:
                acc32, min_r2_32, max_inv_r3_32 = self._debug_cpu_direct_for_idx(
                    positions=positions[:, :3],
                    charges=charges,
                    signs=signs,
                    idx=int(idx),
                    g=g,
                    eps2=eps2,
                    dtype=np.float32,
                )
                acc64, min_r2_64, max_inv_r3_64 = self._debug_cpu_direct_for_idx(
                    positions=positions[:, :3],
                    charges=charges,
                    signs=signs,
                    idx=int(idx),
                    g=g,
                    eps2=eps2,
                    dtype=np.float64,
                )
                print(
                    f"[{backend}][debug] cpu32 idx={int(idx)} acc=({acc32[0]:.6g},{acc32[1]:.6g},{acc32[2]:.6g}) "
                    f"min_r2={min_r2_32:.6g} max_inv_r3={max_inv_r3_32:.6g}",
                    file=sys.stderr,
                )
                print(
                    f"[{backend}][debug] cpu64 idx={int(idx)} acc=({acc64[0]:.6g},{acc64[1]:.6g},{acc64[2]:.6g}) "
                    f"min_r2={min_r2_64:.6g} max_inv_r3={max_inv_r3_64:.6g}",
                    file=sys.stderr,
                )

    def _debug_cpu_direct_for_idx(
        self,
        *,
        positions: "np.ndarray",
        charges: "np.ndarray",
        signs: "np.ndarray",
        idx: int,
        g: float,
        eps2: float,
        dtype: "type[np.floating]",
    ) -> tuple["np.ndarray", float, float]:
        if np is None:
            return np.zeros(3, dtype=np.float32), 0.0, 0.0
        pos = positions.astype(dtype, copy=False)
        chg = charges.astype(dtype, copy=False)
        sgn = signs.astype(dtype, copy=False)
        pi = pos[idx]
        d = pos - pi
        r2 = np.sum(d * d, axis=1, dtype=dtype)
        r2 = r2 + dtype(eps2)
        r2[idx] = np.inf
        r2 = np.maximum(r2, dtype(1.0e-12))
        inv_r = np.reciprocal(np.sqrt(r2, dtype=dtype), dtype=dtype)
        inv_r3 = inv_r * inv_r * inv_r
        f = dtype(g) * sgn[idx] * chg * inv_r3
        acc = np.sum(d * f[:, None], axis=0, dtype=dtype)
        min_r2 = float(np.min(r2))
        max_inv_r3 = float(np.max(inv_r3))
        return acc.astype(np.float32), min_r2, max_inv_r3

    def _apply_bounds(self, pt: Particle3D, *, bounce: float) -> None:
        p = self.params
        if p.bound_mode == "box":
            if not p.bounds_enabled:
                return
            b = float(p.bounds)
            if pt.x < -b:
                pt.x = -b
                pt.vx = -pt.vx * bounce
            elif pt.x > b:
                pt.x = b
                pt.vx = -pt.vx * bounce
            if pt.y < -b:
                pt.y = -b
                pt.vy = -pt.vy * bounce
            elif pt.y > b:
                pt.y = b
                pt.vy = -pt.vy * bounce
            if pt.z < -b:
                pt.z = -b
                pt.vz = -pt.vz * bounce
            elif pt.z > b:
                pt.z = b
                pt.vz = -pt.vz * bounce
            return

        if not p.bounds_enabled:
            return

        r = float(p.bound_sphere_radius)
        inv_fz = 1.0 / max(1e-9, float(p.bound_sphere_flatten_z))
        r2 = (pt.x * pt.x) + (pt.y * pt.y) + ((pt.z * inv_fz) * (pt.z * inv_fz))
        r2_max = r * r
        if r2 <= r2_max:
            return

        rr = math.sqrt(r2)
        if rr <= 0.0:
            return

        # Project back to the spheroid surface along the radial direction in scaled space.
        t = r / rr
        pt.x *= t
        pt.y *= t
        pt.z *= t

        # Compute a (normalized) surface normal from the implicit function:
        # f(x,y,z) = x^2 + y^2 + (z^2)/(fz^2) - R^2
        # grad(f) = (2x, 2y, 2z/(fz^2))  (we can ignore the factor 2)
        inv_fz2 = inv_fz * inv_fz
        nx = pt.x
        ny = pt.y
        nz = pt.z * inv_fz2
        nlen2 = (nx * nx) + (ny * ny) + (nz * nz)
        if nlen2 <= 0.0:
            return
        inv_n = 1.0 / math.sqrt(nlen2)
        nx *= inv_n
        ny *= inv_n
        nz *= inv_n

        vdot = (pt.vx * nx) + (pt.vy * ny) + (pt.vz * nz)
        # restitution only on the normal component: v' = v - (1 + e) * (v·n) n
        k = (1.0 + bounce) * vdot
        pt.vx -= k * nx
        pt.vy -= k * ny
        pt.vz -= k * nz

    def enforce_bounds(self) -> None:
        if not self.params.bounds_enabled:
            return
        bounce = float(self.params.bounce)
        for pt in self.particles:
            self._apply_bounds(pt, bounce=bounce)

    def update_masses(self) -> None:
        """Update all particle masses from current params."""
        p = self.params
        m_pos = float(p.mass_positive)
        m_neg = float(p.mass_negative)
        for pt in self.particles:
            pt.m = m_neg if pt.s < 0 else m_pos

    def reseed_species(self) -> None:
        p = self.params
        m_pos = float(p.mass_positive)
        m_neg = float(p.mass_negative)
        if p.init_mode == "random":
            n_pos, n_neg = self._population_counts()
            total = n_pos + n_neg
            if total != len(self.particles):
                self.reset()
                return
            signs = [1] * n_pos + [-1] * n_neg
            self._rng.shuffle(signs)
            for i, pt in enumerate(self.particles):
                pt.s = signs[i]
                pt.m = m_neg if pt.s < 0 else m_pos
            return
        # structured init depends on the two populations
        self.reset()

    def resize_particles(self, new_count: int) -> None:
        p = self.params
        m_pos = float(p.mass_positive)
        m_neg = float(p.mass_negative)
        new_count = max(1, int(new_count))
        if new_count == len(self.particles):
            return
        # For the structured Janus init, counts affect the whole setup, so rebuild.
        if p.init_mode != "random":
            self.params.particle_count = new_count
            self.params.clamp()
            self.reset()
            return

        if new_count < len(self.particles):
            self.particles = self.particles[:new_count]
            return
        for _ in range(new_count - len(self.particles)):
            if p.bound_mode == "box":
                b = float(p.bounds)
                x = self._rng.uniform(-b, b)
                y = self._rng.uniform(-b, b)
                z = self._rng.uniform(-b, b)
            else:
                r = float(p.bound_sphere_radius)
                inv_fz = 1.0 / max(1e-9, float(p.bound_sphere_flatten_z))
                zmax = r / inv_fz
                r2_max = r * r
                while True:
                    x = self._rng.uniform(-r, r)
                    y = self._rng.uniform(-r, r)
                    z = self._rng.uniform(-zmax, zmax)
                    if (x * x) + (y * y) + ((z * inv_fz) * (z * inv_fz)) <= r2_max:
                        break
            u = self._rng.random()
            v = self._rng.random()
            theta = 2.0 * math.pi * u
            phi = math.acos(2.0 * v - 1.0)
            speed = self._rng.random() * p.initial_speed
            vx = math.cos(theta) * math.sin(phi) * speed
            vy = math.sin(theta) * math.sin(phi) * speed
            vz = math.cos(phi) * speed
            s = -1 if self._rng.random() < p.negative_fraction else 1
            m = m_neg if s < 0 else m_pos
            self.particles.append(Particle3D(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, s=s, m=m))

    def counts(self) -> tuple[int, int]:
        n_pos = sum(1 for pt in self.particles if pt.s > 0)
        return n_pos, len(self.particles) - n_pos

    def positive_center_velocity(self) -> tuple[float, float, float, float, float, float]:
        pos = [pt for pt in self.particles if pt.s > 0]
        if not pos:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n = float(len(pos))
        cx = sum(pt.x for pt in pos) / n
        cy = sum(pt.y for pt in pos) / n
        cz = sum(pt.z for pt in pos) / n
        cvx = sum(pt.vx for pt in pos) / n
        cvy = sum(pt.vy for pt in pos) / n
        cvz = sum(pt.vz for pt in pos) / n
        return cx, cy, cz, cvx, cvy, cvz

    def rotation_curve(self, *, bins: int = 10, r_max: float | None = None) -> list[tuple[float, float, int]]:
        bins = max(3, int(bins))
        cx, cy, _cz, cvx, cvy, _cvz = self.positive_center_velocity()

        if r_max is None:
            r_max = max(1.0, float(self.params.galaxy_radius))
        r_max = max(1e-6, float(r_max))

        sums = [0.0] * bins
        counts = [0] * bins
        dr = r_max / bins

        for pt in self.particles:
            if pt.s <= 0:
                continue
            x = pt.x - cx
            y = pt.y - cy
            vx = pt.vx - cvx
            vy = pt.vy - cvy
            r = math.hypot(x, y)
            if r <= 1e-6 or r > r_max:
                continue
            vphi = (x * vy - y * vx) / r
            k = min(bins - 1, int(r / dr))
            sums[k] += abs(vphi)
            counts[k] += 1

        curve: list[tuple[float, float, int]] = []
        for k in range(bins):
            r_mid = (k + 0.5) * dr
            if counts[k] == 0:
                curve.append((r_mid, 0.0, 0))
            else:
                curve.append((r_mid, sums[k] / counts[k], counts[k]))
        return curve

    def _merge_dense_cells(self, *, accel_mag: list[float] | None = None) -> None:
        p = self.params
        if not bool(getattr(p, "merge_enabled", False)):
            return
        cell = float(getattr(p, "merge_radius", 0.0))
        if not math.isfinite(cell) or cell <= 0.0:
            return
        min_count = max(2, int(getattr(p, "merge_min_count", 2)))
        if len(self.particles) < min_count:
            return
        mode = str(getattr(p, "merge_mode", "all")).strip().lower()
        max_cells = max(0, int(getattr(p, "merge_max_cells", 0)))

        static_shell = bool(
            p.bound_mode == "sphere"
            and p.negative_on_boundary
            and getattr(p, "negative_static_on_boundary", False)
        )

        inv_cell = 1.0 / cell
        cells: dict[tuple[int, int, int, int], list[int]] = {}
        for i, pt in enumerate(self.particles):
            ix = int(math.floor(pt.x * inv_cell))
            iy = int(math.floor(pt.y * inv_cell))
            iz = int(math.floor(pt.z * inv_cell))
            key = (ix, iy, iz, pt.s)
            cells.setdefault(key, []).append(i)

        if not cells:
            return

        new_particles: list[Particle3D] = []
        new_accel: list[float] | None = [] if accel_mag is not None else None
        merged_any = False
        merged_cells = 0

        for key in sorted(cells.keys()):
            idxs = cells[key]
            sign = key[3]
            if mode == "mplus" and sign < 0:
                for idx in idxs:
                    new_particles.append(self.particles[idx])
                    if new_accel is not None:
                        new_accel.append(accel_mag[idx] if idx < len(accel_mag) else 0.0)
                continue
            if static_shell and sign < 0:
                for idx in idxs:
                    new_particles.append(self.particles[idx])
                    if new_accel is not None:
                        new_accel.append(accel_mag[idx] if idx < len(accel_mag) else 0.0)
                continue
            if len(idxs) < min_count:
                for idx in idxs:
                    new_particles.append(self.particles[idx])
                    if new_accel is not None:
                        new_accel.append(accel_mag[idx] if idx < len(accel_mag) else 0.0)
                continue
            if max_cells > 0 and merged_cells >= max_cells:
                for idx in idxs:
                    new_particles.append(self.particles[idx])
                    if new_accel is not None:
                        new_accel.append(accel_mag[idx] if idx < len(accel_mag) else 0.0)
                continue

            temp_thresh = float(getattr(p, "merge_temp_threshold", 0.0))
            if temp_thresh > 0.0:
                temp_sum = 0.0
                for idx in idxs:
                    pt = self.particles[idx]
                    v2 = (pt.vx * pt.vx) + (pt.vy * pt.vy) + (pt.vz * pt.vz)
                    temp_sum += float(pt.m) * v2
                temp_avg = temp_sum / float(len(idxs))
                if temp_avg < temp_thresh:
                    for idx in idxs:
                        new_particles.append(self.particles[idx])
                        if new_accel is not None:
                            new_accel.append(accel_mag[idx] if idx < len(accel_mag) else 0.0)
                    continue

            m_sum = 0.0
            x_sum = y_sum = z_sum = 0.0
            vx_sum = vy_sum = vz_sum = 0.0
            amag_sum = 0.0
            for idx in idxs:
                pt = self.particles[idx]
                m = float(pt.m)
                m_sum += m
                x_sum += pt.x * m
                y_sum += pt.y * m
                z_sum += pt.z * m
                vx_sum += pt.vx * m
                vy_sum += pt.vy * m
                vz_sum += pt.vz * m
                if new_accel is not None and idx < len(accel_mag):
                    amag_sum += accel_mag[idx] * m

            if m_sum <= 0.0:
                for idx in idxs:
                    new_particles.append(self.particles[idx])
                    if new_accel is not None:
                        new_accel.append(accel_mag[idx] if idx < len(accel_mag) else 0.0)
                continue

            merged_any = True
            merged_cells += 1
            x = x_sum / m_sum
            y = y_sum / m_sum
            z = z_sum / m_sum
            vx = vx_sum / m_sum
            vy = vy_sum / m_sum
            vz = vz_sum / m_sum
            new_particles.append(Particle3D(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, s=sign, m=m_sum, blob=True))
            if new_accel is not None:
                new_accel.append(amag_sum / m_sum if m_sum > 0.0 else 0.0)

        if not merged_any:
            return
        self.particles = new_particles
        if new_accel is not None:
            self.accel_mag = new_accel

    def m2_mode(self) -> float:
        # Simple m=2 mode indicator for M+ in the disk plane.
        cx, cy, _cz, _cvx, _cvy, _cvz = self.positive_center_velocity()
        re = 0.0
        im = 0.0
        n = 0
        for pt in self.particles:
            if pt.s <= 0:
                continue
            x = pt.x - cx
            y = pt.y - cy
            r2 = (x * x) + (y * y)
            if r2 <= 1e-8:
                continue
            a = math.atan2(y, x)
            re += math.cos(2.0 * a)
            im += math.sin(2.0 * a)
            n += 1
        if n == 0:
            return 0.0
        return math.sqrt((re * re) + (im * im)) / float(n)

    def step(self, dt: float) -> None:
        """
        Advance the simulation by one time step.
        
        This method computes gravitational forces and integrates particle
        positions and velocities using a simple Euler scheme.
        
        Args:
            dt: Time step in simulation units.
        
        Note:
            Uses NumPy vectorization when available for ~2-3x speedup on
            the integration phase.
        """
        p = self.params

        max_speed = p.max_speed
        damping = p.damping
        bounce = p.bounce
        m_pos = float(p.mass_positive)
        m_neg = float(p.mass_negative)
        static_shell = bool(p.bound_mode == "sphere" and p.negative_on_boundary and getattr(p, "negative_static_on_boundary", False))

        n = len(self.particles)
        if n == 0:
            return
            
        xs = self._xs
        ys = self._ys
        zs = self._zs
        charges = self._charges
        signs = self._signs
        ax = self._ax
        ay = self._ay
        az = self._az

        for arr in (xs, ys, zs, charges, signs, ax, ay, az):
            if len(arr) < n:
                arr.extend([0.0] * (n - len(arr)))
            else:
                del arr[n:]

        space_half = 1.0
        for i, pt in enumerate(self.particles):
            xs[i] = pt.x
            ys[i] = pt.y
            zs[i] = pt.z
            if not getattr(pt, "blob", False):
                pt.m = m_neg if pt.s < 0 else m_pos
            charges[i] = float(pt.s) * float(pt.m)
            signs[i] = 1.0 if pt.s > 0 else -1.0
            ax[i] = 0.0
            ay[i] = 0.0
            az[i] = 0.0
            space_half = max(space_half, abs(pt.x), abs(pt.y), abs(pt.z))

        use_forces = p.janus_enabled and p.janus_g > 0.0 and len(self.particles) > 1
        backend = str(getattr(p, "force_backend", "cpu")).lower()
        use_metal = use_forces and backend == "metal"
        use_direct = use_forces and backend == "cpu_direct"
        eps2 = p.softening * p.softening
        tile_size = int(getattr(p, "force_tile_size", 256))

        self.last_force_backend = "off"
        self.last_force_ms = None

        if use_forces:
            if use_metal:
                self._ensure_metal(tile_size=tile_size)
                t0 = time.perf_counter()
                accel = self._accel_metal(
                    xs=xs,
                    ys=ys,
                    zs=zs,
                    charges=charges,
                    signs=signs,
                    g=p.janus_g,
                    eps2=eps2,
                    debug=bool(getattr(p, "force_debug", False)),
                )
                if accel is not None:
                    for i in range(n):
                        ax[i] = float(accel[i, 0])
                        ay[i] = float(accel[i, 1])
                        az[i] = float(accel[i, 2])
                    self.last_force_backend = "metal"
                    self.last_force_ms = (time.perf_counter() - t0) * 1000.0
                else:
                    use_metal = False

            if use_direct and not use_metal:
                # Initialize solver lazily
                if self._direct_cpu_solver is None or self._direct_cpu_solver.tile_size != tile_size:
                    self._direct_cpu_solver = DirectCPUSolver(tile_size=tile_size)
                
                t0 = time.perf_counter()
                result = self._direct_cpu_solver.compute(
                    xs=xs,
                    ys=ys,
                    zs=zs,
                    charges=charges,
                    signs=signs,
                    g=p.janus_g,
                    eps2=eps2,
                )
                if result is not None:
                    ax[:] = result[0]
                    ay[:] = result[1]
                    az[:] = result[2]
                    self.last_force_backend = "cpu_direct"
                    self.last_force_ms = (time.perf_counter() - t0) * 1000.0
                else:
                    use_direct = False

            if not use_metal and not use_direct:
                # Initialize solver lazily
                if self._barnes_hut_solver is None or self._barnes_hut_solver.theta != p.theta:
                    self._barnes_hut_solver = BarnesHutSolver(theta=p.theta)
                    self._octree_valid = False  # Invalidate cache when theta changes
                
                # Compute particle signs as integers for the solver
                particle_signs = [1 if pt.s > 0 else -1 for pt in self.particles]
                
                t0 = time.perf_counter()
                
                # Use cached octree if available
                if self._octree_valid and self._octree_cache is not None:
                    result = self._barnes_hut_solver.compute(
                        xs=xs,
                        ys=ys,
                        zs=zs,
                        charges=charges,
                        signs=signs,
                        g=p.janus_g,
                        eps2=eps2,
                        particle_signs=particle_signs,
                        cached_octree=self._octree_cache,
                        return_octree=True,
                    )
                    ax[:], ay[:], az[:], self._octree_cache = result
                else:
                    # Build new octree and cache it
                    result = self._barnes_hut_solver.compute(
                        xs=xs,
                        ys=ys,
                        zs=zs,
                        charges=charges,
                        signs=signs,
                        g=p.janus_g,
                        eps2=eps2,
                        particle_signs=particle_signs,
                        return_octree=True,
                    )
                    ax[:], ay[:], az[:], self._octree_cache = result
                    self._octree_valid = True
                
                self.last_force_backend = "cpu"
                self.last_force_ms = (time.perf_counter() - t0) * 1000.0

        if bool(getattr(p, "color_gradient", False)):
            amag = self.accel_mag
            if len(amag) < n:
                amag.extend([0.0] * (n - len(amag)))
            else:
                del amag[n:]
            for i in range(n):
                amag[i] = math.sqrt((ax[i] * ax[i]) + (ay[i] * ay[i]) + (az[i] * az[i]))

        # === OPTIMIZED INTEGRATION USING NUMPY ===
        if np is not None and n > 50:  # Use NumPy for larger particle counts
            self._step_integrate_numpy(dt, n, ax, ay, az, damping, max_speed, static_shell, bounce)
        else:
            # Fallback to Python loop for small systems or when NumPy unavailable
            self._step_integrate_python(dt, n, ax, ay, az, damping, max_speed, static_shell, bounce)

        self._merge_dense_cells(accel_mag=self.accel_mag if bool(getattr(p, "color_gradient", False)) else None)

    def _step_integrate_numpy(
        self,
        dt: float,
        n: int,
        ax: list[float],
        ay: list[float],
        az: list[float],
        damping: float,
        max_speed: float,
        static_shell: bool,
        bounce: float,
    ) -> None:
        """Vectorized integration using NumPy arrays."""
        if np is None:
            return
            
        # Extract particle data into arrays
        pos = np.empty((n, 3), dtype=np.float64)
        vel = np.empty((n, 3), dtype=np.float64)
        acc = np.empty((n, 3), dtype=np.float64)
        signs_arr = np.empty(n, dtype=np.int32)
        
        for i, pt in enumerate(self.particles):
            pos[i, 0] = pt.x
            pos[i, 1] = pt.y
            pos[i, 2] = pt.z
            vel[i, 0] = pt.vx
            vel[i, 1] = pt.vy
            vel[i, 2] = pt.vz
            acc[i, 0] = ax[i]
            acc[i, 1] = ay[i]
            acc[i, 2] = az[i]
            signs_arr[i] = self.particles[i].s
        
        # Mask for particles to integrate (not static shell)
        if static_shell:
            active = signs_arr > 0  # Only M+ particles move
        else:
            active = np.ones(n, dtype=bool)
        
        # Zero velocity for static particles
        if static_shell:
            vel[~active] = 0.0
        
        # Velocity update: v += a * dt
        vel[active] += acc[active] * dt
        
        # Apply damping: v *= damping
        vel[active] *= damping
        
        # Speed clamping
        if max_speed > 0.0:
            speed2 = np.sum(vel[active] ** 2, axis=1)
            fast_mask_local = speed2 > max_speed * max_speed
            if np.any(fast_mask_local):
                # Get indices in the active subset that are too fast
                active_indices = np.where(active)[0]
                fast_indices = active_indices[fast_mask_local]
                speed = np.sqrt(speed2[fast_mask_local])
                scale = max_speed / speed
                vel[fast_indices] *= scale[:, np.newaxis]
        
        # Position update: x += v * dt
        pos[active] += vel[active] * dt
        
        # Write back to particles and apply bounds
        for i, pt in enumerate(self.particles):
            pt.x = pos[i, 0]
            pt.y = pos[i, 1]
            pt.z = pos[i, 2]
            pt.vx = vel[i, 0]
            pt.vy = vel[i, 1]
            pt.vz = vel[i, 2]
            if active[i]:
                self._apply_bounds(pt, bounce=bounce)

    def _step_integrate_python(
        self,
        dt: float,
        n: int,
        ax: list[float],
        ay: list[float],
        az: list[float],
        damping: float,
        max_speed: float,
        static_shell: bool,
        bounce: float,
    ) -> None:
        """Standard Python loop integration (fallback)."""
        for i, pt in enumerate(self.particles):
            if static_shell and pt.s < 0:
                pt.vx = pt.vy = pt.vz = 0.0
                continue

            pt.vx += ax[i] * dt
            pt.vy += ay[i] * dt
            pt.vz += az[i] * dt

            pt.vx *= damping
            pt.vy *= damping
            pt.vz *= damping

            speed2 = pt.vx * pt.vx + pt.vy * pt.vy + pt.vz * pt.vz
            if max_speed > 0.0 and speed2 > max_speed * max_speed:
                s = math.sqrt(speed2)
                if s > 0:
                    scale = max_speed / s
                    pt.vx *= scale
                    pt.vy *= scale
                    pt.vz *= scale

            pt.x += pt.vx * dt
            pt.y += pt.vy * dt
            pt.z += pt.vz * dt
            self._apply_bounds(pt, bounce=bounce)

    def validate_state(self) -> list[str]:
        issues: list[str] = []
        p = self.params
        eps = 1e-6
        check_bounds = bool(p.bounds_enabled)
        bound_mode = str(p.bound_mode)

        if check_bounds and bound_mode == "box":
            b = float(p.bounds)
        elif check_bounds:
            r = float(p.bound_sphere_radius)
            inv_fz = 1.0 / max(1e-9, float(p.bound_sphere_flatten_z))
            r2_max = r * r

        for i, pt in enumerate(self.particles):
            if not (
                math.isfinite(pt.x)
                and math.isfinite(pt.y)
                and math.isfinite(pt.z)
                and math.isfinite(pt.vx)
                and math.isfinite(pt.vy)
                and math.isfinite(pt.vz)
            ):
                issues.append(f"particle {i} has non-finite position/velocity")
                continue

            if not check_bounds:
                continue
            if bound_mode == "box":
                if abs(pt.x) > b + eps or abs(pt.y) > b + eps or abs(pt.z) > b + eps:
                    issues.append(f"particle {i} out of box bounds")
            else:
                z_scaled = pt.z * inv_fz
                r2 = (pt.x * pt.x) + (pt.y * pt.y) + (z_scaled * z_scaled)
                if r2 > r2_max + eps:
                    issues.append(f"particle {i} out of sphere bounds")

        return issues
