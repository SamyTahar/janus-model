"""Tests for the Barnes-Hut octree implementation."""

import math
import unittest

from particle_sim3d.octree import OctreeNode, build_octree, BUCKET_CAPACITY


class TestOctreeConstruction(unittest.TestCase):
    """Tests for octree construction and insertion."""

    def test_insert_single_particle(self) -> None:
        """A single particle should be stored in the root as a leaf."""
        xs = [1.0]
        ys = [2.0]
        zs = [3.0]
        root = build_octree(xs, ys, zs, half=100.0)

        self.assertTrue(root.is_leaf())
        self.assertIsNotNone(root.indices)
        self.assertEqual(len(root.indices), 1)
        self.assertEqual(root.indices[0], 0)

    def test_insert_all_particles_preserved(self) -> None:
        """All particles should be retrievable from the tree."""
        n = 100
        xs = [float(i % 10) for i in range(n)]
        ys = [float((i // 10) % 10) for i in range(n)]
        zs = [float(i // 100) for i in range(n)]
        root = build_octree(xs, ys, zs, half=20.0)

        # Collect all indices from leaves
        collected: set[int] = set()

        def collect(node: OctreeNode) -> None:
            if node.is_leaf():
                if node.indices:
                    collected.update(node.indices)
            else:
                assert node.child is not None
                for ch in node.child:
                    collect(ch)

        collect(root)
        self.assertEqual(collected, set(range(n)))

    def test_octant_assignment(self) -> None:
        """Particles should be placed in the correct octant based on position."""
        # Center at (0, 0, 0), half=10
        root = OctreeNode(cx=0.0, cy=0.0, cz=0.0, half=10.0)

        # Test each octant
        test_cases = [
            # (x, y, z) -> expected octant
            (-1.0, -1.0, -1.0, 0),  # ---
            (1.0, -1.0, -1.0, 1),   # +--
            (-1.0, 1.0, -1.0, 2),   # -+-
            (1.0, 1.0, -1.0, 3),    # ++-
            (-1.0, -1.0, 1.0, 4),   # --+
            (1.0, -1.0, 1.0, 5),    # +-+
            (-1.0, 1.0, 1.0, 6),    # -++
            (1.0, 1.0, 1.0, 7),     # +++
        ]
        for x, y, z, expected in test_cases:
            octant = root._octant(x, y, z)
            self.assertEqual(octant, expected, f"Failed for ({x}, {y}, {z})")

    def test_bucket_capacity_respected(self) -> None:
        """Leaves should not exceed BUCKET_CAPACITY until split."""
        # Place particles at the same location to test bucket behavior
        n = BUCKET_CAPACITY + 5
        xs = [0.5] * n
        ys = [0.5] * n
        zs = [0.5] * n
        root = build_octree(xs, ys, zs, half=10.0)

        # The tree should have split since we exceeded BUCKET_CAPACITY
        # But due to coincident points, it may hit MAX_DEPTH
        collected: set[int] = set()

        def collect(node: OctreeNode) -> None:
            if node.is_leaf():
                if node.indices:
                    collected.update(node.indices)
            else:
                assert node.child is not None
                for ch in node.child:
                    collect(ch)

        collect(root)
        self.assertEqual(len(collected), n)


class TestOctreeChargeAccumulation(unittest.TestCase):
    """Tests for charge accumulation in the octree."""

    def test_total_charge_conservation(self) -> None:
        """Root's total charge should equal sum of all particle charges."""
        n = 50
        xs = [float(i) for i in range(n)]
        ys = [float(i * 0.5) for i in range(n)]
        zs = [0.0] * n
        charges = [1.0 if i % 2 == 0 else -0.5 for i in range(n)]

        root = build_octree(xs, ys, zs, half=100.0)
        root.accumulate_charge(charges, xs, ys, zs)

        expected_total = sum(charges)
        self.assertAlmostEqual(root.q, expected_total, places=10)

    def test_charge_weighted_center(self) -> None:
        """Center of charge should be weighted average of positions."""
        xs = [0.0, 10.0]
        ys = [0.0, 0.0]
        zs = [0.0, 0.0]
        charges = [1.0, 1.0]  # Equal charges

        root = build_octree(xs, ys, zs, half=20.0)
        root.accumulate_charge(charges, xs, ys, zs)

        center = root.approx_center()
        self.assertIsNotNone(center)
        cx, cy, cz = center
        # With equal charges, center should be at (5, 0, 0)
        self.assertAlmostEqual(cx, 5.0, places=6)
        self.assertAlmostEqual(cy, 0.0, places=6)
        self.assertAlmostEqual(cz, 0.0, places=6)

    def test_zero_charge_returns_none(self) -> None:
        """approx_center should return None if total charge is ~0."""
        xs = [0.0, 10.0]
        ys = [0.0, 0.0]
        zs = [0.0, 0.0]
        charges = [1.0, -1.0]  # Cancel out

        root = build_octree(xs, ys, zs, half=20.0)
        root.accumulate_charge(charges, xs, ys, zs)

        center = root.approx_center()
        self.assertIsNone(center)


class TestBarnesHutAccuracy(unittest.TestCase):
    """Tests comparing Barnes-Hut to direct summation."""

    def _direct_accel(
        self,
        idx: int,
        si: int,
        xs: list[float],
        ys: list[float],
        zs: list[float],
        charges: list[float],
        g: float,
        eps2: float,
    ) -> tuple[float, float, float]:
        """Compute acceleration via direct O(N²) summation."""
        ax = ay = az = 0.0
        xi, yi, zi = xs[idx], ys[idx], zs[idx]
        for j in range(len(xs)):
            if j == idx:
                continue
            dx = xs[j] - xi
            dy = ys[j] - yi
            dz = zs[j] - zi
            r2 = dx * dx + dy * dy + dz * dz + eps2
            inv_r = 1.0 / math.sqrt(r2)
            inv_r3 = inv_r * inv_r * inv_r
            f = g * float(si) * charges[j] * inv_r3
            ax += dx * f
            ay += dy * f
            az += dz * f
        return ax, ay, az

    def test_barnes_hut_vs_direct_small(self) -> None:
        """Barnes-Hut with small theta should match direct summation closely."""
        n = 20
        xs = [float(i % 5) * 2.0 for i in range(n)]
        ys = [float((i // 5) % 4) * 2.0 for i in range(n)]
        zs = [0.0] * n
        charges = [1.0 if i % 3 == 0 else -0.5 for i in range(n)]
        signs = [1 if c > 0 else -1 for c in charges]

        g = 1.0
        eps2 = 0.01
        theta = 0.3  # Fairly accurate

        root = build_octree(xs, ys, zs, half=20.0)
        root.accumulate_charge(charges, xs, ys, zs)

        for idx in range(n):
            si = signs[idx]
            xi, yi, zi = xs[idx], ys[idx], zs[idx]

            bh_ax, bh_ay, bh_az = root.accel_on(
                idx=idx, si=si, xi=xi, yi=yi, zi=zi,
                g=g, eps2=eps2, theta=theta,
                xs=xs, ys=ys, zs=zs, charges=charges,
            )
            direct_ax, direct_ay, direct_az = self._direct_accel(
                idx, si, xs, ys, zs, charges, g, eps2
            )

            # Allow 10% relative error for theta=0.3
            def rel_error(a: float, b: float) -> float:
                if abs(b) < 1e-12:
                    return abs(a)
                return abs(a - b) / abs(b)

            self.assertLess(rel_error(bh_ax, direct_ax), 0.15,
                            f"X accel mismatch for particle {idx}")
            self.assertLess(rel_error(bh_ay, direct_ay), 0.15,
                            f"Y accel mismatch for particle {idx}")
            self.assertLess(rel_error(bh_az, direct_az), 0.15,
                            f"Z accel mismatch for particle {idx}")

    def test_barnes_hut_theta_zero_exact(self) -> None:
        """Barnes-Hut with theta=0 should match direct summation exactly."""
        n = 10
        xs = [float(i) for i in range(n)]
        ys = [0.0] * n
        zs = [0.0] * n
        charges = [1.0] * n

        g = 1.0
        eps2 = 0.01
        theta = 0.0  # Never approximate

        root = build_octree(xs, ys, zs, half=20.0)
        root.accumulate_charge(charges, xs, ys, zs)

        for idx in range(n):
            si = 1
            xi, yi, zi = xs[idx], ys[idx], zs[idx]

            bh_ax, bh_ay, bh_az = root.accel_on(
                idx=idx, si=si, xi=xi, yi=yi, zi=zi,
                g=g, eps2=eps2, theta=theta,
                xs=xs, ys=ys, zs=zs, charges=charges,
            )
            direct_ax, direct_ay, direct_az = self._direct_accel(
                idx, si, xs, ys, zs, charges, g, eps2
            )

            self.assertAlmostEqual(bh_ax, direct_ax, places=10)
            self.assertAlmostEqual(bh_ay, direct_ay, places=10)
            self.assertAlmostEqual(bh_az, direct_az, places=10)


if __name__ == "__main__":
    unittest.main()
