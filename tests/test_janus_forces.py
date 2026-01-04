import math
import unittest

import particle_sim3d.sim as sim_module
from particle_sim3d.params import Sim3DParams
from particle_sim3d.sim import Particle3D, ParticleSim3D


@unittest.skipUnless(sim_module.np is not None, "numpy required for cpu_direct tests")
class TestJanusForces(unittest.TestCase):
    def _accel_direct(
        self,
        *,
        xs: list[float],
        ys: list[float],
        zs: list[float],
        charges: list[float],
        signs: list[float],
        g: float = 1.0,
        eps2: float = 0.0,
    ):
        params = Sim3DParams(
            particle_count=2,
            init_mode="random",
            negative_fraction=0.5,
            janus_enabled=True,
            janus_g=g,
            softening=math.sqrt(eps2),
        ).clamp()
        sim = ParticleSim3D(params)
        return sim._accel_cpu_direct(
            xs=xs,
            ys=ys,
            zs=zs,
            charges=charges,
            signs=signs,
            g=g,
            eps2=eps2,
            tile_size=8,
        )

    def test_two_body_force_signs(self) -> None:
        xs = [0.0, 1.0]
        ys = [0.0, 0.0]
        zs = [0.0, 0.0]
        cases = [
            ("pos_pos", [1.0, 1.0], [1.0, 1.0], 1.0, -1.0),
            ("neg_neg", [-1.0, -1.0], [-1.0, -1.0], 1.0, -1.0),
            ("pos_neg", [1.0, -1.0], [1.0, -1.0], -1.0, 1.0),
        ]

        for _name, charges, signs, dir0, dir1 in cases:
            acc = self._accel_direct(xs=xs, ys=ys, zs=zs, charges=charges, signs=signs)
            self.assertIsNotNone(acc)
            ax0 = float(acc[0, 0])
            ax1 = float(acc[1, 0])
            self.assertTrue(ax0 * dir0 > 0.0)
            self.assertTrue(ax1 * dir1 > 0.0)
            self.assertAlmostEqual(ax0 + ax1, 0.0, places=5)
            self.assertAlmostEqual(float(acc[0, 1]), 0.0, places=6)
            self.assertAlmostEqual(float(acc[1, 1]), 0.0, places=6)

    def test_two_body_scaling_with_distance(self) -> None:
        charges = [1.0, 1.0]
        signs = [1.0, 1.0]
        eps2 = 0.0

        acc_r1 = self._accel_direct(xs=[0.0, 1.0], ys=[0.0, 0.0], zs=[0.0, 0.0], charges=charges, signs=signs)
        acc_r2 = self._accel_direct(xs=[0.0, 2.0], ys=[0.0, 0.0], zs=[0.0, 0.0], charges=charges, signs=signs)
        self.assertIsNotNone(acc_r1)
        self.assertIsNotNone(acc_r2)
        ratio = abs(float(acc_r1[0, 0])) / abs(float(acc_r2[0, 0]))
        self.assertAlmostEqual(ratio, 4.0, places=2)

        eps2 = 1.0
        acc_r1 = self._accel_direct(
            xs=[0.0, 1.0], ys=[0.0, 0.0], zs=[0.0, 0.0], charges=charges, signs=signs, eps2=eps2
        )
        acc_r2 = self._accel_direct(
            xs=[0.0, 2.0], ys=[0.0, 0.0], zs=[0.0, 0.0], charges=charges, signs=signs, eps2=eps2
        )
        self.assertIsNotNone(acc_r1)
        self.assertIsNotNone(acc_r2)
        ratio = abs(float(acc_r1[0, 0])) / abs(float(acc_r2[0, 0]))
        expected = (1.0 / ((1.0 + eps2) ** 1.5)) / (2.0 / ((4.0 + eps2) ** 1.5))
        self.assertAlmostEqual(ratio, expected, places=2)

    def test_momentum_conservation_two_body(self) -> None:
        params = Sim3DParams(
            particle_count=2,
            init_mode="random",
            janus_enabled=True,
            janus_g=1.0,
            softening=0.0,
            damping=1.0,
            bounds_enabled=False,
            max_speed=0.0,
            force_backend="cpu_direct",
            force_tile_size=16,
            mass_positive=2.0,
            mass_negative=3.0,
        ).clamp()
        sim = ParticleSim3D(params)
        sim.particles = [
            Particle3D(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, s=1, m=2.0),
            Particle3D(x=1.5, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, s=-1, m=3.0),
        ]

        def momentum() -> tuple[float, float, float]:
            px = py = pz = 0.0
            for pt in sim.particles:
                mass = params.mass_positive if pt.s > 0 else params.mass_negative
                px += mass * pt.vx
                py += mass * pt.vy
                pz += mass * pt.vz
            return px, py, pz

        before = momentum()
        sim.step(0.01)
        after = momentum()

        dx = after[0] - before[0]
        dy = after[1] - before[1]
        dz = after[2] - before[2]
        delta = math.sqrt((dx * dx) + (dy * dy) + (dz * dz))
        self.assertLess(delta, 1e-5)

    @unittest.skip("PM/FFT solver not implemented in this codebase.")
    def test_pm_fft_k0_handling(self) -> None:
        self.assertTrue(False)
