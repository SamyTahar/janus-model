import unittest

from particle_sim3d.params import Sim3DParams
from particle_sim3d.sim import ParticleSim3D


class TestSim(unittest.TestCase):
    def test_no_accel_when_janus_disabled(self) -> None:
        params = Sim3DParams(
            particle_count=2,
            init_mode="random",
            negative_fraction=0.5,
            janus_enabled=True,
            janus_g=10.0,
            softening=1.0,
            damping=1.0,
            bounds_enabled=False,
            max_speed=0.0,
            seed=1,
        ).clamp()

        sim = ParticleSim3D(params)
        sim.step(0.01)

        params.janus_enabled = False
        sim.params = params
        before = [(pt.vx, pt.vy, pt.vz) for pt in sim.particles]
        sim.step(0.01)
        after = [(pt.vx, pt.vy, pt.vz) for pt in sim.particles]

        for (bx, by, bz), (ax, ay, az) in zip(before, after):
            self.assertAlmostEqual(ax, bx, places=12)
            self.assertAlmostEqual(ay, by, places=12)
            self.assertAlmostEqual(az, bz, places=12)

    def test_validate_state_flags_nan(self) -> None:
        params = Sim3DParams(particle_count=1, init_mode="random", bounds_enabled=False).clamp()
        sim = ParticleSim3D(params)

        self.assertEqual(sim.validate_state(), [])

        sim.particles[0].x = float("nan")
        issues = sim.validate_state()
        self.assertTrue(any("non-finite" in issue for issue in issues))
