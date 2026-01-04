import unittest

from particle_sim3d.params import Sim3DParams


class TestParams(unittest.TestCase):
    def test_clamp_basic_bounds(self) -> None:
        params = Sim3DParams(
            width=10,
            height=10,
            init_mode="bad",
            negative_fraction=0.0,
            bound_mode="weird",
            bound_sphere_flatten_z=0.0,
            time_scale=1000.0,
            target_fps=1,
        ).clamp()

        self.assertEqual(params.width, 320)
        self.assertEqual(params.height, 240)
        self.assertEqual(params.init_mode, "janus_galaxy")
        self.assertEqual(params.bound_mode, "box")
        self.assertAlmostEqual(params.negative_fraction, 0.01, places=6)
        self.assertAlmostEqual(params.bound_sphere_flatten_z, 0.05, places=6)
        self.assertEqual(params.time_scale, 100.0)
        self.assertEqual(params.target_fps, 10)

    def test_validate_warnings(self) -> None:
        params = Sim3DParams(
            bound_mode="box",
            bound_wire_visible=True,
            negative_on_boundary=True,
            negative_static_on_boundary=True,
            bounds_enabled=False,
            bounce=0.5,
            janus_enabled=False,
            janus_g=10.0,
            init_mode="random",
        ).clamp()

        warnings = params.validate()
        self.assertTrue(any("sphere mode" in w for w in warnings))
        self.assertTrue(any("bounds are disabled" in w for w in warnings))
        self.assertTrue(any("janus_enabled is false" in w for w in warnings))
        self.assertTrue(any("init_mode=random" in w for w in warnings))
