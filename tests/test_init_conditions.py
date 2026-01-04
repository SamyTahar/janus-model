"""Tests for initial condition generation."""

import math
import unittest

from particle_sim3d.params import Sim3DParams
from particle_sim3d.sim import ParticleSim3D


class TestRandomInitialization(unittest.TestCase):
    """Tests for random initialization mode."""

    def test_random_init_particle_count(self) -> None:
        """Random init should create the correct number of particles."""
        params = Sim3DParams(
            particle_count=100,
            init_mode="random",
            negative_fraction=0.4,
            bounds_enabled=False,
        ).clamp()

        sim = ParticleSim3D(params)
        self.assertEqual(len(sim.particles), 100)

    def test_random_init_population_ratio(self) -> None:
        """Random init should respect negative_fraction approximately."""
        params = Sim3DParams(
            particle_count=1000,
            init_mode="random",
            negative_fraction=0.3,
            bounds_enabled=False,
            seed=42,
        ).clamp()

        sim = ParticleSim3D(params)
        n_neg = sum(1 for p in sim.particles if p.s < 0)
        n_pos = sum(1 for p in sim.particles if p.s > 0)

        # Should be close to 30% negative
        ratio = n_neg / len(sim.particles)
        self.assertAlmostEqual(ratio, 0.3, delta=0.05)
        self.assertEqual(n_pos + n_neg, len(sim.particles))


class TestJanusGalaxyInitialization(unittest.TestCase):
    """Tests for janus_galaxy initialization mode."""

    def test_galaxy_init_particle_count(self) -> None:
        """Galaxy init should create the correct number of particles."""
        params = Sim3DParams(
            particle_count=500,
            init_mode="janus_galaxy",
            negative_fraction=0.6,
            void_radius=50.0,
            galaxy_radius=30.0,
            bounds_enabled=False,
        ).clamp()

        sim = ParticleSim3D(params)
        self.assertEqual(len(sim.particles), 500)

    def test_galaxy_positive_in_disk(self) -> None:
        """M+ particles should be within galaxy_radius in XY plane."""
        params = Sim3DParams(
            particle_count=200,
            init_mode="janus_galaxy",
            negative_fraction=0.5,
            galaxy_radius=50.0,
            galaxy_thickness=5.0,
            void_radius=100.0,
            bounds_enabled=False,
            seed=123,
        ).clamp()

        sim = ParticleSim3D(params)

        for pt in sim.particles:
            if pt.s > 0:  # M+ particle
                r_xy = math.sqrt(pt.x**2 + pt.y**2)
                self.assertLessEqual(r_xy, params.galaxy_radius * 1.1,
                                     f"M+ particle at r={r_xy} exceeds galaxy_radius")

    def test_void_respected(self) -> None:
        """M- particles should be outside void_radius."""
        params = Sim3DParams(
            particle_count=300,
            init_mode="janus_galaxy",
            negative_fraction=0.7,
            void_radius=40.0,
            galaxy_radius=30.0,
            negative_clump_count=5,
            negative_clump_sigma=10.0,
            bound_mode="box",
            bounds=200.0,
            bounds_enabled=True,
            seed=456,
        ).clamp()

        sim = ParticleSim3D(params)

        # Count violations (allow small margin for numerical issues)
        violations = 0
        margin = 0.95  # Allow 5% inside void due to re-projection

        for pt in sim.particles:
            if pt.s < 0:  # M- particle
                r = math.sqrt(pt.x**2 + pt.y**2 + pt.z**2)
                if r < params.void_radius * margin:
                    violations += 1

        # Very few violations allowed (projection should push them out)
        self.assertLess(violations, 5,
                        f"{violations} M- particles inside void")

    def test_exponential_radius_distribution(self) -> None:
        """M+ particles should follow exponential disk profile."""
        params = Sim3DParams(
            particle_count=2000,
            init_mode="janus_galaxy",
            negative_fraction=0.3,  # 70% M+
            galaxy_radius=100.0,
            galaxy_scale_length=20.0,
            galaxy_thickness=2.0,
            void_radius=150.0,
            bounds_enabled=False,
            seed=789,
        ).clamp()

        sim = ParticleSim3D(params)

        # Collect radii of M+ particles
        radii = []
        for pt in sim.particles:
            if pt.s > 0:
                r = math.sqrt(pt.x**2 + pt.y**2)
                radii.append(r)

        # Check that mean radius is reasonable for exponential disk
        # For exponential disk, mean r ≈ 2 * scale_length
        mean_r = sum(radii) / len(radii) if radii else 0
        expected_mean = 2.0 * params.galaxy_scale_length

        # Allow 50% deviation due to truncation at galaxy_radius
        self.assertLess(abs(mean_r - expected_mean) / expected_mean, 0.5,
                        f"Mean radius {mean_r:.1f} far from expected {expected_mean:.1f}")

    def test_galaxy_rotation(self) -> None:
        """M+ particles should have tangential velocity (rotation)."""
        params = Sim3DParams(
            particle_count=500,
            init_mode="janus_galaxy",
            negative_fraction=0.2,
            galaxy_radius=50.0,
            galaxy_vmax=10.0,
            galaxy_sigma_v=0.5,  # Low dispersion
            bounds_enabled=False,
            seed=111,
        ).clamp()

        sim = ParticleSim3D(params)

        # Check that M+ particles have net rotation (positive Lz)
        total_lz = 0.0
        count = 0
        for pt in sim.particles:
            if pt.s > 0:
                # Lz = x * vy - y * vx
                lz = pt.x * pt.vy - pt.y * pt.vx
                total_lz += lz
                count += 1

        avg_lz = total_lz / count if count > 0 else 0
        # Should have significant positive angular momentum
        self.assertGreater(avg_lz, 1.0,
                           f"Average Lz = {avg_lz:.2f}, expected positive rotation")


class TestNegativeOnBoundary(unittest.TestCase):
    """Tests for negative_on_boundary mode."""

    def test_particles_on_sphere_surface(self) -> None:
        """With negative_on_boundary, M- should be on the sphere surface."""
        params = Sim3DParams(
            particle_count=200,
            init_mode="janus_galaxy",
            negative_fraction=0.6,
            bound_mode="sphere",
            bound_sphere_radius=100.0,
            bound_sphere_flatten_z=1.0,  # Perfect sphere
            negative_on_boundary=True,
            bounds_enabled=True,
            seed=222,
        ).clamp()

        sim = ParticleSim3D(params)
        r_expected = params.bound_sphere_radius

        for pt in sim.particles:
            if pt.s < 0:
                r = math.sqrt(pt.x**2 + pt.y**2 + pt.z**2)
                self.assertAlmostEqual(r, r_expected, delta=0.1,
                                       msg=f"M- at r={r:.2f}, expected {r_expected}")

    def test_particles_on_spheroid_surface(self) -> None:
        """With flattening, M- should be on the oblate spheroid surface."""
        params = Sim3DParams(
            particle_count=200,
            init_mode="janus_galaxy",
            negative_fraction=0.6,
            bound_mode="sphere",
            bound_sphere_radius=100.0,
            bound_sphere_flatten_z=0.5,  # Flattened
            negative_on_boundary=True,
            bounds_enabled=True,
            seed=333,
        ).clamp()

        sim = ParticleSim3D(params)
        r = params.bound_sphere_radius
        fz = params.bound_sphere_flatten_z

        for pt in sim.particles:
            if pt.s < 0:
                # Spheroid equation: x²/R² + y²/R² + z²/(R*fz)² = 1
                # => (x² + y²) / R² + z² / (R*fz)² = 1
                term = (pt.x**2 + pt.y**2) / (r**2) + pt.z**2 / ((r * fz)**2)
                self.assertAlmostEqual(term, 1.0, delta=0.01,
                                       msg=f"M- not on spheroid: term={term:.4f}")


class TestExplicitPopulationMode(unittest.TestCase):
    """Tests for population_mode='explicit'."""

    def test_explicit_counts(self) -> None:
        """Explicit mode should use positive_count and negative_count."""
        params = Sim3DParams(
            population_mode="explicit",
            positive_count=100,
            negative_count=50,
            init_mode="random",
            bounds_enabled=False,
        ).clamp()

        sim = ParticleSim3D(params)

        n_pos = sum(1 for p in sim.particles if p.s > 0)
        n_neg = sum(1 for p in sim.particles if p.s < 0)

        self.assertEqual(n_pos, 100)
        self.assertEqual(n_neg, 50)
        self.assertEqual(len(sim.particles), 150)


class TestMassAssignment(unittest.TestCase):
    """Tests for mass assignment to particles."""

    def test_mass_positive_negative_independent(self) -> None:
        """M+ and M- should have independent masses."""
        params = Sim3DParams(
            particle_count=100,
            init_mode="random",
            negative_fraction=0.5,
            mass_positive=2.5,
            mass_negative=7.0,
            bounds_enabled=False,
        ).clamp()

        sim = ParticleSim3D(params)

        for pt in sim.particles:
            if pt.s > 0:
                self.assertAlmostEqual(pt.m, 2.5, places=6)
            else:
                self.assertAlmostEqual(pt.m, 7.0, places=6)


class TestSeedReproducibility(unittest.TestCase):
    """Tests for reproducibility with the same seed."""

    def test_same_seed_same_positions(self) -> None:
        """Same seed should produce identical initial conditions."""
        params1 = Sim3DParams(
            particle_count=50,
            init_mode="janus_galaxy",
            negative_fraction=0.4,
            seed=99999,
            bounds_enabled=False,
        ).clamp()

        params2 = Sim3DParams(
            particle_count=50,
            init_mode="janus_galaxy",
            negative_fraction=0.4,
            seed=99999,
            bounds_enabled=False,
        ).clamp()

        sim1 = ParticleSim3D(params1)
        sim2 = ParticleSim3D(params2)

        self.assertEqual(len(sim1.particles), len(sim2.particles))

        for p1, p2 in zip(sim1.particles, sim2.particles):
            self.assertAlmostEqual(p1.x, p2.x, places=10)
            self.assertAlmostEqual(p1.y, p2.y, places=10)
            self.assertAlmostEqual(p1.z, p2.z, places=10)
            self.assertEqual(p1.s, p2.s)


if __name__ == "__main__":
    unittest.main()
