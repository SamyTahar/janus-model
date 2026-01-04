"""
Initial condition generators for particle simulations.

This module provides functions to create different initial particle
distributions for the Janus cosmological simulation.

Available modes:
- random: Uniform random distribution
- janus_galaxy: Galaxy M+ in void surrounded by M- clumps
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particle_sim3d.params import Sim3DParams


@dataclass(slots=True)
class InitialParticle:
    """
    Initial conditions for a single particle.
    
    Attributes:
        x, y, z: Position coordinates
        vx, vy, vz: Velocity components
        s: Sign (+1 for M+, -1 for M-)
        m: Mass (always positive)
    """
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    s: int
    m: float


def sample_unit_vector(rng: random.Random) -> tuple[float, float, float]:
    """
    Sample a uniformly distributed unit vector on the sphere.
    
    Args:
        rng: Random number generator instance
    
    Returns:
        (x, y, z) unit vector
    """
    u = rng.random()
    v = rng.random()
    theta = 2.0 * math.pi * u
    phi = math.acos(2.0 * v - 1.0)
    sin_phi = math.sin(phi)
    return (
        math.cos(theta) * sin_phi,
        math.sin(theta) * sin_phi,
        math.cos(phi),
    )


def sample_exponential_radius(
    rng: random.Random,
    *,
    scale: float,
    r_max: float,
) -> float:
    """
    Sample radius from an exponential disk profile.
    
    The probability density is proportional to r * exp(-r/scale),
    truncated at r_max.
    
    Args:
        rng: Random number generator
        scale: Scale length of the exponential profile
        r_max: Maximum radius cutoff
    
    Returns:
        Sampled radius value
    """
    # CDF for r*exp(-r/h): 1 - (1 + r/h)*exp(-r/h)
    # Using rejection sampling is simpler for this distribution
    for _ in range(1000):
        r = rng.random() * r_max
        # Probability proportional to r * exp(-r/scale)
        p = r * math.exp(-r / scale) if scale > 0 else 0.0
        p_max = scale * math.exp(-1.0) if scale > 0 else 0.0  # Peak at r=scale
        p_max = max(p_max, 1e-10)
        if rng.random() * p_max < p:
            return r
    return rng.random() * r_max  # Fallback


def rotation_velocity_profile(
    r: float,
    vmax: float,
    turnover: float,
) -> float:
    """
    Compute circular velocity at radius r using a tanh-like profile.
    
    v(r) = vmax * (1 - exp(-r/turnover))
    
    This gives:
    - Linear rise at small r
    - Flat rotation curve at large r
    
    Args:
        r: Radial distance from center
        vmax: Asymptotic maximum velocity
        turnover: Scale radius for velocity rise
    
    Returns:
        Circular velocity at radius r
    """
    if r <= 0 or turnover <= 0:
        return 0.0
    return vmax * (1.0 - math.exp(-r / turnover))


def create_random_distribution(
    params: "Sim3DParams",
    rng: random.Random,
    n_pos: int,
    n_neg: int,
) -> list[InitialParticle]:
    """
    Create a random uniform distribution of particles.
    
    Particles are distributed uniformly within the bounds,
    with random velocities up to initial_speed.
    
    Args:
        params: Simulation parameters
        rng: Random number generator
        n_pos: Number of M+ particles
        n_neg: Number of M- particles
    
    Returns:
        List of InitialParticle objects
    """
    particles: list[InitialParticle] = []
    
    m_pos = float(params.mass_positive)
    m_neg = float(params.mass_negative)
    bound_mode = str(params.bound_mode)
    
    if bound_mode == "box":
        b = float(params.bounds)
    else:
        r_bound = float(params.bound_sphere_radius)
        fz = max(1e-9, float(params.bound_sphere_flatten_z))
    
    total = n_pos + n_neg
    signs = [1] * n_pos + [-1] * n_neg
    rng.shuffle(signs)
    
    for i in range(total):
        s = signs[i]
        m = m_neg if s < 0 else m_pos
        
        # Position sampling
        if bound_mode == "box":
            x = rng.uniform(-b, b)
            y = rng.uniform(-b, b)
            z = rng.uniform(-b, b)
        else:
            # Sample inside oblate spheroid
            inv_fz = 1.0 / fz
            zmax = r_bound * fz
            r2_max = r_bound * r_bound
            while True:
                x = rng.uniform(-r_bound, r_bound)
                y = rng.uniform(-r_bound, r_bound)
                z = rng.uniform(-zmax, zmax)
                if x*x + y*y + (z * inv_fz)**2 <= r2_max:
                    break
        
        # Random velocity direction and magnitude
        u = rng.random()
        v = rng.random()
        theta = 2.0 * math.pi * u
        phi = math.acos(2.0 * v - 1.0)
        speed = rng.random() * float(params.initial_speed)
        vx = math.cos(theta) * math.sin(phi) * speed
        vy = math.sin(theta) * math.sin(phi) * speed
        vz = math.cos(phi) * speed
        
        particles.append(InitialParticle(
            x=x, y=y, z=z,
            vx=vx, vy=vy, vz=vz,
            s=s, m=m,
        ))
    
    return particles


def create_janus_galaxy(
    params: "Sim3DParams",
    rng: random.Random,
    n_pos: int,
    n_neg: int,
) -> list[InitialParticle]:
    """
    Create a Janus galaxy configuration.
    
    This creates:
    - M+ particles: Exponential disk in XY plane with rotation
    - M- particles: Clumpy distribution outside a central void
    
    Args:
        params: Simulation parameters
        rng: Random number generator
        n_pos: Number of M+ (galaxy) particles
        n_neg: Number of M- (environment) particles
    
    Returns:
        List of InitialParticle objects
    """
    particles: list[InitialParticle] = []
    
    m_pos = float(params.mass_positive)
    m_neg = float(params.mass_negative)
    bound_mode = str(params.bound_mode)
    domain = float(params.bounds) if bound_mode == "box" else float(params.bound_sphere_radius)
    fz = max(1e-9, float(params.bound_sphere_flatten_z))
    
    neg_vphi_scale = float(params.negative_vphi_scale)
    static_shell = bool(
        params.bound_mode == "sphere"
        and params.negative_on_boundary
        and params.negative_static_on_boundary
    )
    
    def negative_rotation_velocity(x: float, y: float) -> tuple[float, float]:
        """Compute rotation velocity for M- particles."""
        if abs(neg_vphi_scale) <= 1e-9:
            return 0.0, 0.0
        r = math.hypot(x, y)
        if r <= 1e-6:
            return 0.0, 0.0
        vphi = rotation_velocity_profile(r, params.galaxy_vmax, params.galaxy_turnover) * neg_vphi_scale
        tx = -y / r
        ty = x / r
        return tx * vphi, ty * vphi
    
    # === M+ GALAXY: Exponential disk ===
    for _ in range(n_pos):
        r = sample_exponential_radius(
            rng,
            scale=params.galaxy_scale_length,
            r_max=params.galaxy_radius,
        )
        angle = rng.random() * 2.0 * math.pi
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        z = rng.gauss(0.0, params.galaxy_thickness)
        
        # Rotation velocity
        if r > 1e-6:
            vphi = rotation_velocity_profile(r, params.galaxy_vmax, params.galaxy_turnover)
            tx = -y / r
            ty = x / r
            vx = tx * vphi + rng.gauss(0.0, params.galaxy_sigma_v)
            vy = ty * vphi + rng.gauss(0.0, params.galaxy_sigma_v)
        else:
            vx = rng.gauss(0.0, params.galaxy_sigma_v)
            vy = rng.gauss(0.0, params.galaxy_sigma_v)
        vz = rng.gauss(0.0, params.galaxy_sigma_v * 0.5)
        
        particles.append(InitialParticle(
            x=x, y=y, z=z,
            vx=vx, vy=vy, vz=vz,
            s=1, m=m_pos,
        ))
    
    # === M- ENVIRONMENT ===
    
    # Special case: particles on boundary sphere
    if params.bound_mode == "sphere" and params.negative_on_boundary:
        r = float(params.bound_sphere_radius)
        for _ in range(n_neg):
            dx, dy, dz = sample_unit_vector(rng)
            x = r * dx
            y = r * dy
            z = r * fz * dz
            
            if static_shell:
                vx = vy = vz = 0.0
            else:
                vx, vy = negative_rotation_velocity(x, y)
                vx += rng.gauss(0.0, params.negative_sigma_v)
                vy += rng.gauss(0.0, params.negative_sigma_v)
                vz = rng.gauss(0.0, params.negative_sigma_v)
            
            particles.append(InitialParticle(
                x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz,
                s=-1, m=m_neg,
            ))
        return particles
    
    # Normal case: clumpy distribution outside void
    outer = domain * 0.95
    inner = max(
        params.void_radius + 2.0 * params.negative_clump_sigma,
        params.void_radius + 5.0,
    )
    if inner >= outer:
        inner = max(params.void_radius, outer * 0.7)
    if inner >= outer:
        inner = outer * 0.5
    
    # Generate clump centers
    clump_centers: list[tuple[float, float, float]] = []
    for _ in range(int(params.negative_clump_count)):
        u = rng.random()
        rr = inner + (outer - inner) * (u * u)  # Bias toward inner
        dx, dy, dz = sample_unit_vector(rng)
        if bound_mode == "sphere":
            clump_centers.append((dx * rr, dy * rr, dz * rr * fz))
        else:
            clump_centers.append((dx * rr, dy * rr, dz * rr))
    
    if not clump_centers:
        clump_centers = [(0.0, 0.0, 0.0)]
    
    # Distribute M- particles around clumps
    for _ in range(n_neg):
        cx, cy, cz = clump_centers[rng.randrange(len(clump_centers))]
        x = cx + rng.gauss(0.0, params.negative_clump_sigma)
        y = cy + rng.gauss(0.0, params.negative_clump_sigma)
        z = cz + rng.gauss(0.0, params.negative_clump_sigma)
        
        # Enforce void: push particles outside void_radius
        rr = math.sqrt(x*x + y*y + z*z)
        if rr < params.void_radius:
            dx, dy, dz = sample_unit_vector(rng)
            rr2 = params.void_radius + abs(rng.gauss(0.0, params.negative_clump_sigma))
            x = dx * rr2
            y = dy * rr2
            z = dz * rr2
        
        # Project into bounds
        if bound_mode == "sphere":
            x, y, z = _project_into_spheroid(x, y, z, domain, fz)
        else:
            x = max(-domain, min(domain, x))
            y = max(-domain, min(domain, y))
            z = max(-domain, min(domain, z))
        
        vx, vy = negative_rotation_velocity(x, y)
        vx += rng.gauss(0.0, params.negative_sigma_v)
        vy += rng.gauss(0.0, params.negative_sigma_v)
        vz = rng.gauss(0.0, params.negative_sigma_v)
        
        particles.append(InitialParticle(
            x=x, y=y, z=z,
            vx=vx, vy=vy, vz=vz,
            s=-1, m=m_neg,
        ))
    
    return particles


def _project_into_spheroid(
    x: float, y: float, z: float,
    radius: float,
    flatten_z: float,
) -> tuple[float, float, float]:
    """
    Project a point inside an oblate spheroid if outside.
    
    The spheroid is defined by: x²/R² + y²/R² + z²/(R·fz)² ≤ 1
    
    Args:
        x, y, z: Point coordinates
        radius: Equatorial radius R
        flatten_z: Flattening factor for z-axis
    
    Returns:
        (x, y, z) projected inside the spheroid
    """
    r = radius
    inv_fz = 1.0 / max(1e-9, flatten_z)
    z_scaled = z * inv_fz
    rr = math.sqrt(x*x + y*y + z_scaled*z_scaled)
    
    if rr <= r:
        return x, y, z
    if rr <= 0.0:
        return 0.0, 0.0, 0.0
    
    t = r / rr
    return x * t, y * t, z * t
