"""
Drawing utilities for 3D particle visualization.

This module provides helper functions for rendering particles, trails,
grids, and other visual elements using OpenGL and Pyglet.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    try:
        from pyglet import gl  # type: ignore
    except ImportError:
        gl = None


# =============================================================================
# Gradient Color Utilities
# =============================================================================

# Default gradient stops: cold (blue) -> neutral (white) -> warm (orange) -> hot (red)
DEFAULT_GRADIENT_STOPS = [
    (0.0, (40, 100, 190)),
    (0.45, (245, 245, 245)),
    (0.75, (255, 170, 90)),
    (1.0, (230, 60, 50)),
]


def interpolate_gradient_color(
    t: float,
    stops: list[tuple[float, tuple[int, int, int]]] | None = None,
    alpha: int = 255,
) -> tuple[int, int, int, int]:
    """
    Interpolate a color from a gradient defined by stops.
    
    Args:
        t: Position on the gradient (0.0 to 1.0)
        stops: List of (position, (r, g, b)) tuples, sorted by position
        alpha: Alpha value for the output color
    
    Returns:
        (r, g, b, a) color tuple with values 0-255
    """
    if stops is None:
        stops = DEFAULT_GRADIENT_STOPS
    
    t = max(0.0, min(1.0, float(t)))
    
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            local = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            r = int(round(c0[0] + (c1[0] - c0[0]) * local))
            g = int(round(c0[1] + (c1[1] - c0[1]) * local))
            b = int(round(c0[2] + (c1[2] - c0[2]) * local))
            return r, g, b, alpha
    
    # Past the last stop
    r, g, b = stops[-1][1]
    return r, g, b, alpha


def legend_color_at(
    t: float,
    stops: list[tuple[float, tuple[int, int, int]]],
) -> tuple[int, int, int]:
    """
    Get a color from gradient stops (without alpha).
    
    Args:
        t: Position on gradient (0.0 to 1.0)
        stops: Gradient stops
    
    Returns:
        (r, g, b) color tuple
    """
    rgba = interpolate_gradient_color(t, stops, alpha=255)
    return rgba[0], rgba[1], rgba[2]


# =============================================================================
# Grid Generation
# =============================================================================

def create_grid_vertices(
    size: float,
    step: float,
    z: float = 0.0,
) -> list[float]:
    """
    Create vertex data for an XY grid.
    
    Args:
        size: Half-size of the grid (extends from -size to +size)
        step: Spacing between grid lines
        z: Z coordinate (height) of the grid plane
    
    Returns:
        List of vertices [x, y, z, x, y, z, ...] for GL_LINES
    """
    vertices: list[float] = []
    step = max(0.1, float(step))
    size = max(step, float(size))
    
    # Ensure we have an integer number of steps
    n_steps = int(size / step)
    actual_size = n_steps * step
    
    # X-parallel lines
    x_val = -actual_size
    while x_val <= actual_size + step * 0.5:
        vertices.extend([x_val, -actual_size, z, x_val, actual_size, z])
        x_val += step
    
    # Y-parallel lines
    y_val = -actual_size
    while y_val <= actual_size + step * 0.5:
        vertices.extend([-actual_size, y_val, z, actual_size, y_val, z])
        y_val += step
    
    return vertices


def create_grid_colors(
    vertex_count: int,
    color: tuple[int, int, int, int] = (100, 120, 140, 60),
) -> list[int]:
    """
    Create color data for grid vertices.
    
    Args:
        vertex_count: Number of vertices
        color: RGBA color for all vertices
    
    Returns:
        List of colors [r, g, b, a, r, g, b, a, ...]
    """
    return list(color) * vertex_count


# =============================================================================
# Wireframe Sphere Generation
# =============================================================================

def create_spheroid_wireframe(
    radius: float,
    flatten_z: float = 1.0,
    segments: int = 48,
    n_latitudes: int = 5,
    n_meridians: int = 12,
) -> list[list[float]]:
    """
    Create wireframe loops for an oblate spheroid.
    
    Args:
        radius: Equatorial radius
        flatten_z: Flattening factor for Z axis (1.0 = sphere)
        segments: Number of segments per loop
        n_latitudes: Number of latitude rings
        n_meridians: Number of meridian lines
    
    Returns:
        List of loops, each loop is [x, y, z, x, y, z, ...] for GL_LINE_STRIP
    """
    loops: list[list[float]] = []
    flatten_z = max(0.05, float(flatten_z))
    
    # Latitude rings at various heights
    lat_positions = [-0.66, -0.33, 0.0, 0.33, 0.66]
    if n_latitudes != 5:
        lat_positions = [2.0 * i / (n_latitudes + 1) - 1.0 for i in range(1, n_latitudes + 1)]
    
    for k in lat_positions:
        z = radius * flatten_z * k
        rho = radius * math.sqrt(max(0.0, 1.0 - k * k))
        pts: list[float] = []
        for i in range(segments + 1):
            a = (2.0 * math.pi * i) / segments
            x = rho * math.cos(a)
            y = rho * math.sin(a)
            pts.extend([x, y, z])
        loops.append(pts)
    
    # Meridians (vertical lines)
    for i in range(n_meridians):
        a = (2.0 * math.pi * i) / n_meridians
        pts: list[float] = []
        for k in range(segments + 1):
            t = (2.0 * math.pi * k) / segments
            x = radius * math.cos(a) * math.sin(t)
            y = radius * math.sin(a) * math.sin(t)
            z = radius * flatten_z * math.cos(t)
            pts.extend([x, y, z])
        loops.append(pts)
    
    return loops


# =============================================================================
# Trail Rendering Helpers
# =============================================================================

def create_trail_segment(
    prev_pos: tuple[float, float, float],
    curr_pos: tuple[float, float, float],
    color: tuple[int, int, int, int],
    fade: float = 1.0,
) -> tuple[list[float], list[int]]:
    """
    Create vertex and color data for a single trail segment.
    
    Args:
        prev_pos: Previous position (x, y, z)
        curr_pos: Current position (x, y, z)
        color: Base color (r, g, b, a)
        fade: Fade factor (0.0 = invisible, 1.0 = full opacity)
    
    Returns:
        (vertices, colors) lists
    """
    alpha = int(round(color[3] * fade))
    alpha = max(0, min(255, alpha))
    
    vertices = [
        prev_pos[0], prev_pos[1], prev_pos[2],
        curr_pos[0], curr_pos[1], curr_pos[2],
    ]
    colors = [
        color[0], color[1], color[2], alpha,
        color[0], color[1], color[2], alpha,
    ]
    
    return vertices, colors


# =============================================================================
# Point Size Helpers
# =============================================================================

def compute_blob_point_size(
    base_size: float,
    mass: float,
    base_mass: float = 1.0,
    scale: float = 2.5,
) -> float:
    """
    Compute point size for merged particle blobs.
    
    Larger particles (higher mass) are drawn larger.
    
    Args:
        base_size: Base point size
        mass: Particle mass
        base_mass: Reference mass for scaling
        scale: Size multiplier
    
    Returns:
        Point size for rendering
    """
    if base_mass <= 0:
        base_mass = 1.0
    ratio = max(1.0, float(mass) / float(base_mass))
    # Use cube root for volume-to-radius conversion
    size_factor = ratio ** (1.0 / 3.0)
    return float(base_size) * float(scale) * size_factor


# =============================================================================
# Sprite Point Size
# =============================================================================

def compute_sprite_point_size(
    base_size: float,
    scale: float = 2.5,
) -> float:
    """
    Compute point size for soft sprite rendering.
    
    Args:
        base_size: Base point size from settings
        scale: Sprite scale multiplier
    
    Returns:
        Point size for sprite shader
    """
    return max(1.0, float(base_size) * float(scale))


# =============================================================================
# Legend Helpers
# =============================================================================

def create_legend_gradient_colors(
    width: int,
    stops: list[tuple[float, tuple[int, int, int]]],
) -> list[tuple[int, int, int]]:
    """
    Create a list of colors for a gradient legend bar.
    
    Args:
        width: Number of color samples to generate
        stops: Gradient stops
    
    Returns:
        List of (r, g, b) colors
    """
    colors: list[tuple[int, int, int]] = []
    for i in range(width):
        t = i / max(1, width - 1)
        rgba = interpolate_gradient_color(t, stops)
        colors.append((rgba[0], rgba[1], rgba[2]))
    return colors
