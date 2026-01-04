"""
Color mapping and gradient utilities for 3D particle visualization.

This module provides color gradient computation for particles based on
various metrics like speed, force, density, proximity, and temperature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particle_sim3d.core.sim import Particle3D


# =============================================================================
# Default Colors and Gradients
# =============================================================================

POS_BASE_COLOR = (188, 214, 255, 220)
NEG_BASE_COLOR = (255, 184, 107, 220)

# Default gradient stops: cold (blue) -> neutral (white) -> warm (orange) -> hot (red)
DEFAULT_GRADIENT_STOPS = [
    (0.0, (40, 100, 190)),
    (0.45, (245, 245, 245)),
    (0.75, (255, 170, 90)),
    (1.0, (230, 60, 50)),
]

# Gradient mode labels
GRADIENT_LABELS = {
    "mix": "M+ gradient",
    "speed": "M+ speed",
    "force": "M+ force",
    "density": "M+ density",
    "proximity": "M+ proximity",
    "temperature": "M+ temperature",
}

# Default parameters
GRAD_STAT_ALPHA = 0.02  # EMA smoothing for gradient stats
GRAD_CELL_DIV = 30.0    # Grid divisions for density calculation


# =============================================================================
# Gradient State
# =============================================================================

@dataclass
class GradientStats:
    """
    Running statistics for gradient normalization.
    
    Uses exponential moving average to smooth min/max values.
    """
    stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    alpha: float = GRAD_STAT_ALPHA
    
    def reset(self) -> None:
        """Clear all statistics."""
        self.stats.clear()
    
    def update(self, key: str, cur_min: float, cur_max: float) -> tuple[float, float]:
        """
        Update and return smoothed min/max for a metric.
        
        Args:
            key: Metric key name
            cur_min: Current minimum value
            cur_max: Current maximum value
        
        Returns:
            (smoothed_min, smoothed_max)
        """
        if not (math.isfinite(cur_min) and math.isfinite(cur_max)):
            cur_min, cur_max = 0.0, 1.0
        if cur_min > cur_max:
            cur_min, cur_max = cur_max, cur_min
        
        prev = self.stats.get(key)
        if prev is None:
            self.stats[key] = (cur_min, cur_max)
            return cur_min, cur_max
        
        prev_min, prev_max = prev
        if not math.isfinite(prev_min):
            prev_min = cur_min
        if not math.isfinite(prev_max):
            prev_max = cur_max
        
        # Expand range immediately, contract slowly
        if cur_min < prev_min:
            new_min = cur_min
        else:
            new_min = (1.0 - self.alpha) * prev_min + self.alpha * cur_min
        
        if cur_max > prev_max:
            new_max = cur_max
        else:
            new_max = (1.0 - self.alpha) * prev_max + self.alpha * cur_max
        
        if (new_max - new_min) < 1e-9:
            new_max = new_min + 1e-9
        
        self.stats[key] = (new_min, new_max)
        return new_min, new_max


# =============================================================================
# Color Utilities
# =============================================================================

def gradient_color(
    t: float,
    stops: list[tuple[float, tuple[int, int, int]]] | None = None,
    alpha: int = 220,
) -> tuple[int, int, int, int]:
    """
    Interpolate a color from a gradient.
    
    Args:
        t: Position on gradient (0.0 to 1.0)
        stops: Gradient stops or None for default
        alpha: Output alpha value
    
    Returns:
        (r, g, b, a) color tuple
    """
    if stops is None:
        stops = DEFAULT_GRADIENT_STOPS
    
    t = max(0.0, min(1.0, t))
    
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            local = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            r = int(round(c0[0] + (c1[0] - c0[0]) * local))
            g = int(round(c0[1] + (c1[1] - c0[1]) * local))
            b = int(round(c0[2] + (c1[2] - c0[2]) * local))
            return r, g, b, alpha
    
    r, g, b = stops[-1][1]
    return r, g, b, alpha


def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to 0-1 range.
    
    Args:
        value: Value to normalize
        min_val: Minimum of range
        max_val: Maximum of range
    
    Returns:
        Normalized value in [0, 1]
    """
    if not math.isfinite(value):
        return 0.0
    span = max_val - min_val
    if span <= 1e-9:
        return 0.5
    t = (value - min_val) / span
    return max(0.0, min(1.0, t))


# =============================================================================
# Gradient Computation
# =============================================================================

class ColorMapper:
    """
    Computes particle colors based on various metrics.
    
    Supports multiple gradient modes: mix, speed, force, density,
    proximity, and temperature.
    """
    
    def __init__(
        self,
        stops: list[tuple[float, tuple[int, int, int]]] | None = None,
        cell_div: float = GRAD_CELL_DIV,
    ):
        """
        Initialize the color mapper.
        
        Args:
            stops: Gradient color stops
            cell_div: Grid divisions for density
        """
        self._stops = stops or DEFAULT_GRADIENT_STOPS
        self._cell_div = cell_div
        self._stats = GradientStats()
    
    def reset_stats(self) -> None:
        """Reset gradient statistics."""
        self._stats.reset()
    
    def get_stats(self) -> dict[str, tuple[float, float]]:
        """Get current gradient statistics."""
        return self._stats.stats.copy()
    
    def compute_mplus_gradient(
        self,
        particles: list["Particle3D"],
        accel_mag: list[float],
        *,
        mode: str = "mix",
        center: tuple[float, float, float] | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[tuple[int, int, int, int]] | None:
        """
        Compute gradient colors for M+ particles.
        
        Args:
            particles: List of all particles
            accel_mag: Acceleration magnitudes per particle
            mode: Gradient mode (mix/speed/force/density/proximity/temperature)
            center: Center point for calculations or None to compute
            weights: Weight dict for mix mode
        
        Returns:
            List of RGBA colors indexed by particle index, or None if no M+
        """
        n = len(particles)
        if n == 0:
            return None
        
        pos_indices = [i for i, pt in enumerate(particles) if pt.s > 0]
        if not pos_indices:
            return None
        
        # Compute center
        if center is None:
            cx = cy = cz = 0.0
            for i in pos_indices:
                pt = particles[i]
                cx += pt.x
                cy += pt.y
                cz += pt.z
            cx /= len(pos_indices)
            cy /= len(pos_indices)
            cz /= len(pos_indices)
        else:
            cx, cy, cz = center
        
        # Compute max radius
        r_max = 0.0
        for i in pos_indices:
            pt = particles[i]
            dx = pt.x - cx
            dy = pt.y - cy
            r = math.hypot(dx, dy)
            if r > r_max:
                r_max = r
        r_max = max(r_max, 1e-6)
        
        # Setup density grid
        cell_size = max(1.0, r_max / self._cell_div)
        inv_cell = 1.0 / cell_size
        cell_counts: dict[tuple[int, int, int], int] = {}
        cell_keys: list[tuple[int, int, int]] = []
        
        for i in pos_indices:
            pt = particles[i]
            key = (
                int(math.floor((pt.x - cx) * inv_cell)),
                int(math.floor((pt.y - cy) * inv_cell)),
                int(math.floor((pt.z - cz) * inv_cell)),
            )
            cell_keys.append(key)
            cell_counts[key] = cell_counts.get(key, 0) + 1
        
        # Compute metrics
        speed_vals: list[float] = []
        force_vals: list[float] = []
        dens_vals: list[float] = []
        prox_vals: list[float] = []
        temp_vals: list[float] = []
        
        for idx, key in zip(pos_indices, cell_keys):
            pt = particles[idx]
            v2 = pt.vx**2 + pt.vy**2 + pt.vz**2
            speed = math.sqrt(v2)
            speed_vals.append(math.log1p(speed))
            temp_vals.append(math.log1p(max(0.0, float(pt.m) * v2)))
            
            force = 0.0
            if idx < len(accel_mag):
                force = accel_mag[idx]
                if not math.isfinite(force):
                    force = 0.0
            force_vals.append(math.log1p(abs(force)))
            
            dens_vals.append(math.log1p(cell_counts[key]))
            
            dx = pt.x - cx
            dy = pt.y - cy
            r = math.hypot(dx, dy)
            prox = max(0.0, min(1.0, 1.0 - (r / r_max)))
            prox_vals.append(prox)
        
        # Get normalized ranges
        speed_min, speed_max = self._stats.update("grad_speed", min(speed_vals), max(speed_vals))
        force_min, force_max = self._stats.update("grad_force", min(force_vals), max(force_vals))
        dens_min, dens_max = self._stats.update("grad_density", min(dens_vals), max(dens_vals))
        temp_min, temp_max = self._stats.update("grad_temp", min(temp_vals), max(temp_vals))
        prox_min, prox_max = self._stats.update("grad_prox", min(prox_vals), max(prox_vals))
        
        # Get weights
        if weights is None:
            weights = {}
        w_speed = max(0.0, weights.get("speed", 1.0))
        w_force = max(0.0, weights.get("force", 1.0))
        w_dens = max(0.0, weights.get("density", 1.0))
        w_prox = max(0.0, weights.get("proximity", 1.0))
        w_sum = w_speed + w_force + w_dens + w_prox
        if w_sum <= 1e-9:
            w_speed = w_force = w_dens = w_prox = 1.0
            w_sum = 4.0
        
        # Compute raw values
        mode = str(mode).strip().lower()
        raw_vals: list[float] = []
        
        for i in range(len(pos_indices)):
            t_speed = normalize(speed_vals[i], speed_min, speed_max)
            t_force = normalize(force_vals[i], force_min, force_max)
            t_dens = normalize(dens_vals[i], dens_min, dens_max)
            t_temp = normalize(temp_vals[i], temp_min, temp_max)
            t_prox = prox_vals[i]
            
            if mode == "speed":
                raw_vals.append(t_speed)
            elif mode == "force":
                raw_vals.append(t_force)
            elif mode == "density":
                raw_vals.append(t_dens)
            elif mode == "proximity":
                raw_vals.append(t_prox)
            elif mode == "temperature":
                raw_vals.append(t_temp)
            else:  # mix
                raw_vals.append((w_speed * t_speed + w_force * t_force + 
                                w_dens * t_dens + w_prox * t_prox) / w_sum)
        
        # Normalize to final range
        raw_min, raw_max = self._stats.update("grad_mode_mix", min(raw_vals), max(raw_vals))
        
        # Build color list
        result: list[tuple[int, int, int, int] | None] = [None] * n
        for i, idx in enumerate(pos_indices):
            t = normalize(raw_vals[i], raw_min, raw_max)
            result[idx] = gradient_color(t, self._stops)
        
        return result  # type: ignore


def format_gradient_value(value: float) -> str:
    """Format a gradient value for display."""
    if not math.isfinite(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 1000.0 or abs(value) < 0.01:
        return f"{value:.3g}"
    return f"{value:.2f}"


def get_gradient_range_text(stats: GradientStats, mode: str) -> str | None:
    """
    Get formatted range text for a gradient mode.
    
    Args:
        stats: Gradient statistics
        mode: Gradient mode
    
    Returns:
        Formatted range string or None
    """
    mode = str(mode or "mix").strip().lower()
    
    key_map = {
        "speed": ("grad_speed", math.expm1),
        "force": ("grad_force", math.expm1),
        "density": ("grad_density", math.expm1),
        "proximity": ("grad_prox", None),
        "temperature": ("grad_temp", math.expm1),
    }
    
    key, convert = key_map.get(mode, ("grad_mode_mix", None))
    
    if key not in stats.stats:
        return None
    
    vmin, vmax = stats.stats[key]
    if convert is not None:
        vmin = max(0.0, float(convert(max(0.0, vmin))))
        vmax = max(0.0, float(convert(max(0.0, vmax))))
    
    return f"min {format_gradient_value(vmin)}  max {format_gradient_value(vmax)}"
