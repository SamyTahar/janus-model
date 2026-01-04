"""
Configuration constants and parameter groups for the Janus 3D simulation.

This module centralizes parameter categorization, menu groups, and dependency mappings.
"""

from __future__ import annotations


# =============================================================================
# Parameter Reset Keys - Changes that require simulation reset
# =============================================================================

RESET_KEYS = {
    "init_mode",
    "particle_count",
    "positive_count",
    "negative_count",
    "negative_fraction",
    "population_mode",
    "void_radius",
    "galaxy_radius",
    "galaxy_scale_length",
    "galaxy_thickness",
    "galaxy_vmax",
    "galaxy_turnover",
    "galaxy_sigma_v",
    "negative_clump_count",
    "negative_clump_sigma",
    "negative_sigma_v",
    "negative_vphi_scale",
    "negative_on_boundary",
}


# =============================================================================
# Parameter Categories
# =============================================================================

BOUNDS_KEYS = {
    "bounds",
    "bound_sphere_radius",
    "bound_sphere_flatten_z",
    "bound_mode",
    "bounds_enabled",
}

GRADIENT_KEYS = {
    "color_gradient",
    "color_gradient_mode",
}

TRAIL_KEYS = {
    "trails_enabled",
    "trails_length",
    "trails_stride",
    "trails_alpha",
    "trails_pos_only",
    "trails_blur",
    "trails_width",
}


# =============================================================================
# Dependency Keys - Parameters with conditional availability
# =============================================================================

JANUS_DEP_KEYS = {
    "janus_g",
    "force_backend",
    "force_tile_size",
    "force_debug",
    "softening",
    "theta",
}

BOUND_DEP_KEYS = {
    "bound_mode",
    "bounds",
    "bound_sphere_radius",
    "bound_sphere_flatten_z",
    "bound_wire_visible",
    "bound_wire_opacity",
    "negative_on_boundary",
    "negative_static_on_boundary",
    "bounce",
}


# =============================================================================
# Menu Groups - Organizes parameters in the UI menu
# =============================================================================

MENU_GROUPS = {
    "init_mode": ("Init", "General"),
    "time_scale": ("Init", "Time"),
    "janus_enabled": ("Forces", "Core"),
    "janus_g": ("Forces", "Core"),
    "force_backend": ("Forces", "Backend"),
    "force_tile_size": ("Forces", "Backend"),
    "force_debug": ("Forces", "Backend"),
    "softening": ("Forces", "Tuning"),
    "theta": ("Forces", "Tuning"),
    "split_screen": ("View", "Layout"),
    "multi_view": ("View", "Layout"),
    "multi_view_count": ("View", "Layout"),
    "point_size": ("View", "Points"),
    "camera_min_distance": ("View", "Camera"),
    "camera_max_distance": ("View", "Camera"),
    "color_gradient": ("View", "Color"),
    "color_gradient_mode": ("View", "Color"),
    "sprite_enabled": ("View", "Sprites"),
    "sprite_scale": ("View", "Sprites"),
    "grid_enabled": ("View", "Grid"),
    "grid_step": ("View", "Grid"),
    "trails_enabled": ("View", "Trails"),
    "trails_length": ("View", "Trails"),
    "trails_stride": ("View", "Trails"),
    "trails_alpha": ("View", "Trails"),
    "trails_pos_only": ("View", "Trails"),
    "trails_blur": ("View", "Trails"),
    "trails_width": ("View", "Trails"),
    "bounds_enabled": ("Bounds", "Core"),
    "bound_mode": ("Bounds", "Shape"),
    "bounds": ("Bounds", "Box"),
    "bound_sphere_radius": ("Bounds", "Sphere"),
    "bound_sphere_flatten_z": ("Bounds", "Sphere"),
    "bound_wire_visible": ("Bounds", "Sphere"),
    "bound_wire_opacity": ("Bounds", "Sphere"),
    "negative_on_boundary": ("Bounds", "Sphere"),
    "negative_static_on_boundary": ("Bounds", "Sphere"),
    "particle_count": ("Populations", "Total"),
    "positive_count": ("Populations", "Explicit"),
    "negative_count": ("Populations", "Explicit"),
    "negative_fraction": ("Populations", "Total"),
    "merge_enabled": ("Populations", "Merge"),
    "merge_radius": ("Populations", "Merge"),
    "merge_min_count": ("Populations", "Merge"),
    "merge_mode": ("Populations", "Merge"),
    "merge_max_cells": ("Populations", "Merge"),
    "merge_temp_threshold": ("Populations", "Merge"),
    "merge_blob_scale": ("Populations", "Merge"),
    "population_mode": ("Populations", "Mode"),
    "mass_positive": ("Populations", "Mass"),
    "mass_negative": ("Populations", "Mass"),
    "void_radius": ("Galaxy M+", "Shape"),
    "galaxy_radius": ("Galaxy M+", "Shape"),
    "galaxy_scale_length": ("Galaxy M+", "Shape"),
    "galaxy_thickness": ("Galaxy M+", "Shape"),
    "galaxy_vmax": ("Galaxy M+", "Kinematics"),
    "galaxy_turnover": ("Galaxy M+", "Kinematics"),
    "galaxy_sigma_v": ("Galaxy M+", "Kinematics"),
    "negative_clump_count": ("Env M-", "Clumps"),
    "negative_clump_sigma": ("Env M-", "Clumps"),
    "negative_sigma_v": ("Env M-", "Kinematics"),
    "negative_vphi_scale": ("Env M-", "Kinematics"),
    "damping": ("Integrator", "Dynamics"),
    "bounce": ("Integrator", "Dynamics"),
    "max_speed": ("Integrator", "Dynamics"),
}


# =============================================================================
# Parameter Hints - Help text for each parameter
# =============================================================================

PARAM_HINTS = {
    "init_mode": "Init distribution: janus_galaxy or random.",
    "time_scale": "Simulated time multiplier.",
    "janus_enabled": "Toggle Janus forces.",
    "janus_g": "Gravity constant G.",
    "force_backend": "Force backend: cpu (Barnes-Hut), metal (GPU), or cpu_direct (O(N^2)).",
    "force_tile_size": "Tile size for metal/cpu_direct kernels.",
    "force_debug": "Print extra diagnostics for force backends.",
    "color_gradient": "M+ gradient: mix of speed/proximity/density/force.",
    "color_gradient_mode": "Gradient mode: mix/speed/force/density/proximity/temperature.",
    "split_screen": "Split screen M+ / M- (2 views).",
    "multi_view": "Multi-view cameras (same scene).",
    "multi_view_count": "Number of independent views (1..3).",
    "sprite_enabled": "Soft sprites instead of hard points.",
    "sprite_scale": "Sprite size multiplier.",
    "grid_enabled": "Show an XY grid.",
    "grid_step": "Grid spacing.",
    "camera_min_distance": "Minimum zoom distance.",
    "camera_max_distance": "Maximum zoom distance.",
    "trails_enabled": "Show particle trails.",
    "trails_length": "Number of stored frames for trails.",
    "trails_stride": "Draw 1 in N particles for trails.",
    "trails_alpha": "Trail opacity (0..1).",
    "trails_pos_only": "Trails only for M+ particles.",
    "trails_blur": "Soft/blurred trails (line smoothing).",
    "trails_width": "Trail width.",
    "softening": "Softening length (eps).",
    "theta": "Barnes-Hut opening angle.",
    "bounds_enabled": "Enable boundary collisions.",
    "bound_mode": "Boundary shape: box or sphere.",
    "bounds": "Box half-size.",
    "bound_sphere_radius": "Sphere radius.",
    "bound_sphere_flatten_z": "Sphere flattening in Z.",
    "bound_wire_visible": "Show wireframe (sphere).",
    "bound_wire_opacity": "Wireframe alpha.",
    "negative_on_boundary": "Place M- on sphere surface.",
    "negative_static_on_boundary": "Freeze M- velocities on boundary.",
    "particle_count": "Total particles (population_mode=total).",
    "positive_count": "M+ count (population_mode=explicit).",
    "negative_count": "M- count (population_mode=explicit).",
    "negative_fraction": "Fraction of M- particles (population_mode=total).",
    "merge_enabled": "Merge particles in dense regions.",
    "merge_radius": "Merge cell size (world units).",
    "merge_min_count": "Min particles per cell to merge.",
    "merge_mode": "Merge mode: all or M+ only.",
    "merge_max_cells": "Max merged cells per frame (0 = unlimited).",
    "merge_temp_threshold": "Temperature threshold for merging (0 = off).",
    "merge_blob_scale": "Size multiplier for merged blobs.",
    "population_mode": "Count mode: total (particle_count + negative_fraction) or explicit (positive_count + negative_count).",
    "mass_positive": "Mass per M+ particle.",
    "mass_negative": "Mass per M- particle.",
    "void_radius": "Void radius (Janus init).",
    "galaxy_radius": "Disk radius.",
    "galaxy_scale_length": "Disk scale length.",
    "galaxy_thickness": "Disk thickness (Z).",
    "galaxy_vmax": "Max rotation speed.",
    "galaxy_turnover": "Rotation turnover radius.",
    "galaxy_sigma_v": "Velocity dispersion (M+).",
    "negative_clump_count": "Number of M- clumps.",
    "negative_clump_sigma": "Clump size.",
    "negative_sigma_v": "Velocity dispersion (M-).",
    "negative_vphi_scale": "Rotation scale for M- (0=off, 1=same as M+).",
    "damping": "Velocity damping per tick.",
    "bounce": "Boundary restitution.",
    "max_speed": "Speed clamp.",
    "point_size": "Point size (pixels).",
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_menu_group_title(key: str) -> tuple[str | None, str | None]:
    """
    Get the menu group and subgroup for a parameter.
    
    Args:
        key: Parameter key name
    
    Returns:
        (group_name, subgroup_name) or (None, None) if not found
    """
    group = MENU_GROUPS.get(key)
    if group is None:
        return None, None
    if isinstance(group, tuple):
        return group
    if isinstance(group, str) and "/" in group:
        main, sub = group.split("/", 1)
        return main.strip(), sub.strip()
    return str(group), None


def get_param_hint(key: str) -> str:
    """
    Get the hint/description for a parameter.
    
    Args:
        key: Parameter key name
    
    Returns:
        Hint text or empty string
    """
    return PARAM_HINTS.get(key, "")


def is_reset_required(key: str) -> bool:
    """Check if changing this parameter requires simulation reset."""
    return key in RESET_KEYS


def is_bounds_related(key: str) -> bool:
    """Check if this parameter is bounds-related."""
    return key in BOUNDS_KEYS or key in BOUND_DEP_KEYS


def is_trail_related(key: str) -> bool:
    """Check if this parameter is trail-related."""
    return key in TRAIL_KEYS


def is_gradient_related(key: str) -> bool:
    """Check if this parameter is gradient/color-related."""
    return key in GRADIENT_KEYS
