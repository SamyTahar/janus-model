"""
Camera management for 3D visualization.

This module provides camera utilities for the Pyglet-based renderer,
including orbital camera controls, view transformations, and multi-view support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from pyglet.math import Mat4, Vec3  # type: ignore
    except ImportError:
        Mat4 = Vec3 = object


@dataclass
class CameraState:
    """
    State of a single 3D camera.
    
    Uses an orbital camera model where the camera orbits around a center point.
    
    Attributes:
        yaw: Horizontal rotation angle in radians
        pitch: Vertical rotation angle in radians
        distance: Distance from center to camera position
        center: Point the camera looks at (as Vec3 or tuple)
        follow: Whether camera should follow a target
        follow_offset: Offset from target when following
    """
    yaw: float = 0.9
    pitch: float = 0.25
    distance: float = 1400.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    follow: bool = False
    follow_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class MultiViewState:
    """
    State for multiple camera views.
    
    Attributes:
        cameras: List of CameraState objects, one per view
        active_view: Index of currently active view for input
        view_count: Number of views to render
    """
    cameras: list[CameraState] = field(default_factory=lambda: [CameraState()])
    active_view: int = 0
    view_count: int = 1
    
    def ensure_view_count(self, count: int) -> None:
        """
        Ensure we have enough camera states for the given view count.
        
        Args:
            count: Number of views required
        """
        count = max(1, min(3, int(count)))
        self.view_count = count
        
        # Add cameras if needed
        while len(self.cameras) < count:
            if self.cameras:
                # Clone last camera
                last = self.cameras[-1]
                self.cameras.append(CameraState(
                    yaw=last.yaw,
                    pitch=last.pitch,
                    distance=last.distance,
                    center=last.center,
                    follow=False,
                    follow_offset=(0.0, 0.0, 0.0),
                ))
            else:
                self.cameras.append(CameraState())
        
        # Remove excess cameras
        if len(self.cameras) > count:
            del self.cameras[count:]
        
        # Clamp active view
        if self.active_view >= count:
            self.active_view = max(0, count - 1)
    
    def get_active_camera(self) -> CameraState:
        """Get the currently active camera state."""
        if 0 <= self.active_view < len(self.cameras):
            return self.cameras[self.active_view]
        return self.cameras[0] if self.cameras else CameraState()


def compute_camera_position(
    yaw: float,
    pitch: float,
    distance: float,
    center: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Compute camera position from orbital parameters.
    
    Args:
        yaw: Horizontal angle in radians
        pitch: Vertical angle in radians
        distance: Distance from center
        center: Point camera looks at
    
    Returns:
        (x, y, z) camera position
    """
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    
    offset_x = distance * cp * cy
    offset_y = distance * sp
    offset_z = distance * cp * sy
    
    return (
        center[0] + offset_x,
        center[1] + offset_y,
        center[2] + offset_z,
    )


def compute_camera_basis(
    yaw: float,
    pitch: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Compute camera right and up vectors for panning.
    
    Args:
        yaw: Horizontal angle in radians
        pitch: Vertical angle in radians
    
    Returns:
        (right, up) unit vectors as tuples
    """
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    
    # Forward vector (camera looks toward center)
    forward_x = -cp * cy
    forward_y = -sp
    forward_z = -cp * sy
    
    # Right = cross(forward, Y) normalized
    right_x = -forward_z
    right_y = 0.0
    right_z = forward_x
    r_len = math.sqrt(right_x**2 + right_y**2 + right_z**2)
    r_len = max(1e-6, r_len)
    right = (right_x / r_len, right_y / r_len, right_z / r_len)
    
    # Up = cross(right, forward) normalized
    up_x = right[1] * forward_z - right[2] * forward_y
    up_y = right[2] * forward_x - right[0] * forward_z
    up_z = right[0] * forward_y - right[1] * forward_x
    u_len = math.sqrt(up_x**2 + up_y**2 + up_z**2)
    u_len = max(1e-6, u_len)
    up = (up_x / u_len, up_y / u_len, up_z / u_len)
    
    return right, up


def viewport_for_view_index(
    index: int,
    view_count: int,
    window_width: int,
    window_height: int,
) -> tuple[int, int, int, int]:
    """
    Compute viewport rectangle for a given view index.
    
    Args:
        index: View index (0-based)
        view_count: Total number of views
        window_width: Total window width
        window_height: Total window height
    
    Returns:
        (x, y, width, height) viewport rectangle
    """
    if view_count <= 1:
        return 0, 0, window_width, window_height
    
    slice_w = float(window_width) / float(view_count)
    x0 = int(round(index * slice_w))
    x1 = int(round((index + 1) * slice_w))
    
    return x0, 0, max(1, x1 - x0), window_height


def view_index_from_position(
    x: int,
    view_count: int,
    window_width: int,
) -> int:
    """
    Determine which view a screen position belongs to.
    
    Args:
        x: Screen x coordinate
        view_count: Total number of views
        window_width: Total window width
    
    Returns:
        View index (0-based)
    """
    if view_count <= 1:
        return 0
    
    slice_w = max(1.0, float(window_width) / float(view_count))
    idx = int(float(x) / slice_w)
    return max(0, min(view_count - 1, idx))


def clamp_distance(
    distance: float,
    min_distance: float,
    max_distance: float,
) -> float:
    """
    Clamp camera distance to valid range.
    
    Args:
        distance: Current distance
        min_distance: Minimum allowed distance
        max_distance: Maximum allowed distance
    
    Returns:
        Clamped distance value
    """
    min_d = max(0.01, float(min_distance))
    max_d = max(min_d, float(max_distance))
    return max(min_d, min(max_d, float(distance)))


def clamp_pitch(pitch: float) -> float:
    """
    Clamp pitch angle to avoid gimbal lock.
    
    Args:
        pitch: Pitch angle in radians
    
    Returns:
        Clamped pitch in range (-π/2, π/2)
    """
    max_pitch = math.pi / 2.0 - 0.01
    return max(-max_pitch, min(max_pitch, float(pitch)))


def zoom_camera(
    state: CameraState,
    factor: float,
    min_distance: float = 0.2,
    max_distance: float = 15000.0,
) -> None:
    """
    Zoom the camera by a factor (in-place).
    
    Args:
        state: Camera state to modify
        factor: Zoom factor (< 1 = zoom in, > 1 = zoom out)
        min_distance: Minimum allowed distance
        max_distance: Maximum allowed distance
    """
    state.distance = clamp_distance(
        state.distance * factor,
        min_distance,
        max_distance,
    )


def orbit_camera(
    state: CameraState,
    delta_yaw: float,
    delta_pitch: float,
) -> None:
    """
    Orbit the camera by the given deltas (in-place).
    
    Args:
        state: Camera state to modify
        delta_yaw: Change in yaw angle
        delta_pitch: Change in pitch angle
    """
    state.yaw += delta_yaw
    state.pitch = clamp_pitch(state.pitch + delta_pitch)


def pan_camera(
    state: CameraState,
    delta_right: float,
    delta_up: float,
) -> None:
    """
    Pan the camera by the given amounts (in-place).
    
    Args:
        state: Camera state to modify
        delta_right: Movement in camera-right direction
        delta_up: Movement in camera-up direction
    """
    right, up = compute_camera_basis(state.yaw, state.pitch)
    
    cx, cy, cz = state.center
    cx += right[0] * delta_right + up[0] * delta_up
    cy += right[1] * delta_right + up[1] * delta_up
    cz += right[2] * delta_right + up[2] * delta_up
    state.center = (cx, cy, cz)
