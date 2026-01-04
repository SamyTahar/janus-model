"""
Camera controller for 3D rendering with multi-view support.

This module manages camera states, view transformations, and user input
for orbital camera controls in the Janus 3D simulation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from pyglet.math import Vec3  # type: ignore
    except ImportError:
        Vec3 = object


class CameraController:
    """
    Manages multiple camera views for 3D rendering.
    
    Supports orbital camera controls (yaw, pitch, distance) and multi-view
    rendering for split-screen visualization.
    """
    
    def __init__(self, initial_count: int = 1):
        """
        Initialize camera controller.
        
        Args:
            initial_count: Number of camera views to create initially
        """
        self.view_count = max(1, initial_count)
        self.active_view = 0
        
        # Camera state for each view
        self.camera_yaw: list[float] = []
        self.camera_pitch: list[float] = []
        self.camera_distance: list[float] = []
        self.camera_center: list[object] = []  # Vec3
        self.camera_follow: list[bool] = []
        self.camera_follow_offset: list[object] = []  # Vec3
        
        # Initialize default camera
        self._ensure_view_count(self.view_count)
    
    def _ensure_view_count(self, count: int) -> None:
        """Ensure we have camera state for the given number of views."""
        from pyglet.math import Vec3  # type: ignore
        
        count = max(1, int(count))
        self.view_count = count
        
        # Add cameras if needed
        while len(self.camera_yaw) < count:
            if self.camera_yaw:
                # Clone last camera
                self.camera_yaw.append(self.camera_yaw[-1])
                self.camera_pitch.append(self.camera_pitch[-1])
                self.camera_distance.append(self.camera_distance[-1])
                self.camera_center.append(self.camera_center[-1])
                self.camera_follow.append(False)
                self.camera_follow_offset.append(Vec3(0.0, 0.0, 0.0))
            else:
                # Create first camera with defaults
                self.camera_yaw.append(0.9)
                self.camera_pitch.append(0.25)
                self.camera_distance.append(1400.0)
                self.camera_center.append(Vec3(0.0, 0.0, 0.0))
                self.camera_follow.append(False)
                self.camera_follow_offset.append(Vec3(0.0, 0.0, 0.0))
        
        # Clamp active view
        if self.active_view >= count:
            self.active_view = max(0, count - 1)
    
    def set_view_count(self, count: int) -> None:
        """Update the number of views."""
        self._ensure_view_count(count)
    
    def get_active_camera(self) -> tuple[float, float, float, object]:
        """
        Get active camera parameters.
        
        Returns:
            (yaw, pitch, distance, center) tuple
        """
        idx = self.active_view
        return (
            self.camera_yaw[idx],
            self.camera_pitch[idx],
            self.camera_distance[idx],
            self.camera_center[idx],
        )
    
    def orbit(self, view_idx: int, delta_yaw: float, delta_pitch: float) -> None:
        """
        Orbit camera around center point.
        
        Args:
            view_idx: Index of view to modify
            delta_yaw: Change in yaw angle (radians)
            delta_pitch: Change in pitch angle (radians)
        """
        if 0 <= view_idx < len(self.camera_yaw):
            self.camera_yaw[view_idx] += delta_yaw
            
            # Clamp pitch to avoid gimbal lock
            new_pitch = self.camera_pitch[view_idx] + delta_pitch
            max_pitch = math.pi / 2.0 - 0.01
            self.camera_pitch[view_idx] = max(-max_pitch, min(max_pitch, new_pitch))
    
    def zoom(self, view_idx: int, factor: float, min_dist: float = 0.2, max_dist: float = 15000.0) -> None:
        """
        Zoom camera in or out.
        
        Args:
            view_idx: Index of view to modify
            factor: Zoom factor (< 1 = zoom in, > 1 = zoom out)
            min_dist: Minimum distance
            max_dist: Maximum distance
        """
        if 0 <= view_idx < len(self.camera_distance):
            new_dist = self.camera_distance[view_idx] * factor
            self.camera_distance[view_idx] = max(min_dist, min(max_dist, new_dist))
    
    def pan(self, view_idx: int, delta_right: float, delta_up: float) -> None:
        """
        Pan camera (move center point).
        
        Args:
            view_idx: Index of view to modify
            delta_right: Movement in camera-right direction
            delta_up: Movement in camera-up direction
        """
        from pyglet.math import Vec3  # type: ignore
        
        if 0 <= view_idx < len(self.camera_center):
            # Compute camera basis
            yaw = self.camera_yaw[view_idx]
            pitch = self.camera_pitch[view_idx]
            
            cp = math.cos(pitch)
            sp = math.sin(pitch)
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            
            # Forward vector
            forward_x = -cp * cy
            forward_y = -sp
            forward_z = -cp * sy
            
            # Right = cross(forward, Y)
            right_x = -forward_z
            right_y = 0.0
            right_z = forward_x
            r_len = math.sqrt(right_x**2 + right_y**2 + right_z**2)
            r_len = max(1e-6, r_len)
            right_x /= r_len
            right_y /= r_len
            right_z /= r_len
            
            # Up = cross(right, forward)
            up_x = right_y * forward_z - right_z * forward_y
            up_y = right_z * forward_x - right_x * forward_z
            up_z = right_x * forward_y - right_y * forward_x
            u_len = math.sqrt(up_x**2 + up_y**2 + up_z**2)
            u_len = max(1e-6, u_len)
            up_x /= u_len
            up_y /= u_len
            up_z /= u_len
            
            # Update center
            center = self.camera_center[view_idx]
            new_center = Vec3(
                center.x + right_x * delta_right + up_x * delta_up,
                center.y + right_y * delta_right + up_y * delta_up,
                center.z + right_z * delta_right + up_z * delta_up,
            )
            self.camera_center[view_idx] = new_center
    
    def set_center(self, view_idx: int, x: float, y: float, z: float) -> None:
        """Set camera center point."""
        from pyglet.math import Vec3  # type: ignore
        
        if 0 <= view_idx < len(self.camera_center):
            self.camera_center[view_idx] = Vec3(x, y, z)
    
    def view_index_from_x(self, x: int, window_width: int) -> int:
        """Determine which view a screen x-coordinate belongs to."""
        if self.view_count <= 1:
            return 0
        
        slice_w = max(1.0, float(window_width) / float(self.view_count))
        idx = int(float(x) / slice_w)
        return max(0, min(self.view_count - 1, idx))
    
    def viewport_for_view(self, view_idx: int, window_width: int, window_height: int) -> tuple[int, int, int, int]:
        """
        Get viewport rectangle for a view.
        
        Returns:
            (x, y, width, height) tuple
        """
        if self.view_count <= 1:
            return 0, 0, window_width, window_height
        
        slice_w = float(window_width) / float(self.view_count)
        x0 = int(round(view_idx * slice_w))
        x1 = int(round((view_idx + 1) * slice_w))
        
        return x0, 0, max(1, x1 - x0), window_height
