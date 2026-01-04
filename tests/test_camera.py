"""
Tests for camera module.
"""

import pytest
import math
from particle_sim3d.camera import (
    CameraState,
    MultiViewState,
    compute_camera_position,
    compute_camera_basis,
    viewport_for_view_index,
    zoom_camera,
    orbit_camera,
    pan_camera,
)


class TestCameraState:
    """Tests for CameraState dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        state = CameraState()
        assert state.yaw == 0.9
        assert state.pitch == 0.25
        assert state.distance == 1400.0
        assert state.center == (0.0, 0.0, 0.0)
        assert state.follow is False
    
    def test_custom_values(self):
        """Test custom initialization."""
        state = CameraState(yaw=1.0, pitch=0.5, distance=500.0)
        assert state.yaw == 1.0
        assert state.pitch == 0.5
        assert state.distance == 500.0


class TestMultiViewState:
    """Tests for MultiViewState dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        state = MultiViewState()
        assert state.view_count == 1  # Default is 1
        assert state.active_view == 0


class TestComputeCameraPosition:
    """Tests for compute_camera_position function."""
    
    def test_zero_angles(self):
        """Test with zero yaw and pitch."""
        eye = compute_camera_position(yaw=0.0, pitch=0.0, distance=100.0, center=(0, 0, 0))
        assert len(eye) == 3
        # At yaw=0, pitch=0, should be on positive X axis
        assert abs(eye[0] - 100.0) < 0.01
        assert abs(eye[1]) < 0.01
        assert abs(eye[2]) < 0.01
    
    def test_origin_center(self):
        """Test with explicit center at origin."""
        eye = compute_camera_position(yaw=0.0, pitch=0.0, distance=100.0, center=(0, 0, 0))
        assert len(eye) == 3
    
    def test_offset_center(self):
        """Test with offset center."""
        eye = compute_camera_position(yaw=0.0, pitch=0.0, distance=100.0, center=(50, 50, 50))
        # Eye should be displaced from center
        assert abs(eye[0] - 150.0) < 0.01
        assert abs(eye[1] - 50.0) < 0.01
        assert abs(eye[2] - 50.0) < 0.01
    
    def test_distance_scaling(self):
        """Test that distance scales the eye position."""
        eye1 = compute_camera_position(yaw=0.0, pitch=0.0, distance=100.0, center=(0, 0, 0))
        eye2 = compute_camera_position(yaw=0.0, pitch=0.0, distance=200.0, center=(0, 0, 0))
        
        # Second eye should be twice as far from origin
        dist1 = math.sqrt(sum(c**2 for c in eye1))
        dist2 = math.sqrt(sum(c**2 for c in eye2))
        assert abs(dist2 / dist1 - 2.0) < 0.01


class TestComputeCameraBasis:
    """Tests for compute_camera_basis function."""
    
    def test_returns_two_vectors(self):
        """Test that it returns right, up."""
        right, up = compute_camera_basis(yaw=0.0, pitch=0.0)
        
        assert len(right) == 3
        assert len(up) == 3
    
    def test_vectors_are_normalized(self):
        """Test that vectors have unit length."""
        right, up = compute_camera_basis(yaw=0.5, pitch=0.3)
        
        def length(v):
            return math.sqrt(sum(c**2 for c in v))
        
        assert abs(length(right) - 1.0) < 0.01
        assert abs(length(up) - 1.0) < 0.01


class TestViewportForViewIndex:
    """Tests for viewport_for_view_index function."""
    
    def test_single_view(self):
        """Test single viewport."""
        vp = viewport_for_view_index(
            index=0,
            view_count=1,
            window_width=800,
            window_height=600
        )
        
        assert vp == (0, 0, 800, 600)
    
    def test_two_views_horizontal(self):
        """Test two horizontal viewports."""
        vp0 = viewport_for_view_index(0, 2, 800, 600)
        vp1 = viewport_for_view_index(1, 2, 800, 600)
        
        # Each should be half width
        assert vp0[2] == 400  # width
        assert vp1[2] == 400
        
        # First on left, second on right
        assert vp0[0] == 0
        assert vp1[0] == 400


class TestZoomCamera:
    """Tests for zoom_camera function."""
    
    def test_zoom_in(self):
        """Test zooming in reduces distance."""
        state = CameraState(distance=1000.0)
        old_dist = state.distance
        zoom_camera(state, factor=0.5)
        
        assert state.distance < old_dist
    
    def test_zoom_out(self):
        """Test zooming out increases distance."""
        state = CameraState(distance=1000.0)
        old_dist = state.distance
        zoom_camera(state, factor=2.0)
        
        assert state.distance > old_dist
    
    def test_zoom_respects_limits(self):
        """Test zoom respects min/max limits."""
        state = CameraState(distance=100.0)
        
        # Try to zoom in past minimum
        zoom_camera(state, factor=0.001, min_distance=50.0)
        assert state.distance >= 50.0
        
        # Reset and try to zoom out past maximum
        state.distance = 100.0
        zoom_camera(state, factor=1000.0, max_distance=200.0)
        assert state.distance <= 200.0


class TestOrbitCamera:
    """Tests for orbit_camera function."""
    
    def test_orbit_yaw(self):
        """Test orbiting changes yaw."""
        state = CameraState(yaw=0.0, pitch=0.0)
        orbit_camera(state, delta_yaw=0.5, delta_pitch=0.0)
        
        assert state.yaw == 0.5
        assert state.pitch == 0.0
    
    def test_orbit_pitch(self):
        """Test orbiting changes pitch."""
        state = CameraState(yaw=0.0, pitch=0.0)
        orbit_camera(state, delta_yaw=0.0, delta_pitch=0.3)
        
        assert state.yaw == 0.0
        assert state.pitch == 0.3


class TestPanCamera:
    """Tests for pan_camera function."""
    
    def test_pan_moves_center(self):
        """Test panning moves the center."""
        state = CameraState(center=(0.0, 0.0, 0.0))
        pan_camera(state, delta_right=10.0, delta_up=5.0)
        
        # Center should have moved
        assert state.center != (0.0, 0.0, 0.0)
