"""
Tests for drawing module.
"""

import pytest
import math
from particle_sim3d.drawing import (
    create_grid_vertices,
    legend_color_at,
    compute_blob_point_size,
    compute_sprite_point_size,
)


class TestCreateGridVertices:
    """Tests for create_grid_vertices function."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        vertices = create_grid_vertices(size=100.0, step=10.0)
        assert isinstance(vertices, list)
    
    def test_non_empty(self):
        """Test that grid has vertices."""
        vertices = create_grid_vertices(size=100.0, step=10.0)
        assert len(vertices) > 0
    
    def test_vertex_count_multiple_of_6(self):
        """Test vertices are multiple of 6 (2 points * 3 coords per line)."""
        vertices = create_grid_vertices(size=100.0, step=10.0)
        assert len(vertices) % 6 == 0
    
    def test_custom_z(self):
        """Test with custom z plane."""
        vertices = create_grid_vertices(size=100.0, step=10.0, z=50.0)
        
        # Check that all z coordinates are 50
        z_coords = vertices[2::3]  # Every 3rd element starting from index 2
        for z in z_coords:
            assert z == 50.0
    
    def test_small_step(self):
        """Test with small step produces more lines."""
        v1 = create_grid_vertices(size=100.0, step=50.0)
        v2 = create_grid_vertices(size=100.0, step=10.0)
        
        # Smaller step should produce more vertices
        assert len(v2) > len(v1)


class TestLegendColorAt:
    """Tests for legend_color_at function."""
    
    def test_returns_rgb(self):
        """Test that function returns 3-tuple (rgb)."""
        color = legend_color_at(0.5, [(0.0, (255, 0, 0)), (1.0, (0, 0, 255))])
        assert len(color) == 3
    
    def test_at_zero(self):
        """Test color at t=0 matches first stop."""
        stops = [(0.0, (255, 0, 0)), (1.0, (0, 0, 255))]
        color = legend_color_at(0.0, stops)
        assert color[0] == 255  # Red component
        assert color[2] == 0    # Blue component
    
    def test_at_one(self):
        """Test color at t=1 matches last stop."""
        stops = [(0.0, (255, 0, 0)), (1.0, (0, 0, 255))]
        color = legend_color_at(1.0, stops)
        assert color[0] == 0    # Red component
        assert color[2] == 255  # Blue component
    
    def test_interpolation(self):
        """Test color interpolation at midpoint."""
        stops = [(0.0, (0, 0, 0)), (1.0, (200, 200, 200))]
        color = legend_color_at(0.5, stops)
        
        # Should be around (100, 100, 100)
        assert 90 <= color[0] <= 110
        assert 90 <= color[1] <= 110
        assert 90 <= color[2] <= 110


class TestComputeBlobPointSize:
    """Tests for compute_blob_point_size function."""
    
    def test_basic_size(self):
        """Test basic size calculation."""
        size = compute_blob_point_size(
            base_size=5.0,
            mass=1.0,
            base_mass=1.0,
            scale=2.0
        )
        assert size > 0
    
    def test_larger_mass_larger_size(self):
        """Test that larger mass gives larger size."""
        size1 = compute_blob_point_size(base_size=5.0, mass=1.0, base_mass=1.0, scale=2.0)
        size2 = compute_blob_point_size(base_size=5.0, mass=10.0, base_mass=1.0, scale=2.0)
        
        assert size2 > size1
    
    def test_scale_affects_size(self):
        """Test that scale affects size."""
        size1 = compute_blob_point_size(base_size=5.0, mass=1.0, base_mass=1.0, scale=1.0)
        size2 = compute_blob_point_size(base_size=5.0, mass=1.0, base_mass=1.0, scale=2.0)
        
        assert size2 > size1


class TestComputeSpritePointSize:
    """Tests for compute_sprite_point_size function."""
    
    def test_basic_size(self):
        """Test basic size calculation."""
        size = compute_sprite_point_size(
            base_size=5.0,
            scale=2.0
        )
        assert size > 0
    
    def test_scale_affects_size(self):
        """Test that scale affects size."""
        size1 = compute_sprite_point_size(base_size=5.0, scale=1.0)
        size2 = compute_sprite_point_size(base_size=5.0, scale=2.0)
        
        assert size2 > size1
        
        # Size should be different based on distance
        assert size1 != size2
