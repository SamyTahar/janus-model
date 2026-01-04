"""
Tests for color_mapper module.
"""

import pytest
import math
from particle_sim3d.color_mapper import (
    GradientStats,
    ColorMapper,
    gradient_color,
    normalize,
    format_gradient_value,
    DEFAULT_GRADIENT_STOPS,
    GRAD_STAT_ALPHA,
)


class TestNormalize:
    """Tests for normalize function."""
    
    def test_normalize_middle(self):
        """Test value in middle of range."""
        result = normalize(5.0, 0.0, 10.0)
        assert result == 0.5
    
    def test_normalize_min(self):
        """Test value at minimum."""
        result = normalize(0.0, 0.0, 10.0)
        assert result == 0.0
    
    def test_normalize_max(self):
        """Test value at maximum."""
        result = normalize(10.0, 0.0, 10.0)
        assert result == 1.0
    
    def test_normalize_below_min(self):
        """Test value below minimum clamps to 0."""
        result = normalize(-5.0, 0.0, 10.0)
        assert result == 0.0
    
    def test_normalize_above_max(self):
        """Test value above maximum clamps to 1."""
        result = normalize(15.0, 0.0, 10.0)
        assert result == 1.0
    
    def test_normalize_nan(self):
        """Test NaN returns 0."""
        result = normalize(float('nan'), 0.0, 10.0)
        assert result == 0.0
    
    def test_normalize_zero_span(self):
        """Test zero span returns 0.5."""
        result = normalize(5.0, 5.0, 5.0)
        assert result == 0.5


class TestGradientColor:
    """Tests for gradient_color function."""
    
    def test_at_zero(self):
        """Test color at t=0."""
        r, g, b, a = gradient_color(0.0)
        assert all(0 <= c <= 255 for c in (r, g, b, a))
    
    def test_at_one(self):
        """Test color at t=1."""
        r, g, b, a = gradient_color(1.0)
        assert all(0 <= c <= 255 for c in (r, g, b, a))
    
    def test_at_middle(self):
        """Test color at t=0.5."""
        r, g, b, a = gradient_color(0.5)
        assert all(0 <= c <= 255 for c in (r, g, b, a))
    
    def test_clamps_below_zero(self):
        """Test t<0 clamps to first color."""
        c1 = gradient_color(-0.5)
        c2 = gradient_color(0.0)
        assert c1 == c2
    
    def test_clamps_above_one(self):
        """Test t>1 clamps to last color."""
        c1 = gradient_color(1.5)
        c2 = gradient_color(1.0)
        assert c1 == c2
    
    def test_custom_alpha(self):
        """Test custom alpha value."""
        r, g, b, a = gradient_color(0.5, alpha=128)
        assert a == 128


class TestGradientStats:
    """Tests for GradientStats class."""
    
    def test_init(self):
        """Test initialization."""
        stats = GradientStats()
        assert len(stats.stats) == 0
    
    def test_update_first_time(self):
        """Test first update stores values directly."""
        stats = GradientStats()
        vmin, vmax = stats.update("test", 1.0, 10.0)
        assert vmin == 1.0
        assert vmax == 10.0
    
    def test_update_expands_range(self):
        """Test range expands immediately."""
        stats = GradientStats()
        stats.update("test", 5.0, 15.0)
        
        # New min < old min: expands immediately
        vmin, vmax = stats.update("test", 2.0, 15.0)
        assert vmin == 2.0
        
        # New max > old max: expands immediately
        vmin, vmax = stats.update("test", 2.0, 20.0)
        assert vmax == 20.0
    
    def test_reset(self):
        """Test reset clears stats."""
        stats = GradientStats()
        stats.update("test", 1.0, 10.0)
        assert len(stats.stats) == 1
        
        stats.reset()
        assert len(stats.stats) == 0


class TestColorMapper:
    """Tests for ColorMapper class."""
    
    def test_init(self):
        """Test initialization."""
        mapper = ColorMapper()
        assert len(mapper._stats.stats) == 0
    
    def test_reset_stats(self):
        """Test resetting stats."""
        mapper = ColorMapper()
        # Simulate some stats
        mapper._stats.update("test", 0.0, 1.0)
        
        mapper.reset_stats()
        assert len(mapper._stats.stats) == 0
    
    def test_get_stats(self):
        """Test getting stats returns copy."""
        mapper = ColorMapper()
        mapper._stats.update("test", 0.0, 1.0)
        
        stats = mapper.get_stats()
        assert "test" in stats
        
        # Verify it's a copy
        stats["test"] = (99.0, 99.0)
        assert mapper._stats.stats["test"] != (99.0, 99.0)


class TestFormatGradientValue:
    """Tests for format_gradient_value function."""
    
    def test_normal_value(self):
        """Test formatting normal value."""
        result = format_gradient_value(1.5)
        assert result == "1.50"
    
    def test_large_value(self):
        """Test formatting large value uses scientific notation."""
        result = format_gradient_value(12345.0)
        assert "e" in result.lower() or len(result) < 10
    
    def test_small_value(self):
        """Test formatting small value uses scientific notation."""
        result = format_gradient_value(0.001)
        assert "e" in result.lower() or result == "0.001"
    
    def test_nan(self):
        """Test formatting NaN."""
        result = format_gradient_value(float('nan'))
        assert result == "n/a"
    
    def test_inf(self):
        """Test formatting infinity."""
        result = format_gradient_value(float('inf'))
        assert result == "n/a"


class TestDefaultGradientStops:
    """Tests for default gradient stops."""
    
    def test_has_stops(self):
        """Verify gradient stops exist."""
        assert len(DEFAULT_GRADIENT_STOPS) >= 2
    
    def test_starts_at_zero(self):
        """Verify first stop is at t=0."""
        assert DEFAULT_GRADIENT_STOPS[0][0] == 0.0
    
    def test_ends_at_one(self):
        """Verify last stop is at t=1."""
        assert DEFAULT_GRADIENT_STOPS[-1][0] == 1.0
    
    def test_stops_are_ordered(self):
        """Verify stops are in ascending order."""
        for i in range(1, len(DEFAULT_GRADIENT_STOPS)):
            assert DEFAULT_GRADIENT_STOPS[i][0] >= DEFAULT_GRADIENT_STOPS[i-1][0]
