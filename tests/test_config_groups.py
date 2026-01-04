"""
Tests for config_groups module.
"""

import pytest
from particle_sim3d.config_groups import (
    RESET_KEYS,
    BOUNDS_KEYS,
    TRAIL_KEYS,
    MENU_GROUPS,
    PARAM_HINTS,
    get_menu_group_title,
    get_param_hint,
    is_reset_required,
    is_bounds_related,
    is_trail_related,
)


class TestConfigGroups:
    """Tests for configuration group constants and functions."""
    
    def test_reset_keys_contains_init_mode(self):
        """Verify init_mode requires reset."""
        assert "init_mode" in RESET_KEYS
        
    def test_reset_keys_contains_particle_count(self):
        """Verify particle_count requires reset."""
        assert "particle_count" in RESET_KEYS
    
    def test_bounds_keys_has_bounds_enabled(self):
        """Verify bounds_enabled is in bounds keys."""
        assert "bounds_enabled" in BOUNDS_KEYS
    
    def test_trail_keys_has_trails_enabled(self):
        """Verify trails_enabled is in trail keys."""
        assert "trails_enabled" in TRAIL_KEYS
    
    def test_menu_groups_structure(self):
        """Verify MENU_GROUPS has proper structure."""
        assert len(MENU_GROUPS) > 0
        # Check a known entry
        assert "time_scale" in MENU_GROUPS
        assert MENU_GROUPS["time_scale"] == ("Init", "Time")
    
    def test_param_hints_non_empty(self):
        """Verify PARAM_HINTS contains entries."""
        assert len(PARAM_HINTS) > 0
        assert "time_scale" in PARAM_HINTS


class TestGetMenuGroupTitle:
    """Tests for get_menu_group_title function."""
    
    def test_known_key(self):
        """Test with a known key."""
        group, subgroup = get_menu_group_title("time_scale")
        assert group == "Init"
        assert subgroup == "Time"
    
    def test_unknown_key(self):
        """Test with an unknown key."""
        group, subgroup = get_menu_group_title("not_a_real_key")
        assert group is None
        assert subgroup is None


class TestGetParamHint:
    """Tests for get_param_hint function."""
    
    def test_known_key(self):
        """Test with a known key."""
        hint = get_param_hint("time_scale")
        assert hint != ""
        assert "time" in hint.lower() or "multiplier" in hint.lower()
    
    def test_unknown_key(self):
        """Test with unknown key returns empty."""
        hint = get_param_hint("not_a_real_key")
        assert hint == ""


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_is_reset_required_true(self):
        """Test is_reset_required for keys that need reset."""
        assert is_reset_required("init_mode") is True
        assert is_reset_required("particle_count") is True
    
    def test_is_reset_required_false(self):
        """Test is_reset_required for keys that don't need reset."""
        assert is_reset_required("time_scale") is False
    
    def test_is_bounds_related(self):
        """Test is_bounds_related."""
        assert is_bounds_related("bound_mode") is True
        assert is_bounds_related("time_scale") is False
    
    def test_is_trail_related(self):
        """Test is_trail_related."""
        assert is_trail_related("trails_enabled") is True
        assert is_trail_related("time_scale") is False
