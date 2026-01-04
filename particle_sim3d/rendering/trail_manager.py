"""
Trail management for 3D particle visualization.

This module handles the storage and rendering of particle trails,
maintaining position history for smooth trail visualization.
"""

from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particle_sim3d.core.sim import Particle3D


# =============================================================================
# Default Colors
# =============================================================================

POS_BASE_COLOR = (188, 214, 255, 220)
NEG_BASE_COLOR = (255, 184, 107, 220)


# =============================================================================
# Trail State
# =============================================================================

@dataclass
class TrailState:
    """
    State container for particle trails.
    
    Attributes:
        history_pos: Position history for M+ particles
        history_neg: Position history for M- particles
        max_length: Maximum trail length (frames)
    """
    history_pos: deque[list[float]] = field(default_factory=lambda: deque(maxlen=16))
    history_neg: deque[list[float]] = field(default_factory=lambda: deque(maxlen=16))
    max_length: int = 16
    
    # Reusable buffers
    _xyz: array = field(default_factory=lambda: array("f"))
    _rgba: array = field(default_factory=lambda: array("B"))
    _xyz_pos: array = field(default_factory=lambda: array("f"))
    _rgba_pos: array = field(default_factory=lambda: array("B"))
    _xyz_neg: array = field(default_factory=lambda: array("f"))
    _rgba_neg: array = field(default_factory=lambda: array("B"))


# =============================================================================
# Trail Manager
# =============================================================================

class TrailManager:
    """
    Manages particle trail history and rendering data.
    
    This class stores a sliding window of particle positions to create
    smooth trail effects behind moving particles.
    """
    
    def __init__(self, max_length: int = 16):
        """
        Initialize the trail manager.
        
        Args:
            max_length: Maximum number of frames to store
        """
        self._state = TrailState(max_length=max(2, max_length))
        self._reset()
    
    def _reset(self) -> None:
        """Reset trail history with current max_length."""
        maxlen = self._state.max_length
        self._state.history_pos = deque(maxlen=maxlen)
        self._state.history_neg = deque(maxlen=maxlen)
    
    def set_max_length(self, length: int) -> None:
        """
        Set the maximum trail length.
        
        Args:
            length: Number of frames to store
        """
        new_len = max(2, length)
        if new_len != self._state.max_length:
            self._state.max_length = new_len
            self._reset()
    
    def update(
        self,
        particles: list["Particle3D"],
        *,
        stride: int = 4,
        pos_only: bool = False,
    ) -> None:
        """
        Update trail history with current particle positions.
        
        Args:
            particles: List of particles
            stride: Only store every Nth particle
            pos_only: If True, only store M+ particles
        """
        stride = max(1, stride)
        pos_frame: list[float] = []
        neg_frame: list[float] = []
        
        for i, pt in enumerate(particles):
            if i % stride != 0:
                continue
            if pt.s > 0:
                pos_frame.extend((pt.x, pt.y, pt.z))
            elif not pos_only:
                neg_frame.extend((pt.x, pt.y, pt.z))
        
        # Clear history if particle count changed
        if self._state.history_pos and len(pos_frame) != len(self._state.history_pos[0]):
            self._state.history_pos.clear()
        if self._state.history_neg and len(neg_frame) != len(self._state.history_neg[0]):
            self._state.history_neg.clear()
        
        self._state.history_pos.append(pos_frame)
        self._state.history_neg.append(neg_frame)
    
    def get_trail_data(
        self,
        *,
        alpha: float = 0.35,
        pos_color: tuple[int, int, int, int] = POS_BASE_COLOR,
        neg_color: tuple[int, int, int, int] = NEG_BASE_COLOR,
    ) -> tuple[array, array]:
        """
        Get combined trail vertex data for rendering.
        
        Args:
            alpha: Trail opacity multiplier
            pos_color: Base color for M+ trails
            neg_color: Base color for M- trails
        
        Returns:
            (xyz_array, rgba_array) for rendering
        """
        xyz = self._state._xyz
        rgba = self._state._rgba
        xyz.clear()
        rgba.clear()
        
        self._append_trails(self._state.history_pos, pos_color, xyz, rgba, alpha)
        self._append_trails(self._state.history_neg, neg_color, xyz, rgba, alpha)
        
        return xyz, rgba
    
    def get_trail_data_split(
        self,
        *,
        alpha: float = 0.35,
        pos_color: tuple[int, int, int, int] = POS_BASE_COLOR,
        neg_color: tuple[int, int, int, int] = NEG_BASE_COLOR,
    ) -> tuple[array, array, array, array]:
        """
        Get separate trail vertex data for M+ and M-.
        
        Returns:
            (xyz_pos, rgba_pos, xyz_neg, rgba_neg)
        """
        xyz_pos = self._state._xyz_pos
        rgba_pos = self._state._rgba_pos
        xyz_neg = self._state._xyz_neg
        rgba_neg = self._state._rgba_neg
        
        xyz_pos.clear()
        rgba_pos.clear()
        xyz_neg.clear()
        rgba_neg.clear()
        
        self._append_trails(self._state.history_pos, pos_color, xyz_pos, rgba_pos, alpha)
        self._append_trails(self._state.history_neg, neg_color, xyz_neg, rgba_neg, alpha)
        
        return xyz_pos, rgba_pos, xyz_neg, rgba_neg
    
    def _append_trails(
        self,
        history: deque[list[float]],
        base_color: tuple[int, int, int, int],
        xyz: array,
        rgba: array,
        alpha_mult: float,
    ) -> None:
        """
        Append trail line segments to buffers.
        
        Args:
            history: Position history deque
            base_color: RGBA base color
            xyz: Output position buffer
            rgba: Output color buffer
            alpha_mult: Alpha multiplier (0-1)
        """
        if len(history) < 2:
            return
        
        base_alpha = int(round(float(base_color[3]) * float(alpha_mult)))
        base_alpha = max(0, min(255, base_alpha))
        total = len(history) - 1
        
        for f in range(1, len(history)):
            fade = f / total if total > 0 else 1.0
            alpha = int(round(base_alpha * fade))
            if alpha <= 0:
                continue
            
            prev = history[f - 1]
            cur = history[f]
            limit = min(len(prev), len(cur))
            
            for i in range(0, limit, 3):
                xyz.extend(prev[i : i + 3])
                xyz.extend(cur[i : i + 3])
                rgba.extend((base_color[0], base_color[1], base_color[2], alpha))
                rgba.extend((base_color[0], base_color[1], base_color[2], alpha))
    
    def clear(self) -> None:
        """Clear all trail history."""
        self._state.history_pos.clear()
        self._state.history_neg.clear()
