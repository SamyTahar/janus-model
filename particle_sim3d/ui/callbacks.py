"""
Keyboard and event callbacks for the Janus 3D simulation.

This module provides handler classes for processing user input events
including keyboard shortcuts and text input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    from particle_sim3d.params import Sim3DParams


# =============================================================================
# Key Handler
# =============================================================================

class KeyHandler:
    """
    Handles keyboard input for the simulation application.
    
    This class processes keyboard events and dispatches them to the
    appropriate action handlers.
    """
    
    # Time scale presets
    TIME_PRESETS = {
        "0": 0.0,
        "1": 1.0,
        "2": 2.0,
        "3": 5.0,
        "4": 10.0,
        "5": 20.0,
    }
    
    # Display mode cycle
    DISPLAY_MODES = ["all", "pos", "neg"]
    
    def __init__(
        self,
        *,
        get_params: Callable[[], "Sim3DParams"],
        set_params: Callable[["Sim3DParams"], None],
        get_running: Callable[[], bool],
        set_running: Callable[[bool], None],
        get_menu_open: Callable[[], bool],
        set_menu_open: Callable[[bool], None],
        get_edit_active: Callable[[], bool],
        set_edit_active: Callable[[bool], None],
        get_display_mode: Callable[[], str],
        set_display_mode: Callable[[str], None],
        get_show_hud: Callable[[], bool],
        set_show_hud: Callable[[bool], None],
        on_reset: Callable[[], None],
        on_save: Callable[[], None],
        on_load: Callable[[], None],
        on_mark_hud_dirty: Callable[[], None],
        on_menu_apply: Callable[[str], None],
        on_menu_navigate: Callable[[int], None],
        on_nudge_step_scale: Callable[[int], None],
        on_edit_commit: Callable[[], None],
        on_edit_cancel: Callable[[], None],
        on_edit_backspace: Callable[[], None],
        on_edit_clear: Callable[[], None],
    ):
        """
        Initialize the key handler with callbacks.
        
        All callbacks are functions that access/modify the application state.
        """
        self._get_params = get_params
        self._set_params = set_params
        self._get_running = get_running
        self._set_running = set_running
        self._get_menu_open = get_menu_open
        self._set_menu_open = set_menu_open
        self._get_edit_active = get_edit_active
        self._set_edit_active = set_edit_active
        self._get_display_mode = get_display_mode
        self._set_display_mode = set_display_mode
        self._get_show_hud = get_show_hud
        self._set_show_hud = set_show_hud
        self._on_reset = on_reset
        self._on_save = on_save
        self._on_load = on_load
        self._on_mark_hud_dirty = on_mark_hud_dirty
        self._on_menu_apply = on_menu_apply
        self._on_menu_navigate = on_menu_navigate
        self._on_nudge_step_scale = on_nudge_step_scale
        self._on_edit_commit = on_edit_commit
        self._on_edit_cancel = on_edit_cancel
        self._on_edit_backspace = on_edit_backspace
        self._on_edit_clear = on_edit_clear
    
    def handle_key(self, key: str) -> bool:
        """
        Process a key press event.
        
        Args:
            key: Key name (e.g., 'space', 'f1', 'plus', 'r')
        
        Returns:
            True if the key was handled, False otherwise
        """
        # Toggle menu
        if key in ("f1", "tab"):
            return self._toggle_menu()
        
        menu_open = self._get_menu_open()
        
        # Non-menu shortcuts
        if not menu_open:
            if self._handle_non_menu_key(key):
                return True
        
        # Menu shortcuts
        if menu_open:
            if self._handle_menu_key(key):
                return True
        
        # Global shortcuts (work in both modes)
        return self._handle_global_key(key)
    
    def _toggle_menu(self) -> bool:
        """Toggle menu open/closed."""
        menu_open = not self._get_menu_open()
        self._set_menu_open(menu_open)
        if not menu_open:
            self._set_edit_active(False)
        return True
    
    def _handle_non_menu_key(self, key: str) -> bool:
        """Handle keys when menu is closed."""
        params = self._get_params()
        
        # Time scale adjustment
        if key in ("plus", "minus"):
            self._nudge_time_scale(+1 if key == "plus" else -1)
            return True
        
        # Time scale presets
        if key in self.TIME_PRESETS:
            self._set_time_scale_preset(key)
            return True
        
        # Fast forward toggle
        if key == "f":
            current = float(params.time_scale)
            params.time_scale = 10.0 if current < 9.5 else 1.0
            params.clamp()
            self._set_params(params)
            return True
        
        # HUD toggle
        if key == "v":
            show = not self._get_show_hud()
            self._set_show_hud(show)
            if show:
                self._on_mark_hud_dirty()
            return True
        
        # Display mode cycle
        if key == "x":
            current = self._get_display_mode()
            idx = self.DISPLAY_MODES.index(current) if current in self.DISPLAY_MODES else 0
            next_mode = self.DISPLAY_MODES[(idx + 1) % len(self.DISPLAY_MODES)]
            self._set_display_mode(next_mode)
            return True
        
        # Bound wire toggle
        if key == "b":
            params.bound_wire_visible = not bool(params.bound_wire_visible)
            params.clamp()
            self._set_params(params)
            return True
        
        return False
    
    def _handle_menu_key(self, key: str) -> bool:
        """Handle keys when menu is open."""
        edit_active = self._get_edit_active()
        
        if edit_active:
            # Edit mode keys
            if key == "enter":
                self._on_edit_commit()
                return True
            if key == "esc":
                self._on_edit_cancel()
                return True
            if key == "backspace":
                self._on_edit_backspace()
                return True
            if key == "delete":
                self._on_edit_clear()
                return True
            return True  # Consume all keys in edit mode
        
        # Menu navigation
        if key == "esc":
            self._set_menu_open(False)
            return True
        
        if key in ("pageup", "pagedown"):
            self._on_nudge_step_scale(+1 if key == "pageup" else -1)
            return True
        
        if key in ("up", "down"):
            self._on_menu_navigate(-1 if key == "up" else 1)
            return True
        
        if key in ("left", "right", "minus", "plus", "enter"):
            self._on_menu_apply(key)
            return True
        
        return False
    
    def _handle_global_key(self, key: str) -> bool:
        """Handle keys that work in both menu and non-menu modes."""
        # Pause/resume
        if key == "space":
            self._set_running(not self._get_running())
            return True
        
        # Reset simulation
        if key == "r":
            self._on_reset()
            return True
        
        # Save params
        if key == "s":
            self._on_save()
            return True
        
        # Load params
        if key == "l":
            self._on_load()
            return True
        
        # Exit
        if key == "esc":
            raise SystemExit(0)
        
        return False
    
    def _nudge_time_scale(self, direction: int) -> None:
        """Adjust time scale by a step."""
        params = self._get_params()
        current = float(params.time_scale)
        step = 0.5 if current < 5.0 else 1.0
        params.time_scale = max(0.1, current + direction * step)
        params.clamp()
        self._set_params(params)
    
    def _set_time_scale_preset(self, key: str) -> None:
        """Set time scale to a preset value."""
        if key in self.TIME_PRESETS:
            params = self._get_params()
            params.time_scale = self.TIME_PRESETS[key]
            params.clamp()
            self._set_params(params)


# =============================================================================
# Text Input Handler
# =============================================================================

class TextInputHandler:
    """
    Handles text input for parameter editing.
    """
    
    ALLOWED_CHARS = set("0123456789.-+eE")
    
    def __init__(
        self,
        *,
        get_edit_active: Callable[[], bool],
        get_edit_buffer: Callable[[], str],
        set_edit_buffer: Callable[[str], None],
        get_edit_kind: Callable[[], str],
    ):
        """
        Initialize text input handler.
        
        Args:
            get_edit_active: Callback to check if editing
            get_edit_buffer: Callback to get current buffer
            set_edit_buffer: Callback to set buffer
            get_edit_kind: Callback to get edit type
        """
        self._get_edit_active = get_edit_active
        self._get_edit_buffer = get_edit_buffer
        self._set_edit_buffer = set_edit_buffer
        self._get_edit_kind = get_edit_kind
    
    def handle_text(self, text: str) -> bool:
        """
        Process text input.
        
        Args:
            text: Input text string
        
        Returns:
            True if text was handled
        """
        if not self._get_edit_active():
            return False
        
        kind = self._get_edit_kind()
        buffer = self._get_edit_buffer()
        
        for char in text:
            if self._is_valid_char(char, kind, buffer):
                buffer += char
        
        self._set_edit_buffer(buffer)
        return True
    
    def _is_valid_char(self, char: str, kind: str, buffer: str) -> bool:
        """Check if a character is valid for the current input mode."""
        if kind == "str":
            return True
        
        if kind in ("int", "float"):
            if char not in self.ALLOWED_CHARS:
                return False
            
            # Handle minus sign (only at start)
            if char == "-" and buffer:
                return False
            
            # Handle decimal point (only one, only for float)
            if char == "." and (kind == "int" or "." in buffer):
                return False
            
            return True
        
        return False
