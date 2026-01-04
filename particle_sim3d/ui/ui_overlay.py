"""
UI overlay utilities for 3D particle simulation.

This module provides functions for creating and managing 2D UI elements
overlaid on the 3D rendering, including help panels, legends, and recording indicators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Default Gradient Stops
# =============================================================================

DEFAULT_GRADIENT_STOPS = [
    (0.0, (40, 100, 190)),
    (0.45, (245, 245, 245)),
    (0.75, (255, 170, 90)),
    (1.0, (230, 60, 50)),
]


# =============================================================================
# Legend State
# =============================================================================

@dataclass
class LegendState:
    """State for the legend panel.
    
    Attributes:
        panel: Background panel shape
        pos_label: Positive mass label
        neg_label: Negative mass label
        scale_label: Scale/range label
        pos_swatch: Positive color swatch
        neg_swatch: Negative color swatch
        grad_rects: Gradient bar rectangles
    """
    panel: Any = None
    pos_label: Any = None
    neg_label: Any = None
    scale_label: Any = None
    pos_swatch: Any = None
    neg_swatch: Any = None
    grad_rects: list[Any] = field(default_factory=list)


@dataclass
class OverlayState:
    """State for all UI overlays.
    
    Attributes:
        help_label: Help text label
        help_panel: Help panel background
        menu_label: Menu text label
        menu_panel: Menu panel background
        record_label: Recording indicator label
        record_panel: Recording indicator background
        fps_display: FPS counter display
        legend: Legend panel state
        active_view_edges: View separator edges
        active_view_label: Active view indicator label
    """
    help_label: Any = None
    help_panel: Any = None
    menu_label: Any = None
    menu_panel: Any = None
    record_label: Any = None
    record_panel: Any = None
    fps_display: Any = None
    legend: LegendState = field(default_factory=LegendState)
    active_view_edges: list[Any] = field(default_factory=list)
    active_view_label: Any = None


# =============================================================================
# Legend Color Helpers
# =============================================================================

def legend_color_at(
    t: float,
    stops: list[tuple[float, tuple[int, int, int]]],
) -> tuple[int, int, int]:
    """
    Get a color from gradient stops.
    
    Args:
        t: Position on gradient (0.0 to 1.0)
        stops: Gradient stops list
    
    Returns:
        (r, g, b) color tuple
    """
    t = max(0.0, min(1.0, float(t)))
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            local = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            r = int(round(c0[0] + (c1[0] - c0[0]) * local))
            g = int(round(c0[1] + (c1[1] - c0[1]) * local))
            b = int(round(c0[2] + (c1[2] - c0[2]) * local))
            return r, g, b
    return stops[-1][1]


def create_legend_gradient_colors(
    width: int,
    stops: list[tuple[float, tuple[int, int, int]]] | None = None,
) -> list[tuple[int, int, int]]:
    """
    Create a list of colors for a gradient legend bar.
    
    Args:
        width: Number of color samples to generate
        stops: Gradient stops, or None for defaults
    
    Returns:
        List of (r, g, b) colors
    """
    if stops is None:
        stops = DEFAULT_GRADIENT_STOPS
    
    colors = []
    for i in range(width):
        t = (i + 0.5) / width
        colors.append(legend_color_at(t, stops))
    return colors


# =============================================================================
# UI Creation
# =============================================================================

def create_overlay_state(window: Any, width: int, height: int) -> OverlayState:
    """
    Create initial overlay state with all UI elements.
    
    Args:
        window: Pyglet window
        width: Window width
        height: Window height
    
    Returns:
        Initialized OverlayState
    """
    try:
        import pyglet  # type: ignore
    except ImportError:
        raise RuntimeError("Missing dependency: install pyglet (pip install pyglet).")
    
    state = OverlayState()
    
    # Help label and panel
    state.help_label = pyglet.text.Label(
        "F1/TAB menu | ENTER edit | PGUP/PGDN step | V HUD | X filter | B bound wire | "
        "G focus | H fit | T follow | O record | P screenshot | SPACE pause | R reset | "
        "S save | L load | +/- speed | Click view to select | Mouse drag rotate | "
        "Shift+drag/Right drag pan | Wheel zoom",
        x=10,
        y=10,
        anchor_x="left",
        anchor_y="bottom",
        font_name="Segoe UI",
        font_size=13,
        color=(235, 240, 255, 245),
    )
    
    state.help_panel = pyglet.shapes.BorderedRectangle(
        8, 8, 10, 10,
        border=1,
        color=(0, 0, 0, 170),
        border_color=(170, 190, 210),
    )
    
    # Record label and panel
    state.record_label = pyglet.text.Label(
        "",
        anchor_x="right",
        anchor_y="top",
        font_name="Segoe UI",
        font_size=12,
        color=(255, 90, 90, 255),
    )
    
    state.record_panel = pyglet.shapes.BorderedRectangle(
        8, 8, 10, 10,
        border=1,
        color=(0, 0, 0, 180),
        border_color=(200, 80, 80),
    )
    
    # Menu label and panel
    state.menu_label = pyglet.text.Label(
        "",
        x=10,
        y=height - 10,
        anchor_x="left",
        anchor_y="top",
        font_name="Consolas",
        font_size=16,
        color=(245, 250, 255, 255),
        multiline=True,
        width=640,
    )
    
    state.menu_panel = pyglet.shapes.BorderedRectangle(
        8, 8, 10, 10,
        border=1,
        color=(0, 0, 0, 210),
        border_color=(255, 255, 255),
    )
    
    # FPS display
    state.fps_display = pyglet.window.FPSDisplay(window)
    state.fps_display.label.anchor_x = "right"
    state.fps_display.label.anchor_y = "top"
    state.fps_display.label.x = width - 10
    state.fps_display.label.y = height - 10
    
    return state


# =============================================================================
# Legend Drawing
# =============================================================================

def draw_legend(
    state: LegendState,
    info: dict[str, Any],
    window_width: int,
    window_height: int,
) -> None:
    """
    Draw the legend panel with M+/M- color swatches or gradient.
    
    Args:
        state: Legend state
        info: Legend info dict from get_legend_info callback
        window_width: Window width
        window_height: Window height
    """
    try:
        import pyglet  # type: ignore
    except ImportError:
        return
    
    if not info or not info.get("enabled", False):
        return
    
    # Initialize legend elements if needed
    if state.panel is None:
        state.panel = pyglet.shapes.BorderedRectangle(
            8, 8, 10, 10,
            border=1,
            color=(0, 0, 0, 200),
            border_color=(150, 170, 190),
        )
        state.pos_label = pyglet.text.Label(
            "M+",
            anchor_x="left",
            anchor_y="center",
            font_name="Segoe UI",
            font_size=12,
            color=(230, 235, 245, 235),
        )
        state.neg_label = pyglet.text.Label(
            "M-",
            anchor_x="left",
            anchor_y="center",
            font_name="Segoe UI",
            font_size=12,
            color=(230, 235, 245, 235),
        )
        state.scale_label = pyglet.text.Label(
            "",
            anchor_x="left",
            anchor_y="center",
            font_name="Segoe UI",
            font_size=11,
            color=(190, 205, 220, 220),
        )
        state.pos_swatch = pyglet.shapes.Rectangle(0, 0, 12, 12, color=(180, 200, 255))
        state.neg_swatch = pyglet.shapes.Rectangle(0, 0, 12, 12, color=(255, 180, 110))
    
    # Get legend settings
    pos_color = info.get("pos_color", (188, 214, 255, 220))
    neg_color = info.get("neg_color", (255, 184, 107, 220))
    grad_enabled = bool(info.get("gradient_enabled", False))
    stops = info.get("gradient_stops", DEFAULT_GRADIENT_STOPS)
    
    # Layout constants
    pad = 8
    row_h = 18
    swatch_w = 12
    swatch_h = 12
    grad_w = 90
    grad_h = 8
    label_gap = 6
    
    # Labels
    grad_label = info.get("gradient_label", "M+ gradient")
    state.pos_label.text = grad_label if grad_enabled else "M+"
    state.neg_label.text = "M-"
    
    range_text = info.get("gradient_range_text")
    show_scale = bool(grad_enabled and range_text)
    if state.scale_label is not None:
        state.scale_label.text = range_text or ""
    
    # Panel size
    label_w = max(state.pos_label.content_width, state.neg_label.content_width)
    if show_scale and state.scale_label is not None:
        label_w = max(label_w, state.scale_label.content_width)
    
    panel_w = max(140, int((2 * pad) + label_gap + label_w + (grad_w if grad_enabled else swatch_w)))
    avail_w = max(1, window_width - (2 * pad))
    panel_w = min(panel_w, avail_w)
    panel_h = (2 * pad) + (row_h * (3 if show_scale else 2))
    
    # Position
    base_x = max(pad, window_width - panel_w - pad)
    base_y = pad
    
    state.panel.x = base_x
    state.panel.y = base_y
    state.panel.width = panel_w
    state.panel.height = panel_h
    state.panel.draw()
    
    # Row positions
    row1_y = base_y + panel_h - pad - row_h
    row2_y = row1_y - row_h
    row3_y = row2_y - row_h if show_scale else row2_y
    label_x = base_x + pad + (grad_w if grad_enabled else swatch_w) + label_gap
    
    # Position labels
    state.pos_label.x = label_x
    state.pos_label.y = row1_y + (row_h / 2)
    state.neg_label.x = label_x
    state.neg_label.y = row3_y + (row_h / 2)
    
    if show_scale and state.scale_label is not None:
        state.scale_label.x = base_x + pad
        state.scale_label.y = row2_y + (row_h / 2)
    
    # Draw gradient or swatch
    if grad_enabled:
        seg_count = 24
        if not state.grad_rects or len(state.grad_rects) != seg_count:
            for rect in state.grad_rects:
                rect.delete()
            state.grad_rects = []
            for _ in range(seg_count):
                state.grad_rects.append(
                    pyglet.shapes.Rectangle(0, 0, 1, grad_h, color=(255, 255, 255))
                )
        
        seg_w = grad_w / seg_count
        grad_x = base_x + pad
        grad_y = row1_y + ((row_h - grad_h) / 2)
        
        for i, rect in enumerate(state.grad_rects):
            rect.x = grad_x + (i * seg_w)
            rect.y = grad_y
            rect.width = seg_w + 0.5
            rect.height = grad_h
            r, g, b = legend_color_at((i + 0.5) / seg_count, stops)
            rect.color = (r, g, b)
            rect.opacity = 230
            rect.draw()
        
        if state.pos_swatch is not None:
            state.pos_swatch.opacity = 0
    else:
        if state.pos_swatch is not None:
            state.pos_swatch.x = base_x + pad
            state.pos_swatch.y = row1_y + ((row_h - swatch_h) / 2)
            state.pos_swatch.width = swatch_w
            state.pos_swatch.height = swatch_h
            state.pos_swatch.color = tuple(pos_color[:3])
            state.pos_swatch.opacity = 230
            state.pos_swatch.draw()
    
    # Negative swatch (always shown)
    if state.neg_swatch is not None:
        state.neg_swatch.x = base_x + pad
        state.neg_swatch.y = row3_y + ((row_h - swatch_h) / 2)
        state.neg_swatch.width = swatch_w
        state.neg_swatch.height = swatch_h
        state.neg_swatch.color = tuple(neg_color[:3])
        state.neg_swatch.opacity = 230
        state.neg_swatch.draw()
    
    # Draw labels
    state.pos_label.draw()
    if show_scale and state.scale_label is not None:
        state.scale_label.draw()
    state.neg_label.draw()


# =============================================================================
# Menu Drawing
# =============================================================================

def draw_menu(
    overlay: OverlayState,
    text: str,
    window_width: int,
    window_height: int,
) -> None:
    """
    Draw the menu overlay panel.
    
    Args:
        overlay: Overlay state
        text: Menu text to display
        window_width: Window width
        window_height: Window height
    """
    if not text or overlay.menu_label is None or overlay.menu_panel is None:
        return
    
    overlay.menu_label.text = text
    
    margin = 10
    pad = 12
    panel_w = min(window_width - (2 * margin), 700)
    overlay.menu_label.width = max(120, panel_w - (2 * pad))
    overlay.menu_label.x = margin + pad
    overlay.menu_label.y = window_height - (margin + pad)
    
    panel_h = int(overlay.menu_label.content_height) + (2 * pad)
    panel_h = min(panel_h, window_height - (2 * margin))
    overlay.menu_panel.x = margin
    overlay.menu_panel.y = window_height - margin - panel_h
    overlay.menu_panel.width = panel_w
    overlay.menu_panel.height = panel_h
    
    overlay.menu_panel.draw()
    overlay.menu_label.draw()


# =============================================================================
# Recording Indicator
# =============================================================================

def draw_recording_indicator(
    overlay: OverlayState,
    is_recording: bool,
    frame_number: int,
    window_width: int,
    window_height: int,
) -> None:
    """
    Draw the recording indicator.
    
    Args:
        overlay: Overlay state
        is_recording: Whether currently recording
        frame_number: Current frame number
        window_width: Window width
        window_height: Window height
    """
    if not is_recording or overlay.record_label is None or overlay.record_panel is None:
        return
    
    overlay.record_label.text = f"REC {frame_number:05d}"
    
    pad = 6
    panel_w = int(overlay.record_label.content_width) + (2 * pad)
    panel_h = int(overlay.record_label.content_height) + (2 * pad)
    
    overlay.record_panel.x = window_width - panel_w - 10
    overlay.record_panel.y = window_height - panel_h - 10
    overlay.record_panel.width = panel_w
    overlay.record_panel.height = panel_h
    
    overlay.record_label.x = overlay.record_panel.x + panel_w - pad
    overlay.record_label.y = overlay.record_panel.y + panel_h - pad
    
    overlay.record_panel.draw()
    overlay.record_label.draw()


# =============================================================================
# Help Panel
# =============================================================================

def draw_help_panel(overlay: OverlayState, visible: bool = True) -> None:
    """
    Draw the help text panel at the bottom of the screen.
    
    Args:
        overlay: Overlay state
        visible: Whether to show the help panel
    """
    if not visible or overlay.help_label is None:
        return
    
    if overlay.help_panel is not None:
        # Update panel size based on label
        pad = 6
        panel_w = int(overlay.help_label.content_width) + (2 * pad)
        panel_h = int(overlay.help_label.content_height) + (2 * pad)
        overlay.help_panel.width = panel_w
        overlay.help_panel.height = panel_h
        overlay.help_panel.draw()
    
    overlay.help_label.draw()


# =============================================================================
# View Edges
# =============================================================================

def draw_view_edges(
    overlay: OverlayState,
    active_view: int,
    view_count: int,
    window_width: int,
    window_height: int,
) -> None:
    """
    Draw separator edges between multi-view panels.
    
    Args:
        overlay: Overlay state
        active_view: Currently active view index
        view_count: Total number of views
        window_width: Window width
        window_height: Window height
    """
    try:
        import pyglet  # type: ignore
    except ImportError:
        return
    
    if view_count <= 1:
        # Clear edges if not multi-view
        for edge in overlay.active_view_edges:
            edge.delete()
        overlay.active_view_edges = []
        if overlay.active_view_label is not None:
            overlay.active_view_label.delete()
            overlay.active_view_label = None
        return
    
    # Ensure we have enough edge lines
    needed = (view_count - 1) + 2  # separators + active view frame
    while len(overlay.active_view_edges) < needed:
        edge = pyglet.shapes.Line(0, 0, 0, window_height, 2, color=(200, 220, 255))
        overlay.active_view_edges.append(edge)
    
    # Draw view separators
    slice_w = window_width / view_count
    for i in range(view_count - 1):
        x = int((i + 1) * slice_w)
        edge = overlay.active_view_edges[i]
        edge.x = x
        edge.x2 = x
        edge.y = 0
        edge.y2 = window_height
        edge.color = (100, 120, 150)
        edge.opacity = 180
        edge.draw()
    
    # Active view label
    if overlay.active_view_label is None:
        overlay.active_view_label = pyglet.text.Label(
            "",
            anchor_x="center",
            anchor_y="top",
            font_name="Segoe UI",
            font_size=11,
            color=(200, 220, 255, 200),
        )
    
    view_x = int((active_view + 0.5) * slice_w)
    overlay.active_view_label.text = f"View {active_view + 1}"
    overlay.active_view_label.x = view_x
    overlay.active_view_label.y = window_height - 35
    overlay.active_view_label.draw()
