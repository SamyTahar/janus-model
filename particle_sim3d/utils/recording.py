"""
Video recording utilities for the 3D particle simulation.

This module handles frame capture and video encoding using ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class RecordingState:
    """State for video recording.
    
    Attributes:
        recording: Whether currently recording
        record_dir: Directory for frame files
        record_frame: Current frame number
        record_fps: Target FPS for encoding
        ffmpeg_path: Path to ffmpeg executable
    """
    recording: bool = False
    record_dir: Path | None = None
    record_frame: int = 0
    record_fps: int = 30
    ffmpeg_path: str | None = field(default_factory=lambda: shutil.which("ffmpeg"))


def start_recording(
    state: RecordingState,
    output_base: str = "particle_sim3d/output",
) -> bool:
    """
    Start recording frames to disk.
    
    Args:
        state: Recording state to modify
        output_base: Base directory for output files
    
    Returns:
        True if recording started successfully
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_base)
    state.record_dir = out_root / f"video_{stamp}"
    
    try:
        state.record_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[video] failed to create output dir: {exc}")
        state.record_dir = None
        return False
    
    state.record_frame = 0
    state.recording = True
    print(f"[video] recording frames to {state.record_dir}")
    return True


def stop_recording(state: RecordingState) -> Path | None:
    """
    Stop recording and encode video with ffmpeg.
    
    Args:
        state: Recording state to modify
    
    Returns:
        Path to the encoded video file, or None if encoding failed
    """
    state.recording = False
    
    if state.record_dir is None:
        return None
    
    if state.record_frame == 0:
        print("[video] no frames recorded.")
        return None
    
    out_path = state.record_dir.with_suffix(".mp4")
    
    if state.ffmpeg_path is None:
        print("[video] ffmpeg not found; frames kept for manual encoding.")
        print(f"[video] frames dir: {state.record_dir}")
        return None
    
    cmd = [
        state.ffmpeg_path,
        "-y",
        "-framerate",
        str(state.record_fps),
        "-start_number",
        "0",
        "-i",
        str(state.record_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[video] saved to {out_path}")
        return out_path
    except Exception as exc:
        print(f"[video] ffmpeg failed: {exc}")
        return None


def save_frame(
    state: RecordingState,
    get_color_buffer: Callable[[], Any],
    width: int,
    height: int,
) -> bool:
    """
    Save a single frame during recording.
    
    Args:
        state: Recording state
        get_color_buffer: Callback to get pixel buffer
        width: Frame width
        height: Frame height
    
    Returns:
        True if frame was saved successfully
    """
    if not state.recording or state.record_dir is None:
        return False
    
    try:
        from pyglet import image  # type: ignore
        
        color_buf = get_color_buffer()
        frame_path = state.record_dir / f"frame_{state.record_frame:05d}.png"
        color_buf.save(str(frame_path))
        state.record_frame += 1
        return True
    except Exception as exc:
        print(f"[video] frame save failed: {exc}")
        return False


def toggle_recording(state: RecordingState) -> bool:
    """
    Toggle recording state.
    
    Args:
        state: Recording state to toggle
    
    Returns:
        True if now recording, False if stopped
    """
    if state.recording:
        stop_recording(state)
        return False
    else:
        return start_recording(state)
