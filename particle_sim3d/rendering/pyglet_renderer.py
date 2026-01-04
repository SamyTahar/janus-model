from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Import refactored modules
from . import camera as camera_mod
from . import camera_controller as cam_ctrl
from . import drawing as drawing_mod
from ..utils import recording as rec_mod
from ..ui import ui_overlay as ui_mod


@dataclass(slots=True)
class RenderFrame:
    dt: float
    fps: float


def run_pyglet(
    *,
    width: int,
    height: int,
    background_rgb: tuple[int, int, int],
    get_positions_and_colors: Callable[[], tuple[list[float], list[int]]],
    get_positions_and_colors_split: Callable[[], tuple[list[float], list[int], list[float], list[int]]] | None = None,
    step_simulation: Callable[[float], None],
    on_key: Callable[[str], None],
    on_text_input: Callable[[str], None] | None = None,
    get_overlay_text: Callable[[], str] | None = None,
    get_point_size: Callable[[], float] | None = None,
    get_caption: Callable[[], str] | None = None,
    get_bound_visual: Callable[[], tuple[str, bool, float, float, float]] | None = None,
    get_legend_info: Callable[[], dict[str, Any]] | None = None,
    get_sprite_info: Callable[[], dict[str, Any]] | None = None,
    get_grid_info: Callable[[], dict[str, Any]] | None = None,
    get_trail_data: Callable[[], tuple[list[float], list[int]]] | None = None,
    get_trail_data_split: Callable[[], tuple[list[float], list[int], list[float], list[int]]] | None = None,
    get_trail_info: Callable[[], dict[str, Any]] | None = None,
    get_blob_positions_and_colors: Callable[[], tuple[list[float], list[int]]] | None = None,
    get_blob_positions_and_colors_split: Callable[
        [], tuple[list[float], list[int], list[float], list[int]]
    ]
    | None = None,
    get_blob_info: Callable[[], dict[str, Any]] | None = None,
    get_focus_point: Callable[[], tuple[float, float, float]] | None = None,
    get_fit_bounds: Callable[[], tuple[tuple[float, float, float], float]] | None = None,
    get_camera_limits: Callable[[], tuple[float, float]] | None = None,
    multi_view: bool | Callable[[], bool] = False,
    multi_view_count: int | Callable[[], int] = 3,
    target_fps: int,
    title: str,
    mac_compat: bool = False,
    split_screen: bool | Callable[[], bool] = False,
) -> None:
    try:
        import pyglet  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: install pyglet (pip install pyglet).") from e

    from pyglet import gl  # type: ignore
    from pyglet.math import Mat4, Vec3  # type: ignore
    from pyglet.window import mouse  # type: ignore

    if mac_compat:
        # Some macOS drivers behave better without the shadow window; also force vsync to avoid busy-looping.
        pyglet.options["shadow_window"] = False
        pyglet.options["vsync"] = True

    window = None
    config_candidates: list[dict[str, Any]] = []
    if mac_compat:
        # Prefer a cheap, non-MSAA context on macOS (faster on integrated GPUs and fixes glitches with some drivers).
        config_candidates.extend(
            [
                {"double_buffer": True, "depth_size": 24, "sample_buffers": 0, "samples": 0},
                {"double_buffer": True, "depth_size": 16, "sample_buffers": 0, "samples": 0},
            ]
        )
    # Cross-platform defaults: try 4x MSAA, then fall back to a plain depth buffer.
    config_candidates.extend(
        [
            {"double_buffer": True, "depth_size": 24, "sample_buffers": 1, "samples": 4},
            {"double_buffer": True, "depth_size": 24},
        ]
    )

    for cfg_kwargs in config_candidates:
        try:
            config = gl.Config(**cfg_kwargs)
            window = pyglet.window.Window(
                width=width,
                height=height,
                caption=title,
                config=config,
                resizable=True,
                vsync=True,
            )
            break
        except Exception:
            continue

    if window is None:
        window = pyglet.window.Window(width=width, height=height, caption=title, resizable=True, vsync=True)

    bg_r, bg_g, bg_b = background_rgb
    gl.glClearColor(bg_r / 255.0, bg_g / 255.0, bg_b / 255.0, 1.0)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_DEPTH_TEST)

    if hasattr(gl, "GL_POINT_SMOOTH"):
        if mac_compat:
            gl.glDisable(gl.GL_POINT_SMOOTH)
        else:
            gl.glEnable(gl.GL_POINT_SMOOTH)
    if mac_compat and hasattr(gl, "GL_MULTISAMPLE"):
        gl.glDisable(gl.GL_MULTISAMPLE)
    gl.glPointSize(float(get_point_size()) if get_point_size is not None else 3.0)

    fps_display = pyglet.window.FPSDisplay(window)
    fps_display.label.anchor_x = "right"
    fps_display.label.anchor_y = "top"
    fps_display.label.x = window.width - 10
    fps_display.label.y = window.height - 10

    help_label = pyglet.text.Label(
        "F1/TAB menu | ENTER edit | PGUP/PGDN step | V HUD | X filter | B bound wire | G focus | H fit | T follow | O record | P screenshot | SPACE pause | R reset | S save | L load | +/- speed | Click view to select | Mouse drag rotate | Shift+drag/Right drag pan | Wheel zoom",
        x=10,
        y=10,
        anchor_x="left",
        anchor_y="bottom",
        font_name="Segoe UI",
        font_size=13,
        color=(235, 240, 255, 245),
    )
    help_panel = pyglet.shapes.BorderedRectangle(
        8,
        8,
        10,
        10,
        border=1,
        color=(0, 0, 0, 170),
        border_color=(170, 190, 210),
    )
    record_label = pyglet.text.Label(
        "",
        anchor_x="right",
        anchor_y="top",
        font_name="Segoe UI",
        font_size=12,
        color=(255, 90, 90, 255),
    )
    record_panel = pyglet.shapes.BorderedRectangle(
        8,
        8,
        10,
        10,
        border=1,
        color=(0, 0, 0, 180),
        border_color=(200, 80, 80),
    )

    menu_label = pyglet.text.Label(
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

    menu_panel = pyglet.shapes.BorderedRectangle(
        8,
        8,
        10,
        10,
        border=1,
        color=(0, 0, 0, 210),
        border_color=(255, 255, 255),
    )

    legend_panel = None
    legend_pos_label = None
    legend_neg_label = None
    legend_scale_label = None
    legend_pos_swatch = None
    legend_neg_swatch = None
    legend_grad_rects: list[Any] = []
    active_view_edges: list[Any] = []
    active_view_label = None

    line_program = pyglet.graphics.get_default_shader()
    sprite_program = None
    sprite_failed = False
    point_mode = "default"
    vertex_list = None
    vertex_count = 0
    vertex_list_pos = None
    vertex_list_neg = None
    vertex_count_pos = 0
    vertex_count_neg = 0
    vertex_list_blob = None
    vertex_count_blob = 0
    vertex_list_blob_pos = None
    vertex_list_blob_neg = None
    vertex_count_blob_pos = 0
    vertex_count_blob_neg = 0
    trail_list = None
    trail_count = 0
    trail_list_pos = None
    trail_list_neg = None
    trail_count_pos = 0
    trail_count_neg = 0
    grid_list = None
    grid_count = 0
    grid_key: tuple[float, float] | None = None
    line_width_range: tuple[float, float] | None = None
    wire_cache_key: tuple[float, float] | None = None
    wire_cache_alpha: float = 0.0
    wire_cache_vls: list[Any] = []

    view_count = 1
    active_view = 0
    camera_yaw = [0.9]
    camera_pitch = [0.25]
    camera_distance = [1400.0]
    camera_center = [Vec3(0.0, 0.0, 0.0)]
    camera_follow = [False]
    camera_follow_offset = [Vec3(0.0, 0.0, 0.0)]
    recording = False
    record_dir: Path | None = None
    record_frame = 0
    record_fps = max(10, int(target_fps))
    ffmpeg_path = shutil.which("ffmpeg")
    view_w = float(width)
    view_h = float(height)

    def ensure_view_state(count: int) -> None:
        nonlocal view_count, active_view
        count = max(1, int(count))
        view_count = count
        while len(camera_yaw) < count:
            camera_yaw.append(camera_yaw[-1])
            camera_pitch.append(camera_pitch[-1])
            camera_distance.append(camera_distance[-1])
            camera_center.append(camera_center[-1])
            camera_follow.append(False)
            camera_follow_offset.append(Vec3(0.0, 0.0, 0.0))
        if len(camera_yaw) > count:
            del camera_yaw[count:]
            del camera_pitch[count:]
            del camera_distance[count:]
            del camera_center[count:]
            del camera_follow[count:]
            del camera_follow_offset[count:]
        if active_view >= count:
            active_view = max(0, count - 1)

    def resolve_camera_limits() -> tuple[float, float]:
        min_d = 0.2
        max_d = 15000.0
        if get_camera_limits is not None:
            try:
                min_d, max_d = get_camera_limits()
            except Exception:
                min_d, max_d = 0.2, 15000.0
        try:
            min_d = float(min_d)
            max_d = float(max_d)
        except Exception:
            min_d, max_d = 0.2, 15000.0
        min_d = max(0.01, min_d)
        max_d = max(min_d, max_d)
        return min_d, max_d

    def resolve_multi_view() -> tuple[bool, int]:
        enabled = multi_view() if callable(multi_view) else bool(multi_view)
        count = 1
        if enabled:
            raw_count = multi_view_count() if callable(multi_view_count) else multi_view_count
            try:
                count = int(raw_count)
            except Exception:
                count = 1
            count = max(1, min(3, count))
        ensure_view_state(count)
        return enabled, count

    def view_index_from_x(x: int, count: int) -> int:
        if count <= 1:
            return 0
        slice_w = max(1.0, float(window.width) / float(count))
        idx = int(float(x) / slice_w)
        return max(0, min(count - 1, idx))

    def viewport_for_index(idx: int, count: int) -> tuple[int, int, int, int]:
        if count <= 1:
            return 0, 0, window.width, window.height
        slice_w = float(window.width) / float(count)
        x0 = int(round(idx * slice_w))
        x1 = int(round((idx + 1) * slice_w))
        return x0, 0, max(1, x1 - x0), window.height

    def apply_3d_camera(view_idx: int) -> None:
        aspect = view_w / max(1.0, float(view_h))
        dist = max(0.01, float(camera_distance[view_idx]))
        z_near = max(0.01, min(1.0, dist * 0.1))
        window.projection = Mat4.perspective_projection(aspect=aspect, z_near=z_near, z_far=50000.0, fov=60)
        cp = math.cos(camera_pitch[view_idx])
        sp = math.sin(camera_pitch[view_idx])
        cy = math.cos(camera_yaw[view_idx])
        sy = math.sin(camera_yaw[view_idx])
        offset = Vec3(dist * cp * cy, dist * sp, dist * cp * sy)
        center = camera_center[view_idx]
        pos = center + offset
        window.view = Mat4.look_at(pos, center, Vec3(0.0, 1.0, 0.0))

    def camera_basis(view_idx: int) -> tuple[Vec3, Vec3]:
        cp = math.cos(camera_pitch[view_idx])
        sp = math.sin(camera_pitch[view_idx])
        cy = math.cos(camera_yaw[view_idx])
        sy = math.sin(camera_yaw[view_idx])
        forward = Vec3(-cp * cy, -sp, -cp * sy)
        right = Vec3(-forward.z, 0.0, forward.x)
        rlen = max(1e-6, math.sqrt((right.x * right.x) + (right.y * right.y) + (right.z * right.z)))
        right = right / rlen
        up = right.cross(forward)
        ulen = max(1e-6, math.sqrt((up.x * up.x) + (up.y * up.y) + (up.z * up.z)))
        up = up / ulen
        return right, up

    def draw_bound_wire() -> None:
        nonlocal wire_cache_key, wire_cache_alpha, wire_cache_vls
        if get_bound_visual is None:
            return
        bound_mode, visible, alpha, radius, flat_z = get_bound_visual()
        alpha = max(0.0, min(1.0, float(alpha)))
        if not visible or bound_mode != "sphere" or alpha <= 0.0:
            if wire_cache_vls:
                for vl in wire_cache_vls:
                    vl.delete()
                wire_cache_vls.clear()
                wire_cache_key = None
            return

        r = max(1.0, float(radius))
        fz = max(0.05, float(flat_z))
        key = (round(r, 6), round(fz, 6))

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glLineWidth(1.0)
        line_program.use()

        if wire_cache_key != key:
            for vl in wire_cache_vls:
                vl.delete()
            wire_cache_vls.clear()

            segments = 48
            loops: list[list[float]] = []

            # latitude rings
            for k in (-0.66, -0.33, 0.0, 0.33, 0.66):
                z = r * fz * k
                rho = r * math.sqrt(max(0.0, 1.0 - k * k))
                pts: list[float] = []
                for i in range(segments + 1):
                    a = (2.0 * math.pi * i) / segments
                    x = rho * math.cos(a)
                    y = rho * math.sin(a)
                    pts.extend((x, y, z))
                loops.append(pts)

            # meridians
            for i in range(12):
                a = (2.0 * math.pi * i) / 12.0
                pts: list[float] = []
                for k in range(0, segments + 1):
                    t = (2.0 * math.pi * k) / segments
                    x = r * math.cos(a) * math.sin(t)
                    y = r * math.sin(a) * math.sin(t)
                    z = r * fz * math.cos(t)
                    pts.extend((x, y, z))
                loops.append(pts)

            color_f = (0.8, 0.9, 1.0, alpha)
            for pts in loops:
                n = len(pts) // 3
                if n < 2:
                    continue
                vl = line_program.vertex_list(
                    n,
                    gl.GL_LINE_STRIP,
                    position=("f", pts),
                    colors=("f", list(color_f) * n),
                )
                wire_cache_vls.append(vl)
            wire_cache_key = key
            wire_cache_alpha = alpha
        elif abs(alpha - wire_cache_alpha) > 1e-6:
            color_f = [0.8, 0.9, 1.0, alpha]
            for vl in wire_cache_vls:
                count = len(vl.colors) // 4
                vl.colors[:] = color_f * count
            wire_cache_alpha = alpha

        for vl in wire_cache_vls:
            vl.draw(gl.GL_LINE_STRIP)

        line_program.stop()

    def apply_2d_overlay() -> None:
        window.projection = Mat4.orthogonal_projection(0, max(window.width, 1), 0, max(window.height, 1), -1, 1)
        window.view = Mat4()

    def ensure_sprite_program() -> Any | None:
        nonlocal sprite_program, sprite_failed
        if sprite_program is not None or sprite_failed:
            return sprite_program
        try:
            from pyglet.graphics.shader import Shader, ShaderProgram  # type: ignore

            vert_src = """#version 330 core
            in vec3 position;
            in vec4 colors;
            uniform mat4 projection;
            uniform mat4 view;
            uniform float point_size;
            out vec4 v_color;
            void main() {
                gl_Position = projection * view * vec4(position, 1.0);
                gl_PointSize = point_size;
                v_color = colors;
            }
            """
            frag_src = """#version 330 core
            in vec4 v_color;
            out vec4 fragColor;
            void main() {
                vec2 uv = gl_PointCoord * 2.0 - 1.0;
                float r2 = dot(uv, uv);
                if (r2 > 1.0) discard;
                float alpha = exp(-r2 * 2.5);
                fragColor = vec4(v_color.rgb, v_color.a * alpha);
            }
            """
            sprite_program = ShaderProgram(Shader(vert_src, "vertex"), Shader(frag_src, "fragment"))
            return sprite_program
        except Exception as exc:
            sprite_failed = True
            print(f"[sprite] init failed: {exc}")
            return None

    def clear_point_lists() -> None:
        nonlocal vertex_list, vertex_list_pos, vertex_list_neg, vertex_count, vertex_count_pos, vertex_count_neg
        nonlocal vertex_list_blob, vertex_list_blob_pos, vertex_list_blob_neg
        nonlocal vertex_count_blob, vertex_count_blob_pos, vertex_count_blob_neg
        for vl in (
            vertex_list,
            vertex_list_pos,
            vertex_list_neg,
            vertex_list_blob,
            vertex_list_blob_pos,
            vertex_list_blob_neg,
        ):
            if vl is not None:
                vl.delete()
        vertex_list = None
        vertex_list_pos = None
        vertex_list_neg = None
        vertex_count = 0
        vertex_count_pos = 0
        vertex_count_neg = 0
        vertex_list_blob = None
        vertex_list_blob_pos = None
        vertex_list_blob_neg = None
        vertex_count_blob = 0
        vertex_count_blob_pos = 0
        vertex_count_blob_neg = 0

    def clear_trail_lists() -> None:
        nonlocal trail_list, trail_list_pos, trail_list_neg, trail_count, trail_count_pos, trail_count_neg
        for vl in (trail_list, trail_list_pos, trail_list_neg):
            if vl is not None:
                vl.delete()
        trail_list = None
        trail_list_pos = None
        trail_list_neg = None
        trail_count = 0
        trail_count_pos = 0
        trail_count_neg = 0

    def legend_color_at(t: float, stops: list[tuple[float, tuple[int, int, int]]]) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, float(t)))
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t <= t1:
                local = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                r = int(round(c0[0] + (c1[0] - c0[0]) * local))
                g = int(round(c0[1] + (c1[1] - c0[1]) * local))
                b = int(round(c0[2] + (c1[2] - c0[2]) * local))
                return r, g, b
        return stops[-1][1]

    def draw_grid() -> None:
        nonlocal grid_list, grid_count, grid_key
        if get_grid_info is None:
            return
        info = get_grid_info()
        if not info or not bool(info.get("enabled", False)):
            if grid_list is not None:
                grid_list.delete()
                grid_list = None
                grid_key = None
                grid_count = 0
            return
        size = max(1.0, float(info.get("size", 100.0)))
        step = max(1.0, float(info.get("step", 50.0)))
        key = (round(size, 3), round(step, 3))
        if grid_key != key:
            if grid_list is not None:
                grid_list.delete()
            grid_key = key
            grid_count = 0

            verts: list[float] = []
            colors: list[int] = []
            max_n = int(size / step)
            base = (90, 120, 150, 70)
            axis = (170, 200, 230, 120)

            for i in range(-max_n, max_n + 1):
                v = i * step
                color = axis if i == 0 else base
                # lines parallel to Y
                verts.extend((v, -size, 0.0, v, size, 0.0))
                colors.extend(color * 2)
                # lines parallel to X
                verts.extend((-size, v, 0.0, size, v, 0.0))
                colors.extend(color * 2)

            grid_count = len(verts) // 3
            if grid_count:
                grid_list = line_program.vertex_list(
                    grid_count,
                    gl.GL_LINES,
                    position=("f", verts),
                    colors=("Bn", colors),
                )

        if grid_list is not None:
            line_program.use()
            grid_list.draw(gl.GL_LINES)
            line_program.stop()

    def safe_line_width(width: float) -> float:
        nonlocal line_width_range
        if line_width_range is None:
            try:
                buf = (gl.GLfloat * 2)()
                gl.glGetFloatv(gl.GL_ALIASED_LINE_WIDTH_RANGE, buf)
                line_width_range = (float(buf[0]), float(buf[1]))
            except Exception:
                line_width_range = (1.0, 1.0)
        min_w, max_w = line_width_range
        width = max(min_w, min(max_w, float(width)))
        return max(1.0, width)

    def start_recording() -> None:
        nonlocal recording, record_dir, record_frame
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("particle_sim3d") / "output"
        record_dir = out_root / f"video_{stamp}"
        try:
            record_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[video] failed to create output dir: {exc}")
            record_dir = None
            return
        record_frame = 0
        recording = True
        print(f"[video] recording frames to {record_dir}")

    def stop_recording() -> None:
        nonlocal recording, record_dir, record_frame
        recording = False
        if record_dir is None:
            return
        if record_frame == 0:
            print("[video] no frames recorded.")
            return
        out_path = record_dir.with_suffix(".mp4")
        if ffmpeg_path is None:
            print("[video] ffmpeg not found; frames kept for manual encoding.")
            print(f"[video] frames dir: {record_dir}")
            return
        cmd = [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(record_fps),
            "-start_number",
            "0",
            "-i",
            str(record_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"[video] saved to {out_path}")
        except Exception as exc:
            print(f"[video] ffmpeg failed: {exc}")

    @window.event
    def on_draw() -> None:
        nonlocal vertex_list, vertex_count, vertex_list_pos, vertex_count_pos, vertex_list_neg, vertex_count_neg, view_w, view_h
        nonlocal trail_list, trail_count, trail_list_pos, trail_list_neg, trail_count_pos, trail_count_neg
        nonlocal point_mode
        nonlocal camera_follow, camera_center, camera_follow_offset
        nonlocal recording, record_dir, record_frame
        nonlocal legend_panel, legend_pos_label, legend_neg_label, legend_scale_label
        nonlocal legend_pos_swatch, legend_neg_swatch, legend_grad_rects
        nonlocal active_view_edges, active_view_label
        window.clear()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        multi_view_enabled, active_count = resolve_multi_view()

        if get_focus_point is not None and any(camera_follow[:active_count]):
            fx, fy, fz = get_focus_point()
            focus = Vec3(float(fx), float(fy), float(fz))
            for view_idx in range(active_count):
                if camera_follow[view_idx]:
                    camera_center[view_idx] = focus + camera_follow_offset[view_idx]

        point_size = float(get_point_size()) if get_point_size is not None else 3.0
        sprite_enabled = False
        sprite_scale = 1.0
        blob_enabled = False
        blob_scale = 1.0
        if get_sprite_info is not None:
            sprite_info = get_sprite_info()
            if sprite_info:
                sprite_enabled = bool(sprite_info.get("enabled", False))
                sprite_scale = float(sprite_info.get("scale", 2.5))
        if get_blob_info is not None:
            blob_info = get_blob_info()
            if blob_info:
                blob_enabled = bool(blob_info.get("enabled", False))
                blob_scale = float(blob_info.get("scale", 2.5))
                blob_scale = max(1.0, min(10.0, blob_scale))

        point_program = line_program
        if sprite_enabled:
            sprite_program_local = ensure_sprite_program()
            if sprite_program_local is None:
                sprite_enabled = False
            else:
                point_program = sprite_program_local

        new_mode = "sprite" if sprite_enabled else "default"
        if new_mode != point_mode:
            clear_point_lists()
            point_mode = new_mode

        if sprite_enabled:
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            point_size = max(1.0, point_size * sprite_scale)
        else:
            if hasattr(gl, "GL_PROGRAM_POINT_SIZE"):
                gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)
            gl.glPointSize(point_size)

        gl.glEnable(gl.GL_DEPTH_TEST)

        def draw_points(xyz: list[float], rgba: list[int], cache: str, size_scale: float = 1.0) -> None:
            nonlocal vertex_list, vertex_count, vertex_list_pos, vertex_list_neg, vertex_count_pos, vertex_count_neg
            nonlocal vertex_list_blob, vertex_list_blob_pos, vertex_list_blob_neg
            nonlocal vertex_count_blob, vertex_count_blob_pos, vertex_count_blob_neg
            if not xyz:
                return
            count = len(xyz) // 3
            point_program.use()
            if sprite_enabled:
                point_program["projection"] = window.projection
                point_program["view"] = window.view
                point_program["point_size"] = point_size * size_scale
            else:
                gl.glPointSize(max(1.0, point_size * size_scale))
            if cache == "pos":
                if vertex_list_pos is None:
                    vertex_count_pos = count
                    vertex_list_pos = point_program.vertex_list(
                        vertex_count_pos,
                        gl.GL_POINTS,
                        position=("f", xyz),
                        colors=("Bn", rgba),
                    )
                else:
                    if count != vertex_count_pos:
                        vertex_count_pos = count
                        vertex_list_pos.resize(vertex_count_pos)
                    vertex_list_pos.position[:] = xyz
                    vertex_list_pos.colors[:] = rgba
                vertex_list_pos.draw(gl.GL_POINTS)
            elif cache == "neg":
                if vertex_list_neg is None:
                    vertex_count_neg = count
                    vertex_list_neg = point_program.vertex_list(
                        vertex_count_neg,
                        gl.GL_POINTS,
                        position=("f", xyz),
                        colors=("Bn", rgba),
                    )
                else:
                    if count != vertex_count_neg:
                        vertex_count_neg = count
                        vertex_list_neg.resize(vertex_count_neg)
                    vertex_list_neg.position[:] = xyz
                    vertex_list_neg.colors[:] = rgba
                vertex_list_neg.draw(gl.GL_POINTS)
            elif cache == "blob":
                if vertex_list_blob is None:
                    vertex_count_blob = count
                    vertex_list_blob = point_program.vertex_list(
                        vertex_count_blob,
                        gl.GL_POINTS,
                        position=("f", xyz),
                        colors=("Bn", rgba),
                    )
                else:
                    if count != vertex_count_blob:
                        vertex_count_blob = count
                        vertex_list_blob.resize(vertex_count_blob)
                    vertex_list_blob.position[:] = xyz
                    vertex_list_blob.colors[:] = rgba
                vertex_list_blob.draw(gl.GL_POINTS)
            elif cache == "blob_pos":
                if vertex_list_blob_pos is None:
                    vertex_count_blob_pos = count
                    vertex_list_blob_pos = point_program.vertex_list(
                        vertex_count_blob_pos,
                        gl.GL_POINTS,
                        position=("f", xyz),
                        colors=("Bn", rgba),
                    )
                else:
                    if count != vertex_count_blob_pos:
                        vertex_count_blob_pos = count
                        vertex_list_blob_pos.resize(vertex_count_blob_pos)
                    vertex_list_blob_pos.position[:] = xyz
                    vertex_list_blob_pos.colors[:] = rgba
                vertex_list_blob_pos.draw(gl.GL_POINTS)
            elif cache == "blob_neg":
                if vertex_list_blob_neg is None:
                    vertex_count_blob_neg = count
                    vertex_list_blob_neg = point_program.vertex_list(
                        vertex_count_blob_neg,
                        gl.GL_POINTS,
                        position=("f", xyz),
                        colors=("Bn", rgba),
                    )
                else:
                    if count != vertex_count_blob_neg:
                        vertex_count_blob_neg = count
                        vertex_list_blob_neg.resize(vertex_count_blob_neg)
                    vertex_list_blob_neg.position[:] = xyz
                    vertex_list_blob_neg.colors[:] = rgba
                vertex_list_blob_neg.draw(gl.GL_POINTS)
            else:
                if vertex_list is None:
                    vertex_count = count
                    vertex_list = point_program.vertex_list(
                        vertex_count,
                        gl.GL_POINTS,
                        position=("f", xyz),
                        colors=("Bn", rgba),
                    )
                else:
                    if count != vertex_count:
                        vertex_count = count
                        vertex_list.resize(vertex_count)
                    vertex_list.position[:] = xyz
                    vertex_list.colors[:] = rgba
                vertex_list.draw(gl.GL_POINTS)
            point_program.stop()

        def draw_trails(xyz: list[float], rgba: list[int], cache: str) -> None:
            nonlocal trail_list, trail_list_pos, trail_list_neg, trail_count, trail_count_pos, trail_count_neg
            if not xyz:
                return
            count = len(xyz) // 3
            blur_enabled = False
            trail_width = 1.0
            if get_trail_info is not None:
                info = get_trail_info()
                if info:
                    blur_enabled = bool(info.get("blur", False))
                    trail_width = float(info.get("width", 1.5))
            if blur_enabled and hasattr(gl, "GL_LINE_SMOOTH"):
                gl.glEnable(gl.GL_LINE_SMOOTH)
                if hasattr(gl, "GL_LINE_SMOOTH_HINT") and hasattr(gl, "GL_NICEST"):
                    gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            try:
                gl.glLineWidth(safe_line_width(trail_width))
            except Exception:
                gl.glLineWidth(1.0)
            if cache == "pos":
                if trail_list_pos is None:
                    trail_count_pos = count
                    trail_list_pos = line_program.vertex_list(
                        trail_count_pos,
                        gl.GL_LINES,
                        position=("f", xyz),
                        colors=("Bn", rgba),
                    )
                else:
                    if count != trail_count_pos:
                        trail_count_pos = count
                        trail_list_pos.resize(trail_count_pos)
                    trail_list_pos.position[:] = xyz
                    trail_list_pos.colors[:] = rgba
                line_program.use()
                trail_list_pos.draw(gl.GL_LINES)
                line_program.stop()
            elif cache == "neg":
                if trail_list_neg is None:
                    trail_count_neg = count
                    trail_list_neg = line_program.vertex_list(
                        trail_count_neg,
                        gl.GL_LINES,
                        position=("f", xyz),
                        colors=("Bn", rgba),
                    )
                else:
                    if count != trail_count_neg:
                        trail_count_neg = count
                        trail_list_neg.resize(trail_count_neg)
                    trail_list_neg.position[:] = xyz
                    trail_list_neg.colors[:] = rgba
                line_program.use()
                trail_list_neg.draw(gl.GL_LINES)
                line_program.stop()
            else:
                if trail_list is None:
                    trail_count = count
                    trail_list = line_program.vertex_list(
                        trail_count,
                        gl.GL_LINES,
                        position=("f", xyz),
                        colors=("Bn", rgba),
                    )
                else:
                    if count != trail_count:
                        trail_count = count
                        trail_list.resize(trail_count)
                    trail_list.position[:] = xyz
                    trail_list.colors[:] = rgba
                line_program.use()
                trail_list.draw(gl.GL_LINES)
                line_program.stop()
            if blur_enabled and hasattr(gl, "GL_LINE_SMOOTH"):
                gl.glDisable(gl.GL_LINE_SMOOTH)
            try:
                gl.glLineWidth(safe_line_width(1.0))
            except Exception:
                gl.glLineWidth(1.0)

        split_screen_enabled = (not multi_view_enabled) and (
            split_screen() if callable(split_screen) else bool(split_screen)
        )
        trail_all = None
        trail_pos = trail_neg = None
        if split_screen_enabled and get_trail_data_split is not None:
            trail_pos, trail_pos_rgba, trail_neg, trail_neg_rgba = get_trail_data_split()
        elif get_trail_data is not None:
            trail_all = get_trail_data()

        blob_all = None
        blob_pos = blob_neg = None
        blob_pos_rgba = blob_neg_rgba = None
        if blob_enabled:
            if split_screen_enabled and get_blob_positions_and_colors_split is not None:
                blob_pos, blob_pos_rgba, blob_neg, blob_neg_rgba = get_blob_positions_and_colors_split()
            elif get_blob_positions_and_colors is not None:
                blob_all = get_blob_positions_and_colors()

        if multi_view_enabled:
            xyz, rgba = get_positions_and_colors()
            for view_idx in range(active_count):
                vx, vy, vw, vh = viewport_for_index(view_idx, active_count)
                view_w = float(vw)
                view_h = float(vh)
                gl.glViewport(vx, vy, vw, vh)
                apply_3d_camera(view_idx)
                draw_grid()
                if trail_all is not None:
                    draw_trails(trail_all[0], trail_all[1], "all")
                draw_points(xyz, rgba, "all")
                if blob_all is not None:
                    draw_points(blob_all[0], blob_all[1], "blob", blob_scale)
                draw_bound_wire()
        elif split_screen_enabled and get_positions_and_colors_split is not None and window.width >= 2:
            half_w = max(1, window.width // 2)
            xyz_pos, rgba_pos, xyz_neg, rgba_neg = get_positions_and_colors_split()
            # left: M+, right: M-
            view_w = float(half_w)
            view_h = float(window.height)
            gl.glViewport(0, 0, half_w, window.height)
            apply_3d_camera(0)
            draw_grid()
            if trail_pos is not None:
                draw_trails(trail_pos, trail_pos_rgba, "pos")
            elif trail_all is not None:
                draw_trails(trail_all[0], trail_all[1], "all")
            draw_points(xyz_pos, rgba_pos, "pos")
            if blob_pos is not None:
                draw_points(blob_pos, blob_pos_rgba, "blob_pos", blob_scale)
            elif blob_all is not None:
                draw_points(blob_all[0], blob_all[1], "blob", blob_scale)

            view_w = float(window.width - half_w)
            view_h = float(window.height)
            gl.glViewport(half_w, 0, window.width - half_w, window.height)
            apply_3d_camera(0)
            draw_grid()
            if trail_neg is not None:
                draw_trails(trail_neg, trail_neg_rgba, "neg")
            elif trail_all is not None:
                draw_trails(trail_all[0], trail_all[1], "all")
            draw_points(xyz_neg, rgba_neg, "neg")
            if blob_neg is not None:
                draw_points(blob_neg, blob_neg_rgba, "blob_neg", blob_scale)
            elif blob_all is not None:
                draw_points(blob_all[0], blob_all[1], "blob", blob_scale)
        else:
            view_w = float(window.width)
            view_h = float(window.height)
            gl.glViewport(0, 0, window.width, window.height)
            apply_3d_camera(0)
            draw_grid()
            if trail_all is not None:
                draw_trails(trail_all[0], trail_all[1], "all")
            xyz, rgba = get_positions_and_colors()
            draw_points(xyz, rgba, "all")
            if blob_all is not None:
                draw_points(blob_all[0], blob_all[1], "blob", blob_scale)

        gl.glViewport(0, 0, window.width, window.height)
        view_w = float(window.width)
        view_h = float(window.height)

        if not multi_view_enabled:
            # Draw bound wireframe even if there are zero/hidden particles.
            draw_bound_wire()

        gl.glDisable(gl.GL_DEPTH_TEST)
        apply_2d_overlay()
        if multi_view_enabled:
            vx, vy, vw, vh = viewport_for_index(active_view, active_count)
            thickness = 2
            if not active_view_edges or len(active_view_edges) != 4:
                for rect in active_view_edges:
                    rect.delete()
                active_view_edges = [
                    pyglet.shapes.Rectangle(0, 0, thickness, thickness, color=(120, 200, 255))
                    for _ in range(4)
                ]
            edge_color = (120, 200, 255)
            vx_i = int(vx)
            vy_i = int(vy)
            vw_i = max(1, int(vw))
            vh_i = max(1, int(vh))
            left, right, bottom, top = active_view_edges
            left.x = vx_i
            left.y = vy_i
            left.width = thickness
            left.height = vh_i
            right.x = max(vx_i, vx_i + vw_i - thickness)
            right.y = vy_i
            right.width = thickness
            right.height = vh_i
            bottom.x = vx_i
            bottom.y = vy_i
            bottom.width = vw_i
            bottom.height = thickness
            top.x = vx_i
            top.y = max(vy_i, vy_i + vh_i - thickness)
            top.width = vw_i
            top.height = thickness
            for rect in active_view_edges:
                rect.color = edge_color
                rect.opacity = 210
                rect.draw()

            if active_view_label is None:
                active_view_label = pyglet.text.Label(
                    "",
                    anchor_x="left",
                    anchor_y="top",
                    font_name="Segoe UI",
                    font_size=12,
                    color=(210, 235, 255, 245),
                )
            active_view_label.text = f"View {active_view + 1}/{active_count}"
            active_view_label.x = vx_i + 8
            active_view_label.y = vy_i + vh_i - 6
            active_view_label.draw()

        fps_display.draw()
        help_margin = 8
        help_pad = 6
        help_panel.width = min(window.width - (2 * help_margin), int(help_label.content_width) + (2 * help_pad))
        help_panel.height = int(help_label.content_height) + (2 * help_pad)
        help_panel.x = help_margin
        help_panel.y = help_margin
        help_label.x = help_margin + help_pad
        help_label.y = help_margin + help_pad
        help_panel.draw()
        help_label.draw()

        if get_legend_info is not None:
            info = get_legend_info()
            if info:
                if legend_panel is None:
                    legend_panel = pyglet.shapes.BorderedRectangle(
                        8,
                        8,
                        10,
                        10,
                        border=1,
                        color=(0, 0, 0, 190),
                        border_color=(200, 210, 220),
                    )
                    legend_pos_label = pyglet.text.Label(
                        "M+",
                        anchor_x="left",
                        anchor_y="center",
                        font_name="Segoe UI",
                        font_size=12,
                        color=(230, 235, 245, 235),
                    )
                    legend_neg_label = pyglet.text.Label(
                        "M-",
                        anchor_x="left",
                        anchor_y="center",
                        font_name="Segoe UI",
                        font_size=12,
                        color=(230, 235, 245, 235),
                    )
                    legend_scale_label = pyglet.text.Label(
                        "",
                        anchor_x="left",
                        anchor_y="center",
                        font_name="Segoe UI",
                        font_size=11,
                        color=(190, 205, 220, 220),
                    )
                    legend_pos_swatch = pyglet.shapes.Rectangle(0, 0, 12, 12, color=(180, 200, 255))
                    legend_neg_swatch = pyglet.shapes.Rectangle(0, 0, 12, 12, color=(255, 180, 110))

                pos_color = info.get("pos_color", (188, 214, 255, 220))
                neg_color = info.get("neg_color", (255, 184, 107, 220))
                grad_enabled = bool(info.get("gradient_enabled", False))
                stops = info.get(
                    "gradient_stops",
                    [
                        (0.0, (40, 100, 190)),
                        (0.45, (245, 245, 245)),
                        (0.75, (255, 170, 90)),
                        (1.0, (230, 60, 50)),
                    ],
                )

                pad = 8
                row_h = 18
                swatch_w = 12
                swatch_h = 12
                grad_w = 90
                grad_h = 8
                label_gap = 6

                grad_label = info.get("gradient_label", "M+ gradient")
                legend_pos_label.text = grad_label if grad_enabled else "M+"
                legend_neg_label.text = "M-"
                range_text = info.get("gradient_range_text")
                show_scale = bool(grad_enabled and range_text)
                if legend_scale_label is not None:
                    legend_scale_label.text = range_text or ""
                label_w = max(legend_pos_label.content_width, legend_neg_label.content_width)
                if show_scale and legend_scale_label is not None:
                    label_w = max(label_w, legend_scale_label.content_width)
                panel_w = max(140, int((2 * pad) + label_gap + label_w + (grad_w if grad_enabled else swatch_w)))
                avail_w = max(1, window.width - (2 * pad))
                panel_w = min(panel_w, avail_w)
                panel_h = (2 * pad) + (row_h * (3 if show_scale else 2))

                base_x = max(pad, window.width - panel_w - pad)
                base_y = pad
                legend_panel.x = base_x
                legend_panel.y = base_y
                legend_panel.width = panel_w
                legend_panel.height = panel_h
                legend_panel.draw()

                row1_y = base_y + panel_h - pad - row_h
                row2_y = row1_y - row_h
                row3_y = row2_y - row_h if show_scale else row2_y
                label_x = base_x + pad + (grad_w if grad_enabled else swatch_w) + label_gap

                legend_pos_label.x = label_x
                legend_pos_label.y = row1_y + (row_h / 2)
                legend_neg_label.x = label_x
                legend_neg_label.y = row3_y + (row_h / 2)
                if show_scale and legend_scale_label is not None:
                    legend_scale_label.x = base_x + pad
                    legend_scale_label.y = row2_y + (row_h / 2)

                if grad_enabled:
                    seg_count = 24
                    if not legend_grad_rects or len(legend_grad_rects) != seg_count:
                        for rect in legend_grad_rects:
                            rect.delete()
                        legend_grad_rects = []
                        for _ in range(seg_count):
                            legend_grad_rects.append(pyglet.shapes.Rectangle(0, 0, 1, grad_h, color=(255, 255, 255)))
                    seg_w = grad_w / seg_count
                    grad_x = base_x + pad
                    grad_y = row1_y + ((row_h - grad_h) / 2)
                    for i, rect in enumerate(legend_grad_rects):
                        rect.x = grad_x + (i * seg_w)
                        rect.y = grad_y
                        rect.width = seg_w + 0.5
                        rect.height = grad_h
                        r, g, b = legend_color_at((i + 0.5) / seg_count, stops)
                        rect.color = (r, g, b)
                        rect.opacity = 230
                        rect.draw()
                    if legend_pos_swatch is not None:
                        legend_pos_swatch.opacity = 0
                else:
                    if legend_pos_swatch is not None:
                        legend_pos_swatch.x = base_x + pad
                        legend_pos_swatch.y = row1_y + ((row_h - swatch_h) / 2)
                        legend_pos_swatch.width = swatch_w
                        legend_pos_swatch.height = swatch_h
                        legend_pos_swatch.color = tuple(pos_color[:3])
                        legend_pos_swatch.opacity = 230
                        legend_pos_swatch.draw()

                if legend_neg_swatch is not None:
                    legend_neg_swatch.x = base_x + pad
                    legend_neg_swatch.y = row3_y + ((row_h - swatch_h) / 2)
                    legend_neg_swatch.width = swatch_w
                    legend_neg_swatch.height = swatch_h
                    legend_neg_swatch.color = tuple(neg_color[:3])
                    legend_neg_swatch.opacity = 230
                    legend_neg_swatch.draw()

                legend_pos_label.draw()
                if show_scale and legend_scale_label is not None:
                    legend_scale_label.draw()
                legend_neg_label.draw()

        if get_overlay_text is not None:
            overlay = get_overlay_text()
            if overlay:
                menu_label.text = overlay

                margin = 10
                pad = 12
                panel_w = min(window.width - (2 * margin), 700)
                menu_label.width = max(120, panel_w - (2 * pad))
                menu_label.x = margin + pad
                menu_label.y = window.height - (margin + pad)

                panel_h = int(menu_label.content_height) + (2 * pad)
                panel_h = min(panel_h, window.height - (2 * margin))
                menu_panel.x = margin
                menu_panel.y = window.height - margin - panel_h
                menu_panel.width = panel_w
                menu_panel.height = panel_h

                menu_panel.draw()
                menu_label.draw()

        if recording:
            record_label.text = f"REC {record_frame:05d}"
            pad = 6
            panel_w = int(record_label.content_width) + (2 * pad)
            panel_h = int(record_label.content_height) + (2 * pad)
            record_panel.x = window.width - panel_w - 10
            record_panel.y = window.height - panel_h - 10
            record_panel.width = panel_w
            record_panel.height = panel_h
            record_label.x = record_panel.x + panel_w - pad
            record_label.y = record_panel.y + panel_h - pad
            record_panel.draw()
            record_label.draw()

            if record_dir is not None:
                try:
                    if hasattr(gl, "glFinish"):
                        gl.glFinish()
                    buf = pyglet.image.get_buffer_manager().get_color_buffer()
                    frame_path = record_dir / f"frame_{record_frame:05d}.png"
                    buf.save(str(frame_path))
                    record_frame += 1
                except Exception as exc:
                    print(f"[video] capture failed: {exc}")
                    stop_recording()

    @window.event
    def on_resize(w: int, h: int) -> None:
        nonlocal view_w, view_h
        view_w = float(w)
        view_h = float(h)
        fps_display.label.x = w - 10
        fps_display.label.y = h - 10
        help_label.y = 10

    @window.event
    def on_close() -> None:
        nonlocal recording
        if recording:
            stop_recording()
        pyglet.app.exit()

    @window.event
    def on_key_press(symbol: int, modifiers: int) -> None:  # noqa: ARG001
        from pyglet.window import key  # type: ignore
        nonlocal camera_yaw, camera_pitch, camera_distance, camera_center, camera_follow, camera_follow_offset
        nonlocal recording

        mapping = {
            key.SPACE: "space",
            key.R: "r",
            key.S: "s",
            key.L: "l",
            key.F: "f",
            key.V: "v",
            key.X: "x",
            key.B: "b",
            key.E: "e",
            key.C: "c",
            key.G: "g",
            key.H: "h",
            key.T: "t",
            key.O: "o",
            key.P: "p",
            key._1: "1",
            key._2: "2",
            key._3: "3",
            key._4: "4",
            key._5: "5",
            key._0: "0",
            key.ESCAPE: "esc",
            key.F1: "f1",
            key.UP: "up",
            key.DOWN: "down",
            key.LEFT: "left",
            key.RIGHT: "right",
            key.BACKSPACE: "backspace",
            key.DELETE: "delete",
            key.PAGEUP: "pageup",
            key.PAGEDOWN: "pagedown",
            key.ENTER: "enter",
            key.NUM_ENTER: "enter",
            key.TAB: "tab",
            key.PLUS: "plus",
            key.EQUAL: "plus",
            key.NUM_ADD: "plus",
            key.MINUS: "minus",
            key.NUM_SUBTRACT: "minus",
        }
        k = mapping.get(symbol)
        if k is not None:
            resolve_multi_view()
            view_idx = active_view
            if k == "p":
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                candidates = [
                    Path("particle_sim3d") / "output",
                    Path.cwd() / "output",
                    Path(tempfile.gettempdir()) / "particle_sim3d",
                ]
                error: Exception | None = None
                for out_dir in candidates:
                    try:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        path = out_dir / f"screenshot_{stamp}.png"
                        window.switch_to()
                        if hasattr(gl, "glFinish"):
                            gl.glFinish()
                        buf = pyglet.image.get_buffer_manager().get_color_buffer()
                        buf.save(str(path))
                        print(f"[screenshot] saved to {path}")
                        error = None
                        break
                    except Exception as e:  # pragma: no cover - runtime env
                        error = e
                        continue
                if error is not None:
                    print(f"[screenshot] failed: {error}")
                return
            if k == "c":
                camera_yaw[view_idx] = 0.9
                camera_pitch[view_idx] = 0.25
                camera_distance[view_idx] = 1400.0
                camera_center[view_idx] = Vec3(0.0, 0.0, 0.0)
                camera_follow[view_idx] = False
                camera_follow_offset[view_idx] = Vec3(0.0, 0.0, 0.0)
                return
            if k == "g":
                if get_focus_point is None:
                    camera_center[view_idx] = Vec3(0.0, 0.0, 0.0)
                else:
                    fx, fy, fz = get_focus_point()
                    focus = Vec3(float(fx), float(fy), float(fz))
                    camera_center[view_idx] = focus
                    if camera_follow[view_idx]:
                        camera_follow_offset[view_idx] = Vec3(0.0, 0.0, 0.0)
                return
            if k == "h":
                if get_fit_bounds is not None:
                    (cx, cy, cz), radius = get_fit_bounds()
                    camera_center[view_idx] = Vec3(float(cx), float(cy), float(cz))
                    radius = max(1.0, float(radius))
                    aspect = window.width / max(1.0, float(window.height))
                    fov_y = math.radians(60.0)
                    fov_x = 2.0 * math.atan(math.tan(fov_y / 2.0) * aspect)
                    dist_y = radius / max(1e-6, math.tan(fov_y / 2.0))
                    dist_x = radius / max(1e-6, math.tan(fov_x / 2.0))
                    min_dist, max_dist = resolve_camera_limits()
                    camera_distance[view_idx] = max(dist_x, dist_y) * 1.1
                    camera_distance[view_idx] = max(min_dist, min(max_dist, camera_distance[view_idx]))
                    if camera_follow[view_idx] and get_focus_point is not None:
                        fx, fy, fz = get_focus_point()
                        focus = Vec3(float(fx), float(fy), float(fz))
                        camera_follow_offset[view_idx] = camera_center[view_idx] - focus
                return
            if k == "t":
                camera_follow[view_idx] = not camera_follow[view_idx]
                if camera_follow[view_idx] and get_focus_point is not None:
                    fx, fy, fz = get_focus_point()
                    focus = Vec3(float(fx), float(fy), float(fz))
                    camera_follow_offset[view_idx] = camera_center[view_idx] - focus
                else:
                    camera_follow_offset[view_idx] = Vec3(0.0, 0.0, 0.0)
                state = "on" if camera_follow[view_idx] else "off"
                print(f"[camera] follow {state}")
                return
            if k == "o":
                if recording:
                    stop_recording()
                else:
                    start_recording()
                return
            try:
                on_key(k)
            except SystemExit:
                pyglet.app.exit()

    @window.event
    def on_text(text: str) -> None:  # type: ignore[override]
        if on_text_input is not None and text:
            on_text_input(text)

    @window.event
    def on_mouse_press(x: int, y: int, button: int, modifiers: int) -> None:  # noqa: ARG001
        nonlocal active_view
        multi_view_enabled, count = resolve_multi_view()
        if multi_view_enabled:
            active_view = view_index_from_x(x, count)

    @window.event
    def on_mouse_drag(x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:  # noqa: ARG001
        nonlocal active_view, camera_yaw, camera_pitch, camera_center, camera_follow, camera_follow_offset
        from pyglet.window import key  # type: ignore

        multi_view_enabled, count = resolve_multi_view()
        view_idx = active_view
        if multi_view_enabled:
            view_idx = view_index_from_x(x, count)
            active_view = view_idx

        if buttons & mouse.LEFT and not (modifiers & key.MOD_SHIFT):
            camera_yaw[view_idx] += float(dx) * 0.006
            camera_pitch[view_idx] += float(dy) * 0.006
            camera_pitch[view_idx] = max(-1.35, min(1.35, camera_pitch[view_idx]))
            return

        if (buttons & mouse.RIGHT) or (buttons & mouse.LEFT and (modifiers & key.MOD_SHIFT)):
            right, up = camera_basis(view_idx)
            pan_scale = camera_distance[view_idx] * 0.0015
            delta = (-right * (float(dx) * pan_scale)) + (up * (float(dy) * pan_scale))
            if camera_follow[view_idx]:
                camera_follow_offset[view_idx] += delta
            else:
                camera_center[view_idx] += delta

    @window.event
    def on_mouse_scroll(x: int, y: int, scroll_x: float, scroll_y: float) -> None:  # noqa: ARG001
        nonlocal active_view, camera_distance
        multi_view_enabled, count = resolve_multi_view()
        view_idx = active_view
        if multi_view_enabled:
            view_idx = view_index_from_x(x, count)
            active_view = view_idx
        min_dist, max_dist = resolve_camera_limits()
        camera_distance[view_idx] *= 0.90 ** float(scroll_y)
        camera_distance[view_idx] = max(min_dist, min(max_dist, camera_distance[view_idx]))

    def tick(dt: float) -> None:
        step_simulation(dt)
        window.set_caption(get_caption() if get_caption is not None else title)

    pyglet.clock.schedule_interval(tick, 1.0 / max(10, target_fps))
    pyglet.app.run()
