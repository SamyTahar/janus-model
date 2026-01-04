from __future__ import annotations

from array import array
from collections import deque
import math
import time
from pathlib import Path
import sys

from particle_sim3d.params import Sim3DParams
from particle_sim3d.rendering.pyglet_renderer import run_pyglet
from particle_sim3d.core.sim import ParticleSim3D
from . import menu

# Import refactored modules
from particle_sim3d.utils.config_groups import (
    RESET_KEYS,
    BOUNDS_KEYS,
    GRADIENT_KEYS,
    TRAIL_KEYS,
    JANUS_DEP_KEYS,
    BOUND_DEP_KEYS,
    MENU_GROUPS,
    PARAM_HINTS,
    get_menu_group_title,
)
from particle_sim3d.rendering.trail_manager import TrailManager, POS_BASE_COLOR, NEG_BASE_COLOR
from particle_sim3d.rendering.color_mapper import (
    ColorMapper,
    GradientStats,
    gradient_color,
    normalize,
    GRADIENT_LABELS,
    DEFAULT_GRADIENT_STOPS as GRADIENT_STOPS,
    GRAD_STAT_ALPHA,
    GRAD_CELL_DIV,
)
from particle_sim3d.utils.export import export_particles_csv, export_summary

HUD_METRICS_INTERVAL_S = menu.HUD_METRICS_INTERVAL_S



class ParticleSim3DApp:
    def __init__(self) -> None:
        self.params_path = Path(__file__).resolve().parent / "params.json"
        self.params = self._load_initial_params()
        self.sim = ParticleSim3D(self.params)

        self._running = True
        self._menu_open = False
        self._menu_index = 0
        self._menu_step_scale = 1.0
        self._edit_active = False
        self._edit_key = ""
        self._edit_kind = ""
        self._edit_label = ""
        self._edit_buffer = ""
        self._edit_error = ""

        self._show_hud = True
        self._display_mode = "all"  # all | pos | neg
        self._sim_time = 0.0

        self._hud_metrics_last = 0.0
        self._hud_metrics_valid = False
        self._hud_curve: list[tuple[float, float, int]] = []
        self._hud_m2 = 0.0

        self._xyz_buffer = array("f")
        self._rgba_buffer = array("B")
        self._xyz_buffer_pos = array("f")
        self._rgba_buffer_pos = array("B")
        self._xyz_buffer_neg = array("f")
        self._rgba_buffer_neg = array("B")
        self._xyz_blob = array("f")
        self._rgba_blob = array("B")
        self._xyz_blob_pos = array("f")
        self._rgba_blob_pos = array("B")
        self._xyz_blob_neg = array("f")
        self._rgba_blob_neg = array("B")
        self._params_error = ""
        self._grad_stats: dict[str, tuple[float, float]] = {}
        self._trail_history_pos: deque[list[float]] = deque()
        self._trail_history_neg: deque[list[float]] = deque()
        self._trail_xyz = array("f")
        self._trail_rgba = array("B")
        self._trail_xyz_pos = array("f")
        self._trail_rgba_pos = array("B")
        self._trail_xyz_neg = array("f")
        self._trail_rgba_neg = array("B")

        self._last_params_mtime: float | None = self.params_path.stat().st_mtime if self.params_path.exists() else None
        self._last_autoreload = time.monotonic()

        self._reset_trails()

    def run(self) -> None:
        run_pyglet(
            width=self.params.width,
            height=self.params.height,
            background_rgb=tuple(self.params.background),  # type: ignore[arg-type]
            get_positions_and_colors=self._get_positions_and_colors,
            get_positions_and_colors_split=self._get_positions_and_colors_split,
            step_simulation=self._step,
            on_key=self._on_key,
            on_text_input=self._on_text_input,
            get_overlay_text=self._get_overlay_text,
            get_point_size=lambda: float(self.params.point_size),
            get_caption=self._get_caption,
            get_bound_visual=self._get_bound_visual,
            get_legend_info=self._get_legend_info,
            get_sprite_info=self._get_sprite_info,
            get_grid_info=self._get_grid_info,
            get_trail_data=self._get_trail_data,
            get_trail_data_split=self._get_trail_data_split,
            get_trail_info=self._get_trail_info,
            get_blob_positions_and_colors=self._get_blob_positions_and_colors,
            get_blob_positions_and_colors_split=self._get_blob_positions_and_colors_split,
            get_blob_info=self._get_blob_info,
            get_focus_point=self._get_camera_focus_point,
            get_fit_bounds=self._get_camera_fit_bounds,
            get_camera_limits=self._get_camera_limits,
            multi_view=lambda: bool(getattr(self.params, "multi_view", False)),
            multi_view_count=lambda: int(getattr(self.params, "multi_view_count", 3)),
            target_fps=self.params.target_fps,
            title="Janus galaxy 3D - pyglet/OpenGL",
            mac_compat=bool(self.params.mac_compat and sys.platform.startswith("darwin")),
            split_screen=lambda: bool(self.params.split_screen),
        )

    def _get_caption(self) -> str:
        speed = float(self.params.time_scale)
        state = "PAUSE" if not self._running else "RUN"
        n_pos, n_neg = self.sim.counts()
        mpos = float(self.params.mass_positive)
        mneg = float(self.params.mass_negative)
        ratio = (float(n_neg) * mneg) / max(1e-9, float(n_pos) * mpos)
        return (
            f"Janus A | t={self._sim_time:7.1f}s | x{speed:4.2f} | {state} | "
            f"M+={n_pos}  M-={n_neg}  (~{ratio:4.0f}x)"
        )

    def _get_bound_visual(self) -> tuple[str, bool, float, float, float]:
        return (
            str(self.params.bound_mode),
            bool(self.params.bound_wire_visible and self.params.bounds_enabled),
            float(self.params.bound_wire_opacity),
            float(self.params.bound_sphere_radius),
            float(self.params.bound_sphere_flatten_z),
        )

    def _get_legend_info(self) -> dict[str, object]:
        mode = str(getattr(self.params, "color_gradient_mode", "mix")).strip().lower()
        label = GRADIENT_LABELS.get(mode, "M+ gradient")
        return {
            "pos_color": POS_BASE_COLOR,
            "neg_color": NEG_BASE_COLOR,
            "gradient_enabled": bool(self.params.color_gradient),
            "gradient_stops": GRADIENT_STOPS,
            "gradient_label": label,
            "gradient_range_text": self._gradient_range_text(mode) if bool(self.params.color_gradient) else None,
        }

    def _center_all_particles(self) -> tuple[float, float, float]:
        if not self.sim.particles:
            return 0.0, 0.0, 0.0
        sx = sy = sz = 0.0
        for pt in self.sim.particles:
            sx += pt.x
            sy += pt.y
            sz += pt.z
        n = float(len(self.sim.particles))
        return sx / n, sy / n, sz / n

    def _get_camera_focus_point(self) -> tuple[float, float, float]:
        n_pos, _n_neg = self.sim.counts()
        if n_pos <= 0:
            return self._center_all_particles()
        cx, cy, cz, _cvx, _cvy, _cvz = self.sim.positive_center_velocity()
        return cx, cy, cz

    def _get_camera_fit_bounds(self) -> tuple[tuple[float, float, float], float]:
        if bool(self.params.bounds_enabled):
            if str(self.params.bound_mode) == "box":
                radius = float(self.params.bounds) * math.sqrt(3.0)
            else:
                radius = float(self.params.bound_sphere_radius) * max(1.0, float(self.params.bound_sphere_flatten_z))
            return (0.0, 0.0, 0.0), radius
        center = self._center_all_particles()
        max_r2 = 1.0
        cx, cy, cz = center
        for pt in self.sim.particles:
            dx = pt.x - cx
            dy = pt.y - cy
            dz = pt.z - cz
            r2 = (dx * dx) + (dy * dy) + (dz * dz)
            if r2 > max_r2:
                max_r2 = r2
        return center, math.sqrt(max_r2)

    def _load_initial_params(self) -> Sim3DParams:
        try:
            if self.params_path.exists():
                return Sim3DParams.load(self.params_path)
        except Exception:
            pass
        return Sim3DParams().clamp()

    def _maybe_autoreload(self) -> None:
        if not self.params_path.exists():
            return
        now = time.monotonic()
        if (now - self._last_autoreload) < 0.5:
            return
        self._last_autoreload = now

        mtime = self.params_path.stat().st_mtime
        if self._last_params_mtime is None or mtime > self._last_params_mtime:
            self._last_params_mtime = mtime
            self._load_params()

    def _mark_hud_dirty(self) -> None:
        self._hud_metrics_valid = False

    def _reset_gradient_stats(self) -> None:
        self._grad_stats = {}

    def _export_csv(self) -> None:
        """Export particle data to CSV file."""
        from datetime import datetime
        output_dir = Path(__file__).resolve().parent / "output"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"particles_{timestamp}.csv"
        summary_path = output_dir / f"summary_{timestamp}.txt"
        
        try:
            stats = export_particles_csv(
                self.sim.particles,
                csv_path,
                include_velocity=True,
                include_acceleration=True,
                accel_mag=self.sim.accel_mag if hasattr(self.sim, 'accel_mag') else None,
            )
            export_summary(self.sim.particles, summary_path)
            print(f"[export] Saved {stats.particle_count} particles to {csv_path}")
            print(f"[export] Saved summary to {summary_path}")
        except Exception as e:
            print(f"[export] Error: {e}", file=sys.stderr)

    def _format_gradient_value(self, value: float) -> str:
        if not math.isfinite(value):
            return "n/a"
        value = float(value)
        if abs(value) >= 1000.0 or abs(value) < 0.01:
            return f"{value:.3g}"
        return f"{value:.2f}"

    def _gradient_range_text(self, mode: str) -> str | None:
        stats = self._grad_stats
        if not stats:
            return None
        mode = str(mode or "mix").strip().lower()
        key = None
        convert = None
        if mode == "speed":
            key = "grad_speed"
            convert = math.expm1
        elif mode == "force":
            key = "grad_force"
            convert = math.expm1
        elif mode == "density":
            key = "grad_density"
            convert = math.expm1
        elif mode == "proximity":
            key = "grad_prox"
        elif mode == "temperature":
            key = "grad_temp"
            convert = math.expm1
        else:
            key = "grad_mode_mix"
        if key not in stats:
            return None
        vmin, vmax = stats[key]
        if convert is not None:
            vmin = max(0.0, float(convert(max(0.0, vmin))))
            vmax = max(0.0, float(convert(max(0.0, vmax))))
        return f"min {self._format_gradient_value(vmin)}  max {self._format_gradient_value(vmax)}"

    def _menu_group_title(self, key: str) -> tuple[str | None, str | None]:
        group = MENU_GROUPS.get(key)
        if group is None:
            return None, None
        if isinstance(group, tuple):
            return group
        if isinstance(group, str) and "/" in group:
            main, sub = group.split("/", 1)
            return main.strip(), sub.strip()
        return str(group), None

    def _menu_disabled_reason(self, key: str) -> str | None:
        p = self.params
        pop_mode = str(getattr(p, "population_mode", "total")).strip().lower()
        if key in JANUS_DEP_KEYS and not bool(p.janus_enabled):
            return "Enable Janus forces first."
        if key == "force_tile_size" and str(p.force_backend) not in {"metal", "cpu_direct"}:
            return "Force backend must be metal or cpu_direct."
        if key == "split_screen" and bool(getattr(p, "multi_view", False)):
            return "Disable multi_view first."
        if key in BOUND_DEP_KEYS and not bool(p.bounds_enabled):
            return "Enable bounds first."
        if key == "bounds" and str(p.bound_mode) != "box":
            return "Bound mode must be box."
        if key in {
            "bound_sphere_radius",
            "bound_sphere_flatten_z",
            "bound_wire_visible",
            "bound_wire_opacity",
            "negative_on_boundary",
            "negative_static_on_boundary",
        } and str(p.bound_mode) != "sphere":
            return "Bound mode must be sphere."
        if key == "bound_wire_opacity" and not bool(p.bound_wire_visible):
            return "Enable bound_wire_visible first."
        if key == "negative_static_on_boundary" and not bool(p.negative_on_boundary):
            return "Enable negative_on_boundary first."
        if key == "sprite_scale" and not bool(getattr(p, "sprite_enabled", False)):
            return "Enable sprites first."
        if key == "grid_step" and not bool(getattr(p, "grid_enabled", False)):
            return "Enable grid first."
        if key in {
            "trails_length",
            "trails_stride",
            "trails_alpha",
            "trails_pos_only",
            "trails_blur",
            "trails_width",
        } and not bool(getattr(p, "trails_enabled", False)):
            return "Enable trails first."
        if key == "trails_width" and not bool(getattr(p, "trails_blur", False)):
            return "Enable trails_blur first."
        if key in {
            "merge_mode",
            "merge_radius",
            "merge_min_count",
            "merge_max_cells",
            "merge_temp_threshold",
            "merge_blob_scale",
        } and not bool(getattr(p, "merge_enabled", False)):
            return "Enable merge_enabled first."
        if key == "multi_view_count" and not bool(getattr(p, "multi_view", False)):
            return "Enable multi_view first."
        if pop_mode == "explicit" and key in {"particle_count", "negative_fraction"}:
            return "Using explicit M+/M- counts."
        if pop_mode == "total" and key in {"positive_count", "negative_count"}:
            return "Using total particle_count."
        return None

    def _menu_is_hidden(self, key: str) -> bool:
        return self._menu_disabled_reason(key) is not None

    def _reset_trails(self) -> None:
        maxlen = max(2, int(getattr(self.params, "trails_length", 16)))
        self._trail_history_pos = deque(maxlen=maxlen)
        self._trail_history_neg = deque(maxlen=maxlen)

    def _update_trails(self) -> None:
        if not bool(getattr(self.params, "trails_enabled", False)):
            return
        stride = max(1, int(getattr(self.params, "trails_stride", 4)))
        pos_only = bool(getattr(self.params, "trails_pos_only", False))
        pos_frame: list[float] = []
        neg_frame: list[float] = []
        for i, pt in enumerate(self.sim.particles):
            if i % stride != 0:
                continue
            if pt.s > 0:
                pos_frame.extend((pt.x, pt.y, pt.z))
            elif not pos_only:
                neg_frame.extend((pt.x, pt.y, pt.z))

        if self._trail_history_pos and len(pos_frame) != len(self._trail_history_pos[0]):
            self._trail_history_pos.clear()
        if self._trail_history_neg and len(neg_frame) != len(self._trail_history_neg[0]):
            self._trail_history_neg.clear()

        self._trail_history_pos.append(pos_frame)
        self._trail_history_neg.append(neg_frame)

    def _append_trails(
        self,
        history: deque[list[float]],
        base_color: tuple[int, int, int, int],
        xyz: array,
        rgba: array,
    ) -> None:
        if len(history) < 2:
            return
        base_alpha = int(round(float(base_color[3]) * float(getattr(self.params, "trails_alpha", 0.35))))
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

    def _get_trail_data(self) -> tuple[array, array]:
        xyz = self._trail_xyz
        rgba = self._trail_rgba
        xyz.clear()
        rgba.clear()
        if not bool(getattr(self.params, "trails_enabled", False)):
            return xyz, rgba
        self._append_trails(self._trail_history_pos, POS_BASE_COLOR, xyz, rgba)
        self._append_trails(self._trail_history_neg, NEG_BASE_COLOR, xyz, rgba)
        return xyz, rgba

    def _get_trail_data_split(self) -> tuple[array, array, array, array]:
        xyz_pos = self._trail_xyz_pos
        rgba_pos = self._trail_rgba_pos
        xyz_neg = self._trail_xyz_neg
        rgba_neg = self._trail_rgba_neg
        xyz_pos.clear()
        rgba_pos.clear()
        xyz_neg.clear()
        rgba_neg.clear()
        if not bool(getattr(self.params, "trails_enabled", False)):
            return xyz_pos, rgba_pos, xyz_neg, rgba_neg
        self._append_trails(self._trail_history_pos, POS_BASE_COLOR, xyz_pos, rgba_pos)
        self._append_trails(self._trail_history_neg, NEG_BASE_COLOR, xyz_neg, rgba_neg)
        return xyz_pos, rgba_pos, xyz_neg, rgba_neg

    def _get_sprite_info(self) -> dict[str, float | bool]:
        return {
            "enabled": bool(getattr(self.params, "sprite_enabled", False)),
            "scale": float(getattr(self.params, "sprite_scale", 2.5)),
        }

    def _get_grid_info(self) -> dict[str, float | bool]:
        if not bool(getattr(self.params, "grid_enabled", False)):
            return {"enabled": False}
        step = max(1.0, float(getattr(self.params, "grid_step", 50.0)))
        if bool(self.params.bounds_enabled):
            if str(self.params.bound_mode) == "box":
                size = float(self.params.bounds)
            else:
                size = float(self.params.bound_sphere_radius)
        else:
            size = max(float(self.params.galaxy_radius), float(self.params.galaxy_scale_length) * 3.0)
        size = max(step, size)
        return {"enabled": True, "size": size, "step": step}

    def _get_trail_info(self) -> dict[str, float | bool]:
        return {
            "blur": bool(getattr(self.params, "trails_blur", False)),
            "width": float(getattr(self.params, "trails_width", 1.5)),
        }

    def _get_camera_limits(self) -> tuple[float, float]:
        return float(self.params.camera_min_distance), float(self.params.camera_max_distance)

    def _get_blob_info(self) -> dict[str, float | bool]:
        has_blob = any(getattr(pt, "blob", False) for pt in self.sim.particles)
        return {
            "enabled": bool(getattr(self.params, "merge_enabled", False)) or has_blob,
            "scale": float(getattr(self.params, "merge_blob_scale", 2.5)),
        }

    def _metric_range(self, key: str, cur_min: float, cur_max: float) -> tuple[float, float]:
        if not (math.isfinite(cur_min) and math.isfinite(cur_max)):
            cur_min, cur_max = 0.0, 1.0
        if cur_min > cur_max:
            cur_min, cur_max = cur_max, cur_min
        prev = self._grad_stats.get(key)
        if prev is None:
            self._grad_stats[key] = (cur_min, cur_max)
            return cur_min, cur_max
        prev_min, prev_max = prev
        if not math.isfinite(prev_min):
            prev_min = cur_min
        if not math.isfinite(prev_max):
            prev_max = cur_max
        if cur_min < prev_min:
            prev_min = cur_min
        else:
            prev_min = (1.0 - GRAD_STAT_ALPHA) * prev_min + GRAD_STAT_ALPHA * cur_min
        if cur_max > prev_max:
            prev_max = cur_max
        else:
            prev_max = (1.0 - GRAD_STAT_ALPHA) * prev_max + GRAD_STAT_ALPHA * cur_max
        if (prev_max - prev_min) < 1e-9:
            prev_max = prev_min + 1e-9
        self._grad_stats[key] = (prev_min, prev_max)
        return prev_min, prev_max

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        if not math.isfinite(value):
            return 0.0
        span = max_val - min_val
        if span <= 1e-9:
            return 0.5
        t = (value - min_val) / span
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return t

    def _gradient_color(self, t: float, alpha: int = 220) -> tuple[int, int, int, int]:
        t = max(0.0, min(1.0, t))
        stops = GRADIENT_STOPS
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t <= t1:
                local = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                r = int(round(c0[0] + (c1[0] - c0[0]) * local))
                g = int(round(c0[1] + (c1[1] - c0[1]) * local))
                b = int(round(c0[2] + (c1[2] - c0[2]) * local))
                return r, g, b, alpha
        r, g, b = stops[-1][1]
        return r, g, b, alpha

    def _compute_mplus_gradient(self) -> list[tuple[int, int, int, int]] | None:
        particles = self.sim.particles
        n = len(particles)
        if n == 0:
            return None
        pos_indices = [i for i, pt in enumerate(particles) if pt.s > 0]
        if not pos_indices:
            return None

        cx, cy, cz, _cvx, _cvy, _cvz = self.sim.positive_center_velocity()

        r_max = 0.0
        for i in pos_indices:
            pt = particles[i]
            dx = pt.x - cx
            dy = pt.y - cy
            r = math.hypot(dx, dy)
            if r > r_max:
                r_max = r
        r_max = max(r_max, 1e-6)

        cell_div = float(getattr(self.params, "color_gradient_cell_div", GRAD_CELL_DIV))
        if not math.isfinite(cell_div):
            cell_div = GRAD_CELL_DIV
        cell_div = max(4.0, min(200.0, cell_div))
        cell_size = max(1.0, r_max / cell_div)
        inv_cell = 1.0 / cell_size
        cell_counts: dict[tuple[int, int, int], int] = {}
        cell_keys: list[tuple[int, int, int]] = []

        for i in pos_indices:
            pt = particles[i]
            key = (
                int(math.floor((pt.x - cx) * inv_cell)),
                int(math.floor((pt.y - cy) * inv_cell)),
                int(math.floor((pt.z - cz) * inv_cell)),
            )
            cell_keys.append(key)
            cell_counts[key] = cell_counts.get(key, 0) + 1

        accel_mag = self.sim.accel_mag
        speed_vals: list[float] = []
        force_vals: list[float] = []
        dens_vals: list[float] = []
        prox_vals: list[float] = []
        temp_vals: list[float] = []

        for idx, key in zip(pos_indices, cell_keys):
            pt = particles[idx]
            v2 = (pt.vx * pt.vx) + (pt.vy * pt.vy) + (pt.vz * pt.vz)
            speed = math.sqrt(v2)
            speed_vals.append(math.log1p(speed))
            temp_vals.append(math.log1p(max(0.0, float(pt.m) * v2)))

            force = 0.0
            if idx < len(accel_mag):
                force = accel_mag[idx]
                if not math.isfinite(force):
                    force = 0.0
            force_vals.append(math.log1p(abs(force)))

            dens_vals.append(math.log1p(cell_counts[key]))

            dx = pt.x - cx
            dy = pt.y - cy
            r = math.hypot(dx, dy)
            prox = 1.0 - (r / r_max)
            if prox < 0.0:
                prox = 0.0
            elif prox > 1.0:
                prox = 1.0
            prox_vals.append(prox)

        speed_min, speed_max = self._metric_range("grad_speed", min(speed_vals), max(speed_vals))
        force_min, force_max = self._metric_range("grad_force", min(force_vals), max(force_vals))
        dens_min, dens_max = self._metric_range("grad_density", min(dens_vals), max(dens_vals))
        temp_min, temp_max = self._metric_range("grad_temp", min(temp_vals), max(temp_vals))
        prox_min, prox_max = self._metric_range("grad_prox", min(prox_vals), max(prox_vals))

        w_speed = max(0.0, float(getattr(self.params, "color_gradient_speed_weight", 1.0)))
        w_force = max(0.0, float(getattr(self.params, "color_gradient_force_weight", 1.0)))
        w_dens = max(0.0, float(getattr(self.params, "color_gradient_density_weight", 1.0)))
        w_prox = max(0.0, float(getattr(self.params, "color_gradient_proximity_weight", 1.0)))
        w_sum = w_speed + w_force + w_dens + w_prox
        if w_sum <= 1e-9:
            w_speed = w_force = w_dens = w_prox = 1.0
            w_sum = 4.0

        mode = str(getattr(self.params, "color_gradient_mode", "mix")).strip().lower()
        raw_vals: list[float] = []
        for i in range(len(pos_indices)):
            t_speed = self._normalize(speed_vals[i], speed_min, speed_max)
            t_force = self._normalize(force_vals[i], force_min, force_max)
            t_dens = self._normalize(dens_vals[i], dens_min, dens_max)
            t_temp = self._normalize(temp_vals[i], temp_min, temp_max)
            t_prox = prox_vals[i]
            if mode == "speed":
                raw_vals.append(t_speed)
            elif mode == "force":
                raw_vals.append(t_force)
            elif mode == "density":
                raw_vals.append(t_dens)
            elif mode == "proximity":
                raw_vals.append(t_prox)
            elif mode == "temperature":
                raw_vals.append(t_temp)
            else:
                raw_vals.append((w_speed * t_speed + w_force * t_force + w_dens * t_dens + w_prox * t_prox) / w_sum)

        mix_key = f"grad_mode_{mode}"
        mix_min, mix_max = self._metric_range(mix_key, min(raw_vals), max(raw_vals))
        colors: list[tuple[int, int, int, int]] = [POS_BASE_COLOR] * n
        for idx, mix in zip(pos_indices, raw_vals):
            t = self._normalize(mix, mix_min, mix_max)
            colors[idx] = self._gradient_color(t)
        return colors

    def _force_hud_line(self) -> str:
        backend = str(getattr(self.sim, "last_force_backend", "off"))
        ms = getattr(self.sim, "last_force_ms", None)
        tile = int(getattr(self.params, "force_tile_size", 0))
        line = f"Forces: {backend}"
        if ms is not None:
            line += f"  {ms:.2f} ms"
        if backend in {"metal", "cpu_direct"} and tile > 0:
            line += f"  tile={tile}"
        return line

    def _maybe_refresh_hud_metrics(self) -> None:
        if not self._show_hud:
            return
        now = time.monotonic()
        if self._hud_metrics_valid and (now - self._hud_metrics_last) < HUD_METRICS_INTERVAL_S:
            return
        self._hud_curve = self.sim.rotation_curve(bins=8, r_max=float(self.params.galaxy_radius))
        self._hud_m2 = self.sim.m2_mode()
        self._hud_metrics_last = now
        self._hud_metrics_valid = True

    def _load_params(self) -> None:
        try:
            loaded = Sim3DParams.load(self.params_path)
        except Exception as e:
            self._params_error = f"Params reload failed: {e}"
            print(self._params_error, file=sys.stderr)
            return

        self._params_error = ""
        self.params = loaded
        self.sim.params = self.params
        self.sim.reset()
        self._sim_time = 0.0
        self._mark_hud_dirty()
        self._reset_gradient_stats()
        self._reset_trails()

    def _on_key(self, k: str) -> None:
        if k in ("f1", "tab"):
            self._menu_open = not self._menu_open
            if not self._menu_open:
                self._edit_active = False
            return

        if not self._menu_open:
            if k in ("plus", "minus"):
                self._nudge_time_scale(+1 if k == "plus" else -1)
                return
            if k in ("0", "1", "2", "3", "4", "5"):
                self._set_time_scale_preset(k)
                return
            if k == "f":
                self.params.time_scale = 10.0 if float(self.params.time_scale) < 9.5 else 1.0
                self.params.clamp()
                return
            if k == "v":
                self._show_hud = not self._show_hud
                if self._show_hud:
                    self._mark_hud_dirty()
                return
            if k == "x":
                self._display_mode = {"all": "pos", "pos": "neg", "neg": "all"}[self._display_mode]
                return
            if k == "b":
                self.params.bound_wire_visible = not bool(self.params.bound_wire_visible)
                self.params.clamp()
                return
            if k == "e":
                self._export_csv()
                return

        if self._menu_open:
            if self._edit_active:
                if k == "enter":
                    self._commit_edit()
                    return
                if k == "esc":
                    self._cancel_edit()
                    return
                if k == "backspace":
                    self._edit_buffer = self._edit_buffer[:-1]
                    return
                if k == "delete":
                    self._edit_buffer = ""
                    return
                return

            if k == "esc":
                self._menu_open = False
                return
            if k in ("pageup", "pagedown"):
                self._nudge_menu_step_scale(+1 if k == "pageup" else -1)
                return
            if k in ("up", "down"):
                items = self._menu_items()
                if not items:
                    return
                delta = -1 if k == "up" else 1
                self._menu_index = (self._menu_index + delta) % len(items)
                return
            if k in ("left", "right", "minus", "plus", "enter"):
                self._menu_apply(k)
                return

        if k == "space":
            self._running = not self._running
            return
        if k == "r":
            self.sim.reset()
            self._sim_time = 0.0
            self._mark_hud_dirty()
            self._reset_gradient_stats()
            self._reset_trails()
            return
        if k == "s":
            self.params.save(self.params_path)
            if self.params_path.exists():
                self._last_params_mtime = self.params_path.stat().st_mtime
            return
        if k == "l":
            self._load_params()
            return
        if k == "esc":
            raise SystemExit(0)

    def _on_text_input(self, text: str) -> None:
        if not self._menu_open:
            return
        if not text:
            return
        # Support french keyboards: accept "," as decimal separator.
        if text == ",":
            text = "."
        # Accept a minimal set of characters for float/int parsing.
        if text not in "0123456789+-eE.":
            return

        # If the user types while the menu is open, start editing the selected numeric item.
        if not self._edit_active:
            items = self._menu_items()
            if not items:
                return
            self._menu_index = max(0, min(self._menu_index, len(items) - 1))
            key, label, kind, _step = items[self._menu_index]
            if kind not in {"int", "float"}:
                return
            if self._menu_disabled_reason(key):
                return
            self._begin_edit(key=key, label=label, kind=kind, prefill=False)

        if len(self._edit_buffer) >= 48:
            return
        self._edit_buffer += text
        self._edit_error = ""

    def _step(self, dt: float) -> None:
        self._maybe_autoreload()
        if not self._running:
            return
        dt_base = max(0.0, min(0.03, float(dt)))
        dt_scaled = dt_base * float(self.params.time_scale)
        dt_scaled = min(dt_scaled, 0.25)
        if dt_scaled <= 0.0:
            return
        self.sim.step(dt_scaled)
        self._sim_time += dt_scaled
        self._mark_hud_dirty()
        self._update_trails()

    def _menu_items(self) -> list[tuple[str, str, str, float]]:
        # (key, label, kind, step)
        items = [
            ("init_mode", "Init", "choice", 1.0),
            ("time_scale", "Speed (x)", "float", 0.25),
            ("janus_enabled", "Janus enabled", "bool", 1.0),
            ("janus_g", "G", "float", 100.0),
            ("force_backend", "Force backend", "choice", 1.0),
            ("force_tile_size", "Force tile", "int", 32.0),
            ("force_debug", "Force debug", "bool", 1.0),
            ("softening", "Softening eps", "float", 1.0),
            ("theta", "Theta (BH)", "float", 0.05),
            ("split_screen", "Split screen", "bool", 1.0),
            ("multi_view", "Multi view", "bool", 1.0),
            ("multi_view_count", "View count", "int", 1.0),
            ("color_gradient", "Color gradient", "bool", 1.0),
            ("color_gradient_mode", "Gradient mode", "choice", 1.0),
            ("sprite_enabled", "Sprites", "bool", 1.0),
            ("sprite_scale", "Sprite scale", "float", 0.1),
            ("grid_enabled", "Grid", "bool", 1.0),
            ("grid_step", "Grid step", "float", 5.0),
            ("camera_min_distance", "Zoom min", "float", 0.1),
            ("camera_max_distance", "Zoom max", "float", 10.0),
            ("trails_enabled", "Trails", "bool", 1.0),
            ("trails_length", "Trail length", "int", 1.0),
            ("trails_stride", "Trail stride", "int", 1.0),
            ("trails_alpha", "Trail alpha", "float", 0.05),
            ("trails_pos_only", "Trails M+ only", "bool", 1.0),
            ("trails_blur", "Trail blur", "bool", 1.0),
            ("trails_width", "Trail width", "float", 0.1),
            ("bounds_enabled", "Bounds enabled", "bool", 1.0),
            ("bound_mode", "Bound mode", "choice", 1.0),
            ("bounds", "Box half", "float", 20.0),
            ("bound_sphere_radius", "Sphere radius", "float", 20.0),
            ("bound_sphere_flatten_z", "Sphere flatZ", "float", 0.05),
            ("bound_wire_visible", "Bound wire", "bool", 1.0),
            ("bound_wire_opacity", "Bound alpha", "float", 0.05),
            ("negative_on_boundary", "Neg on boundary", "bool", 1.0),
            ("negative_static_on_boundary", "Neg boundary static", "bool", 1.0),
            ("population_mode", "Count mode", "choice", 1.0),
            ("particle_count", "Particles total", "int", 200.0),
            ("negative_fraction", "Neg fraction", "float", 0.02),
            ("positive_count", "M+ count", "int", 200.0),
            ("negative_count", "M- count", "int", 200.0),
            ("merge_enabled", "Merge dense", "bool", 1.0),
            ("merge_mode", "Merge mode", "choice", 1.0),
            ("merge_radius", "Merge radius", "float", 1.0),
            ("merge_min_count", "Merge count", "int", 1.0),
            ("merge_max_cells", "Merge limit", "int", 1.0),
            ("merge_temp_threshold", "Merge temp", "float", 1.0),
            ("merge_blob_scale", "Blob size", "float", 0.1),
            ("mass_positive", "M+ mass", "float", 0.25),
            ("mass_negative", "M- mass", "float", 1.0),
            ("void_radius", "Void radius", "float", 10.0),
            ("galaxy_radius", "Galaxy radius", "float", 10.0),
            ("galaxy_scale_length", "Galaxy scale", "float", 5.0),
            ("galaxy_thickness", "Galaxy thick", "float", 1.0),
            ("galaxy_vmax", "Galaxy vmax", "float", 10.0),
            ("galaxy_turnover", "Galaxy turn", "float", 5.0),
            ("galaxy_sigma_v", "Galaxy sigma", "float", 1.0),
            ("negative_clump_count", "M- clumps", "int", 1.0),
            ("negative_clump_sigma", "M- clump sig", "float", 1.0),
            ("negative_sigma_v", "M- sigma v", "float", 1.0),
            ("negative_vphi_scale", "M- vphi", "float", 0.1),
            ("damping", "Damping", "float", 0.0005),
            ("bounce", "Bounce", "float", 0.05),
            ("max_speed", "Max speed", "float", 50.0),
            ("point_size", "Point size", "float", 0.5),
        ]
        return [item for item in items if not self._menu_is_hidden(item[0])]

    def _nudge_menu_step_scale(self, direction: int) -> None:
        # direction: +1 -> larger step, -1 -> smaller step
        factor = 10.0 if direction > 0 else 0.1
        self._menu_step_scale = max(1e-6, min(1e6, float(self._menu_step_scale) * factor))

    def _begin_edit(self, *, key: str, label: str, kind: str, prefill: bool = True) -> None:
        if prefill:
            value = getattr(self.params, key)
            if kind == "int":
                self._edit_buffer = str(int(value))
            else:
                self._edit_buffer = f"{float(value):.8g}"
        else:
            self._edit_buffer = ""
        self._edit_active = True
        self._edit_key = key
        self._edit_kind = kind
        self._edit_label = label
        self._edit_error = ""

    def _cancel_edit(self) -> None:
        self._edit_active = False
        self._edit_error = ""

    def _commit_edit(self) -> None:
        key = self._edit_key
        kind = self._edit_kind
        raw = self._edit_buffer.strip()
        if not key or not kind:
            self._cancel_edit()
            return
        reason = self._menu_disabled_reason(key)
        if reason:
            self._edit_error = f"Disabled: {reason}"
            return
        try:
            if kind == "int":
                parsed = int(round(float(raw)))
            else:
                parsed = float(raw)
        except Exception:
            self._edit_error = "Invalid value"
            return

        setattr(self.params, key, parsed)
        self.params.clamp()
        self.sim.params = self.params

        if key in RESET_KEYS:
            self.sim.reset()
            self._sim_time = 0.0
            self._mark_hud_dirty()
            self._reset_gradient_stats()
            self._reset_trails()
        elif key in ("mass_positive", "mass_negative"):
            self.sim.update_masses()
            self._mark_hud_dirty()
        elif key in BOUNDS_KEYS:
            self.sim.enforce_bounds()
            self._mark_hud_dirty()
        else:
            self._mark_hud_dirty()
            if key in GRADIENT_KEYS:
                self._reset_gradient_stats()
            if key in TRAIL_KEYS:
                self._reset_trails()
        self._edit_active = False
        self._edit_error = ""

    def _nudge_time_scale(self, direction: int) -> None:
        factor = 1.25 ** int(direction)
        self.params.time_scale = float(self.params.time_scale) * factor
        self.params.clamp()

    def _set_time_scale_preset(self, k: str) -> None:
        presets = {
            "0": 1.0,
            "1": 1.0,
            "2": 2.0,
            "3": 5.0,
            "4": 10.0,
            "5": 20.0,
        }
        self.params.time_scale = float(presets.get(k, 1.0))
        self.params.clamp()

    def _menu_apply(self, k: str) -> None:
        items = self._menu_items()
        if not items:
            return
        self._menu_index = max(0, min(self._menu_index, len(items) - 1))
        key, _label, kind, step = items[self._menu_index]
        if self._menu_disabled_reason(key):
            return

        if kind == "choice":
            if k not in ("enter", "left", "right", "plus", "minus"):
                return
            if key == "init_mode":
                choices = ["janus_galaxy", "random"]
            elif key == "force_backend":
                choices = ["cpu", "metal", "cpu_direct"]
            elif key == "bound_mode":
                choices = ["box", "sphere"]
            elif key == "population_mode":
                choices = ["total", "explicit"]
            elif key == "color_gradient_mode":
                choices = ["mix", "speed", "force", "density", "proximity", "temperature"]
            elif key == "merge_mode":
                choices = ["all", "mplus"]
            else:
                return
            cur = str(getattr(self.params, key))
            try:
                i = choices.index(cur)
            except ValueError:
                i = 0
            i = (i + 1) % len(choices)
            setattr(self.params, key, choices[i])
            self.params.clamp()
            self.sim.params = self.params
            if key in RESET_KEYS:
                self.sim.reset()
                self._sim_time = 0.0
                self._reset_gradient_stats()
                self._reset_trails()
            elif key in BOUNDS_KEYS:
                self.sim.enforce_bounds()
            self._mark_hud_dirty()
            if key in GRADIENT_KEYS:
                self._reset_gradient_stats()
            if key in TRAIL_KEYS:
                self._reset_trails()
            return

        if kind == "bool":
            if k in ("enter", "left", "right", "plus", "minus"):
                setattr(self.params, key, not bool(getattr(self.params, key)))
                self.params.clamp()
                self.sim.params = self.params
                if key in RESET_KEYS:
                    self.sim.reset()
                    self._sim_time = 0.0
                    self._reset_gradient_stats()
                    self._reset_trails()
                elif key in BOUNDS_KEYS:
                    self.sim.enforce_bounds()
                self._mark_hud_dirty()
                if key in GRADIENT_KEYS:
                    self._reset_gradient_stats()
                if key in TRAIL_KEYS:
                    self._reset_trails()
            return

        if k == "enter":
            self._begin_edit(key=key, label=_label, kind=kind, prefill=True)
            return
        delta = 0.0
        if k in ("right", "plus"):
            delta = +1.0
        elif k in ("left", "minus"):
            delta = -1.0
        else:
            return

        eff_step = float(step) * float(self._menu_step_scale)
        cur = getattr(self.params, key)
        if kind == "int":
            step_i = max(1, int(round(eff_step)))
            new_value = int(round(float(cur) + delta * float(step_i)))
        else:
            new_value = float(cur) + delta * eff_step
        setattr(self.params, key, new_value)
        self.params.clamp()
        self.sim.params = self.params

        if key in RESET_KEYS:
            self.sim.reset()
            self._sim_time = 0.0
            self._mark_hud_dirty()
            self._reset_gradient_stats()
            self._reset_trails()
        elif key in BOUNDS_KEYS:
            self.sim.enforce_bounds()
            self._mark_hud_dirty()
        else:
            self._mark_hud_dirty()
            if key in GRADIENT_KEYS:
                self._reset_gradient_stats()
            if key in TRAIL_KEYS:
                self._reset_trails()

    def _get_overlay_text(self) -> str:
        if not self._menu_open and not self._show_hud:
            return ""

        if self._show_hud:
            self._maybe_refresh_hud_metrics()

        lines: list[str] = []
        n_pos, n_neg = self.sim.counts()
        mpos = float(self.params.mass_positive)
        mneg = float(self.params.mass_negative)
        ratio = (float(n_neg) * mneg) / max(1e-9, float(n_pos) * mpos)

        if self._show_hud:
            bound = str(self.params.bound_mode)
            if bound == "sphere":
                btxt = (
                    f"sphere R={float(self.params.bound_sphere_radius):.0f} "
                    f"flatZ={float(self.params.bound_sphere_flatten_z):.2f}"
                    f" wire={'on' if self.params.bound_wire_visible and self.params.bounds_enabled else 'off'}"
                )
            else:
                btxt = f"box half={float(self.params.bounds):.0f}"
            if not self.params.bounds_enabled:
                btxt += " [OFF]"
            lines.extend(
                [
                    "HUD (V hide/show, F1/TAB menu)",
                    f"t={self._sim_time:7.1f}s  x{float(self.params.time_scale):.2f}  filter={self._display_mode}",
                    f"M+={n_pos}  M-={n_neg}  m+={mpos:.3g}  m-={mneg:.3g}  (~{ratio:4.0f}x)",
                    self._force_hud_line(),
                    f"{btxt}",
                    f"void={float(self.params.void_radius):.0f}  galaxy_r={float(self.params.galaxy_radius):.0f}  m2={self._hud_m2:.3f}",
                    "",
                ]
            )

            curve = self._hud_curve
            lines.append("Rotation curve (M+):  r   mean|vphi|   n")
            for r_mid, v_mean, n in curve:
                if n == 0:
                    continue
                lines.append(f"{r_mid:7.0f}  {v_mean:10.1f}  {n:4d}")
            lines.append("")
            lines.append("Keys: +/- speed | 1..5 presets | X filter | G focus | H fit | T follow | Shift+drag pan")

        if self._menu_open:
            items = self._menu_items()
            if not items:
                return "\n".join(lines)
            self._menu_index = max(0, min(self._menu_index, len(items) - 1))

            _sel_key, _sel_label, _sel_kind, sel_step = items[self._menu_index]
            eff_step = float(sel_step) * float(self._menu_step_scale)
            lines.extend(
                [
                    "",
                    "Menu",
                    "UP/DOWN select | LEFT/RIGHT or +/- change | TYPE/ENTER edit | PGUP/PGDN step",
                    f"Step: {eff_step:g} (x{self._menu_step_scale:g})",
                ]
            )
            hint = PARAM_HINTS.get(_sel_key)
            if hint:
                lines.append(f"Hint: {hint}")
            disabled_reason = self._menu_disabled_reason(_sel_key)
            if disabled_reason:
                lines.append(f"Disabled: {disabled_reason}")
            if _sel_key in RESET_KEYS:
                lines.append("Effect: reset simulation")
            elif _sel_key in BOUNDS_KEYS:
                lines.append("Effect: enforce bounds")
            lines.append("")
            if self._edit_active:
                cursor = "_" if (len(self._edit_buffer) < 48) else ""
                lines.append(f"Edit {self._edit_label}: {self._edit_buffer}{cursor}")
                lines.append("ENTER apply | ESC cancel | BACKSPACE delete")
                if self._edit_error:
                    lines.append(f"Error: {self._edit_error}")
                lines.append("")
            prev_group = None
            prev_sub = None
            for i, (key, label, kind, _step) in enumerate(items):
                group_title, sub_title = self._menu_group_title(key)
                if group_title and group_title != prev_group:
                    if lines and lines[-1] != "":
                        lines.append("")
                    lines.append(f"[{group_title}]")
                    prev_group = group_title
                    prev_sub = None
                if sub_title and sub_title != prev_sub:
                    lines.append(f" {sub_title}")
                    prev_sub = sub_title
                prefix = ">" if i == self._menu_index else " "
                value = getattr(self.params, key)
                if key == "merge_mode":
                    text_val = "M+" if str(value).lower() in {"mplus", "m+"} else str(value)
                elif kind == "float":
                    text_val = f"{float(value):.5g}"
                elif kind == "int":
                    text_val = f"{int(value)}"
                else:
                    text_val = str(value)
                disabled = self._menu_disabled_reason(key) is not None
                if disabled:
                    text_val = f"{text_val} (disabled)"
                lines.append(f"{prefix} {label:<16}: {text_val}")
            warnings = self.params.validate()
            if warnings:
                lines.append("")
                lines.append("Warnings")
                for msg in warnings[:3]:
                    lines.append(f"! {msg}")
                if len(warnings) > 3:
                    lines.append(f"... {len(warnings) - 3} more")

        if self._params_error:
            lines.append("")
            lines.append(f"[params] {self._params_error}")

        return "\n".join(lines).rstrip()

    def _get_positions_and_colors(self) -> tuple[array, array]:
        xyz = self._xyz_buffer
        rgba = self._rgba_buffer
        xyz.clear()
        rgba.clear()

        mode = self._display_mode
        grad_colors = None
        if bool(self.params.color_gradient) and mode != "neg":
            grad_colors = self._compute_mplus_gradient()

        for i, pt in enumerate(self.sim.particles):
            if getattr(pt, "blob", False):
                continue
            if mode == "pos" and pt.s < 0:
                continue
            if mode == "neg" and pt.s > 0:
                continue
            xyz.extend((pt.x, pt.y, pt.z))
            if pt.s < 0:
                rgba.extend(NEG_BASE_COLOR)
            else:
                if grad_colors is not None:
                    rgba.extend(grad_colors[i])
                else:
                    rgba.extend(POS_BASE_COLOR)
        return xyz, rgba

    def _get_positions_and_colors_split(self) -> tuple[array, array, array, array]:
        xyz_pos = self._xyz_buffer_pos
        rgba_pos = self._rgba_buffer_pos
        xyz_neg = self._xyz_buffer_neg
        rgba_neg = self._rgba_buffer_neg
        xyz_pos.clear()
        rgba_pos.clear()
        xyz_neg.clear()
        rgba_neg.clear()

        mode = self._display_mode
        grad_colors = None
        if bool(self.params.color_gradient) and mode != "neg":
            grad_colors = self._compute_mplus_gradient()

        for i, pt in enumerate(self.sim.particles):
            if getattr(pt, "blob", False):
                continue
            if pt.s >= 0:
                if mode == "neg":
                    continue
                xyz_pos.extend((pt.x, pt.y, pt.z))
                if grad_colors is not None:
                    rgba_pos.extend(grad_colors[i])
                else:
                    rgba_pos.extend(POS_BASE_COLOR)
            else:
                if mode == "pos":
                    continue
                xyz_neg.extend((pt.x, pt.y, pt.z))
                rgba_neg.extend(NEG_BASE_COLOR)
        return xyz_pos, rgba_pos, xyz_neg, rgba_neg

    def _get_blob_positions_and_colors(self) -> tuple[array, array]:
        xyz = self._xyz_blob
        rgba = self._rgba_blob
        xyz.clear()
        rgba.clear()

        mode = self._display_mode
        grad_colors = None
        if bool(self.params.color_gradient) and mode != "neg":
            grad_colors = self._compute_mplus_gradient()

        for i, pt in enumerate(self.sim.particles):
            if not getattr(pt, "blob", False):
                continue
            if mode == "pos" and pt.s < 0:
                continue
            if mode == "neg" and pt.s > 0:
                continue
            xyz.extend((pt.x, pt.y, pt.z))
            if pt.s < 0:
                rgba.extend(NEG_BASE_COLOR)
            else:
                if grad_colors is not None:
                    rgba.extend(grad_colors[i])
                else:
                    rgba.extend(POS_BASE_COLOR)
        return xyz, rgba

    def _get_blob_positions_and_colors_split(self) -> tuple[array, array, array, array]:
        xyz_pos = self._xyz_blob_pos
        rgba_pos = self._rgba_blob_pos
        xyz_neg = self._xyz_blob_neg
        rgba_neg = self._rgba_blob_neg
        xyz_pos.clear()
        rgba_pos.clear()
        xyz_neg.clear()
        rgba_neg.clear()

        mode = self._display_mode
        grad_colors = None
        if bool(self.params.color_gradient) and mode != "neg":
            grad_colors = self._compute_mplus_gradient()

        for i, pt in enumerate(self.sim.particles):
            if not getattr(pt, "blob", False):
                continue
            if pt.s >= 0:
                if mode == "neg":
                    continue
                xyz_pos.extend((pt.x, pt.y, pt.z))
                if grad_colors is not None:
                    rgba_pos.extend(grad_colors[i])
                else:
                    rgba_pos.extend(POS_BASE_COLOR)
            else:
                if mode == "pos":
                    continue
                xyz_neg.extend((pt.x, pt.y, pt.z))
                rgba_neg.extend(NEG_BASE_COLOR)
        return xyz_pos, rgba_pos, xyz_neg, rgba_neg
