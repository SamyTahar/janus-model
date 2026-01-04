from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Sim3DParams:
    width: int = 1100
    height: int = 720
    background: tuple[int, int, int] = (11, 16, 32)

    init_mode: str = "janus_galaxy"  # janus_galaxy | random

    population_mode: str = "total"  # total | explicit
    particle_count: int = 5000
    positive_count: int = 0  # used when population_mode="explicit"
    negative_count: int = 0
    initial_speed: float = 120.0
    max_speed: float = 900.0
    damping: float = 0.999
    bounce: float = 0.85

    janus_enabled: bool = True
    janus_g: float = 1200.0
    force_backend: str = "cpu"  # cpu | metal | cpu_direct
    force_tile_size: int = 256
    force_debug: bool = False
    softening: float = 20.0
    negative_fraction: float = 0.67
    merge_enabled: bool = False
    merge_radius: float = 12.0
    merge_min_count: int = 10
    merge_mode: str = "all"
    merge_max_cells: int = 0
    merge_temp_threshold: float = 0.0
    merge_blob_scale: float = 2.5
    mass_positive: float = 1.0
    mass_negative: float = 32.0  # magnitude (positive number)

    void_radius: float = 240.0
    galaxy_radius: float = 200.0
    galaxy_scale_length: float = 60.0
    galaxy_thickness: float = 15.0
    galaxy_vmax: float = 240.0
    galaxy_turnover: float = 55.0
    galaxy_sigma_v: float = 12.0

    negative_clump_count: int = 24
    negative_clump_sigma: float = 55.0
    negative_sigma_v: float = 22.0
    negative_vphi_scale: float = 0.0
    negative_on_boundary: bool = False
    negative_static_on_boundary: bool = False

    bounds_enabled: bool = True
    split_screen: bool = False
    multi_view: bool = False
    multi_view_count: int = 3
    bound_mode: str = "box"  # box | sphere
    bounds: float = 420.0  # box half-size: [-bounds, +bounds]
    bound_sphere_radius: float = 420.0
    bound_sphere_flatten_z: float = 1.0  # 1.0 = sphere, <1.0 = flattened along Z
    theta: float = 0.7  # Barnes-Hut opening angle
    bound_wire_visible: bool = False
    bound_wire_opacity: float = 0.2

    mac_compat: bool = True  # options pyglet pour macOS (contexte sans MSAA, shadow window off)

    point_size: float = 3.0
    camera_min_distance: float = 0.2
    camera_max_distance: float = 15000.0
    color_gradient: bool = False
    color_gradient_mode: str = "mix"
    color_gradient_speed_weight: float = 1.0
    color_gradient_force_weight: float = 1.0
    color_gradient_density_weight: float = 1.0
    color_gradient_proximity_weight: float = 1.0
    color_gradient_cell_div: float = 20.0
    sprite_enabled: bool = False
    sprite_scale: float = 2.5
    grid_enabled: bool = False
    grid_step: float = 50.0
    trails_enabled: bool = False
    trails_length: int = 16
    trails_stride: int = 4
    trails_alpha: float = 0.35
    trails_pos_only: bool = False
    trails_blur: bool = False
    trails_width: float = 1.5
    time_scale: float = 1.0
    target_fps: int = 60
    seed: int = 1

    def clamp(self) -> "Sim3DParams":
        self.width = max(320, int(self.width))
        self.height = max(240, int(self.height))
        self.init_mode = str(self.init_mode or "janus_galaxy").strip().lower()
        if self.init_mode not in {"janus_galaxy", "random"}:
            self.init_mode = "janus_galaxy"
        self.population_mode = str(self.population_mode or "total").strip().lower()
        if self.population_mode not in {"total", "explicit"}:
            self.population_mode = "total"
        self.particle_count = max(1, int(self.particle_count))
        self.positive_count = max(0, int(self.positive_count))
        self.negative_count = max(0, int(self.negative_count))
        self.initial_speed = max(0.0, float(self.initial_speed))
        self.max_speed = max(0.0, float(self.max_speed))
        self.damping = min(1.0, max(0.0, float(self.damping)))
        self.bounce = min(1.0, max(0.0, float(self.bounce)))
        self.janus_enabled = bool(self.janus_enabled)
        self.janus_g = max(0.0, float(self.janus_g))
        self.force_backend = str(self.force_backend or "cpu").strip().lower()
        if self.force_backend not in {"cpu", "metal", "cpu_direct"}:
            self.force_backend = "cpu"
        self.force_tile_size = max(16, min(2048, int(self.force_tile_size)))
        self.force_debug = bool(self.force_debug)
        self.softening = max(0.0, float(self.softening))
        self.negative_fraction = min(0.99, max(0.01, float(self.negative_fraction)))
        self.merge_enabled = bool(self.merge_enabled)
        self.merge_radius = max(1.0, float(self.merge_radius))
        self.merge_min_count = max(2, int(self.merge_min_count))
        self.merge_mode = str(self.merge_mode or "all").strip().lower()
        if self.merge_mode in {"m+", "mplus"}:
            self.merge_mode = "mplus"
        if self.merge_mode not in {"all", "mplus"}:
            self.merge_mode = "all"
        self.merge_max_cells = max(0, int(self.merge_max_cells))
        self.merge_temp_threshold = max(0.0, float(self.merge_temp_threshold))
        self.merge_blob_scale = min(10.0, max(1.0, float(self.merge_blob_scale)))
        self.mass_positive = max(1e-9, float(self.mass_positive))
        self.mass_negative = max(1e-9, float(self.mass_negative))
        self.galaxy_thickness = max(0.0, float(self.galaxy_thickness))
        self.galaxy_vmax = max(0.0, float(self.galaxy_vmax))
        self.galaxy_turnover = max(1.0, float(self.galaxy_turnover))
        self.galaxy_sigma_v = max(0.0, float(self.galaxy_sigma_v))
        self.negative_clump_count = max(1, int(self.negative_clump_count))
        self.negative_clump_sigma = max(0.0, float(self.negative_clump_sigma))
        self.negative_sigma_v = max(0.0, float(self.negative_sigma_v))
        self.negative_vphi_scale = max(-5.0, min(5.0, float(self.negative_vphi_scale)))
        self.negative_on_boundary = bool(self.negative_on_boundary)
        self.negative_static_on_boundary = bool(self.negative_static_on_boundary)
        self.bounds_enabled = bool(self.bounds_enabled)
        self.split_screen = bool(self.split_screen)
        self.multi_view = bool(self.multi_view)
        self.multi_view_count = max(1, min(3, int(self.multi_view_count)))
        self.bound_mode = str(self.bound_mode or "box").strip().lower()
        if self.bound_mode not in {"box", "sphere"}:
            self.bound_mode = "box"
        self.bounds = max(10.0, float(self.bounds))
        self.bound_sphere_radius = max(10.0, float(self.bound_sphere_radius))
        self.bound_sphere_flatten_z = min(5.0, max(0.05, float(self.bound_sphere_flatten_z)))
        self.bound_wire_visible = bool(self.bound_wire_visible)
        self.bound_wire_opacity = min(1.0, max(0.0, float(self.bound_wire_opacity)))

        domain = self.bounds if self.bound_mode == "box" else self.bound_sphere_radius
        self.void_radius = min(domain * 0.95, max(0.0, float(self.void_radius)))
        self.galaxy_radius = min(self.void_radius * 0.95, max(1.0, float(self.galaxy_radius)))
        self.galaxy_scale_length = min(self.galaxy_radius, max(1.0, float(self.galaxy_scale_length)))

        self.theta = min(2.0, max(0.1, float(self.theta)))
        self.mac_compat = bool(self.mac_compat)
        self.point_size = max(1.0, float(self.point_size))
        self.camera_min_distance = max(0.01, float(self.camera_min_distance))
        self.camera_max_distance = max(self.camera_min_distance, float(self.camera_max_distance))
        self.color_gradient = bool(self.color_gradient)
        self.color_gradient_mode = str(self.color_gradient_mode or "mix").strip().lower()
        if self.color_gradient_mode not in {"mix", "speed", "force", "density", "proximity", "temperature"}:
            self.color_gradient_mode = "mix"
        self.color_gradient_speed_weight = max(0.0, float(self.color_gradient_speed_weight))
        self.color_gradient_force_weight = max(0.0, float(self.color_gradient_force_weight))
        self.color_gradient_density_weight = max(0.0, float(self.color_gradient_density_weight))
        self.color_gradient_proximity_weight = max(0.0, float(self.color_gradient_proximity_weight))
        self.color_gradient_cell_div = min(200.0, max(4.0, float(self.color_gradient_cell_div)))
        self.sprite_enabled = bool(self.sprite_enabled)
        self.sprite_scale = min(8.0, max(0.5, float(self.sprite_scale)))
        self.grid_enabled = bool(self.grid_enabled)
        self.grid_step = max(1.0, float(self.grid_step))
        self.trails_enabled = bool(self.trails_enabled)
        self.trails_length = max(2, min(128, int(self.trails_length)))
        self.trails_stride = max(1, min(64, int(self.trails_stride)))
        self.trails_alpha = min(1.0, max(0.0, float(self.trails_alpha)))
        self.trails_pos_only = bool(self.trails_pos_only)
        self.trails_blur = bool(self.trails_blur)
        self.trails_width = min(6.0, max(0.5, float(self.trails_width)))
        self.time_scale = min(100.0, max(0.05, float(self.time_scale)))
        self.target_fps = max(10, int(self.target_fps))
        self.seed = int(self.seed)
        return self

    def validate(self) -> list[str]:
        warnings: list[str] = []

        if self.bound_mode != "sphere":
            if self.bound_wire_visible:
                warnings.append("bound_wire_visible only applies in sphere mode.")
            if self.negative_on_boundary or self.negative_static_on_boundary:
                warnings.append("negative_on_boundary/negative_static_on_boundary only apply in sphere mode.")
        else:
            if not self.bounds_enabled and self.bound_wire_visible:
                warnings.append("bound_wire_visible needs bounds_enabled.")
            if self.negative_static_on_boundary and not self.negative_on_boundary:
                warnings.append("negative_static_on_boundary needs negative_on_boundary.")

        if self.population_mode == "explicit":
            if self.positive_count > 0 and self.negative_count > 0:
                pass
            else:
                warnings.append("population_mode=explicit needs positive_count and negative_count > 0.")
        elif self.population_mode == "total":
            if self.positive_count > 0 or self.negative_count > 0:
                warnings.append("positive_count/negative_count ignored when population_mode=total.")

        if not self.bounds_enabled and float(self.bounce) > 0.0:
            warnings.append("bounce has no effect when bounds are disabled.")
        if not self.janus_enabled and float(self.janus_g) > 0.0:
            warnings.append("janus_g ignored while janus_enabled is false.")
        if self.init_mode == "random":
            warnings.append("init_mode=random: galaxy params are ignored until reset.")
            if abs(float(self.negative_vphi_scale)) > 1e-6:
                warnings.append("negative_vphi_scale ignored when init_mode=random.")

        return warnings

    @classmethod
    def load(cls, path: str | Path) -> "Sim3DParams":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Le fichier de parametres doit contenir un objet JSON.")
        # Backward compatibility: older configs used `mass_ratio = |M-|/M+`.
        if "mass_ratio" in data and "mass_negative" not in data:
            try:
                mpos = float(data.get("mass_positive", cls.mass_positive))
                data["mass_negative"] = float(data["mass_ratio"]) * mpos
            except Exception:
                pass
        if "population_mode" not in data:
            try:
                pos = int(data.get("positive_count", cls.positive_count))
                neg = int(data.get("negative_count", cls.negative_count))
            except Exception:
                pos = 0
                neg = 0
            data["population_mode"] = "explicit" if (pos > 0 and neg > 0) else "total"
        filtered: dict[str, Any] = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**filtered).clamp()

    def save(self, path: str | Path) -> None:
        data = asdict(self)
        if str(data.get("merge_mode")).lower() == "mplus":
            data["merge_mode"] = "M+"
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
