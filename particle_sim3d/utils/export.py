"""
Export utilities for particle simulation data.

This module provides functions to export simulation state to various formats:
- CSV: Positions, velocities, and particle properties
- Stats: Summary statistics

Usage:
    >>> from particle_sim3d.export import export_particles_csv
    >>> export_particles_csv(particles, "output.csv")
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particle_sim3d.core.sim import Particle3D


@dataclass
class ExportStats:
    """Statistics from an export operation."""
    file_path: Path
    particle_count: int
    positive_count: int
    negative_count: int
    timestamp: str


def export_particles_csv(
    particles: list["Particle3D"],
    output_path: str | Path,
    *,
    include_velocity: bool = True,
    include_acceleration: bool = False,
    accel_mag: list[float] | None = None,
    frame: int | None = None,
) -> ExportStats:
    """
    Export particle data to a CSV file.
    
    Args:
        particles: List of Particle3D objects
        output_path: Path to output CSV file
        include_velocity: Include velocity columns (vx, vy, vz)
        include_acceleration: Include acceleration magnitude column
        accel_mag: Acceleration magnitudes (required if include_acceleration=True)
        frame: Optional frame number to include in output
    
    Returns:
        ExportStats with export details
    
    Example:
        >>> stats = export_particles_csv(sim.particles, "particles.csv")
        >>> print(f"Exported {stats.particle_count} particles")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build header
    header = ["index", "x", "y", "z", "sign", "mass"]
    if include_velocity:
        header.extend(["vx", "vy", "vz"])
    if include_acceleration and accel_mag is not None:
        header.append("accel_mag")
    if frame is not None:
        header.insert(0, "frame")
    
    # Count particles
    n_pos = sum(1 for p in particles if p.s > 0)
    n_neg = len(particles) - n_pos
    
    # Write CSV
    timestamp = datetime.now().isoformat()
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header comment
        writer.writerow([f"# Janus 3D Export - {timestamp}"])
        writer.writerow([f"# Particles: {len(particles)} (M+: {n_pos}, M-: {n_neg})"])
        writer.writerow(header)
        
        for i, p in enumerate(particles):
            row = []
            if frame is not None:
                row.append(frame)
            row.extend([
                i,
                f"{p.x:.6f}",
                f"{p.y:.6f}",
                f"{p.z:.6f}",
                1 if p.s > 0 else -1,
                f"{p.m:.6f}",
            ])
            if include_velocity:
                row.extend([
                    f"{p.vx:.6f}",
                    f"{p.vy:.6f}",
                    f"{p.vz:.6f}",
                ])
            if include_acceleration and accel_mag is not None and i < len(accel_mag):
                row.append(f"{accel_mag[i]:.6f}")
            writer.writerow(row)
    
    return ExportStats(
        file_path=output_path,
        particle_count=len(particles),
        positive_count=n_pos,
        negative_count=n_neg,
        timestamp=timestamp,
    )


def export_summary(
    particles: list["Particle3D"],
    output_path: str | Path,
) -> Path:
    """
    Export summary statistics to a text file.
    
    Args:
        particles: List of Particle3D objects
        output_path: Path to output file
    
    Returns:
        Path to the created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_pos = sum(1 for p in particles if p.s > 0)
    n_neg = len(particles) - n_pos
    
    # Compute statistics
    if particles:
        cx = sum(p.x for p in particles) / len(particles)
        cy = sum(p.y for p in particles) / len(particles)
        cz = sum(p.z for p in particles) / len(particles)
        
        speeds = [(p.vx**2 + p.vy**2 + p.vz**2)**0.5 for p in particles]
        avg_speed = sum(speeds) / len(speeds)
        max_speed = max(speeds)
    else:
        cx = cy = cz = 0.0
        avg_speed = max_speed = 0.0
    
    with open(output_path, "w") as f:
        f.write(f"Janus 3D Simulation Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"\n")
        f.write(f"Particle Counts:\n")
        f.write(f"  Total: {len(particles)}\n")
        f.write(f"  M+ (positive): {n_pos}\n")
        f.write(f"  M- (negative): {n_neg}\n")
        f.write(f"\n")
        f.write(f"Center of Mass:\n")
        f.write(f"  X: {cx:.4f}\n")
        f.write(f"  Y: {cy:.4f}\n")
        f.write(f"  Z: {cz:.4f}\n")
        f.write(f"\n")
        f.write(f"Velocity Statistics:\n")
        f.write(f"  Average speed: {avg_speed:.4f}\n")
        f.write(f"  Max speed: {max_speed:.4f}\n")
    
    return output_path
