# Janus 3D - Particle Simulation

A 3D N-body particle simulation implementing the Janus cosmological model with positive and negative mass particles.

## Features

- **Janus Physics**: Simulation of positive (M+) and negative (M-) mass particles with anti-gravity interactions
- **High Performance**: Metal GPU acceleration on macOS, Barnes-Hut tree algorithm for O(N log N) force calculation
- **Dual Frontend**: 
  - Native Pyglet/OpenGL renderer
  - React + Three.js web interface with real-time WebSocket streaming
- **Rich Controls**: Full parameter panel with 50+ adjustable simulation parameters

## Quick Start

### Prerequisites

- Python 3.11+
- macOS (for Metal GPU support) or Linux/Windows (CPU mode)
- Node.js 18+ (for React frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/SamyTahar/janus-model.git
cd janus-model

# Create conda environment
conda create -n janus python=3.11
conda activate janus

# Install Python dependencies
pip install -r requirements.txt

```

### Running the Simulation

#### Native Pyglet Renderer
```bash
python -m particle_sim3d.main
```

## Project Structure

```
├── particle_sim3d/
│   ├── core/           # Simulation core (sim.py, init_conditions.py)
│   ├── physics/        # Force calculations (forces.py, octree.py, metal_backend.py)
│   ├── output/         # Videos and image output folder
│   ├── rendering/      # Pyglet renderer
│   └── utils/          # Export, recording, benchmarks
└── docs/               # Documentation
```

## Configuration

Edit `particle_sim3d/params.json` to customize:
- Particle counts and distribution
- Force backend (cpu/metal)
- Galaxy parameters
- Visual settings

## The Janus Model

This simulation implements concepts from the Janus cosmological model:
- **Positive mass (M+)**: Normal matter, attracts other positive mass
- **Negative mass (M-)**: "Dark matter" analog, repels positive mass
- **Anti-gravity**: M+ and M- repel each other, creating cosmic structure

## License

MIT License

## Acknowledgments

Based on the Janus cosmological model by Jean-Pierre Petit.
