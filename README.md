# Janus 3D - Particle Simulation

A 3D N-body particle simulation implementing the Janus cosmological model with positive and negative mass particles.

![Janus Simulation](docs/preview.png)

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
git clone https://github.com/YOUR_USERNAME/janus_AI_AntiGravity.git
cd janus_AI_AntiGravity

# Create conda environment
conda create -n janus python=3.11
conda activate janus

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Running the Simulation

#### Native Pyglet Renderer
```bash
python -m particle_sim3d.main
```

#### Web Interface (React + Three.js)
```bash
# Terminal 1: Start API server
python -m particle_sim3d.run_api

# Terminal 2: Start frontend dev server
cd frontend && npm run dev
```

Then open http://localhost:3000

## Project Structure

```
├── particle_sim3d/
│   ├── core/           # Simulation core (sim.py, init_conditions.py)
│   ├── physics/        # Force calculations (forces.py, octree.py, metal_backend.py)
│   ├── rendering/      # Pyglet renderer
│   ├── ui/             # Native UI (app.py, menu.py)
│   ├── api/            # FastAPI backend for web interface
│   └── utils/          # Export, recording, benchmarks
├── frontend/           # React + Three.js web frontend
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
