"""
Unit tests for Janus API.

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from particle_sim3d.api.server import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


# =============================================================================
# Status Tests
# =============================================================================

class TestStatus:
    """Test API status endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["engine"] == "janus"
    
    def test_status(self, client):
        """Test status endpoint."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "1.0.0"


# =============================================================================
# Simulation Tests
# =============================================================================

class TestSimulation:
    """Test simulation control endpoints."""
    
    def test_get_state(self, client):
        """Test getting simulation state."""
        response = client.get("/api/v1/simulation")
        assert response.status_code == 200
        data = response.json()
        assert "running" in data
        assert "particle_count" in data
        assert "sim_time" in data
    
    def test_start(self, client):
        """Test starting simulation."""
        response = client.post("/api/v1/simulation/start")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        
        # Cleanup: pause
        client.post("/api/v1/simulation/pause")
    
    def test_pause(self, client):
        """Test pausing simulation."""
        response = client.post("/api/v1/simulation/pause")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"
    
    def test_reset(self, client):
        """Test resetting simulation."""
        response = client.post("/api/v1/simulation/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"
        assert data["sim_time"] == 0.0
    
    def test_step(self, client):
        """Test stepping simulation."""
        # Reset first
        client.post("/api/v1/simulation/reset")
        
        response = client.post("/api/v1/simulation/step?dt=0.016")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stepped"
        assert data["sim_time"] > 0
    
    def test_step_invalid_dt(self, client):
        """Test step with invalid dt."""
        response = client.post("/api/v1/simulation/step?dt=-1")
        assert response.status_code == 400


# =============================================================================
# Parameters Tests
# =============================================================================

class TestParams:
    """Test parameter endpoints."""
    
    def test_get_params(self, client):
        """Test getting parameters."""
        response = client.get("/api/v1/params")
        assert response.status_code == 200
        data = response.json()
        assert "time_scale" in data
        assert "particle_count" in data
        assert "janus_g" in data
    
    def test_update_params(self, client):
        """Test updating parameters."""
        # Get original
        original = client.get("/api/v1/params").json()
        original_scale = original["time_scale"]
        
        # Update
        response = client.patch("/api/v1/params", json={"time_scale": 2.0})
        assert response.status_code == 200
        data = response.json()
        assert data["time_scale"] == 2.0
        
        # Restore
        client.patch("/api/v1/params", json={"time_scale": original_scale})


# =============================================================================
# Particles Tests
# =============================================================================

class TestParticles:
    """Test particle data endpoints."""
    
    def test_get_particles(self, client):
        """Test getting particle list."""
        response = client.get("/api/v1/particles?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "particles" in data
        assert len(data["particles"]) <= 10
    
    def test_get_particle_frame(self, client):
        """Test getting particle frame."""
        response = client.get("/api/v1/particles/frame")
        assert response.status_code == 200
        data = response.json()
        assert "frame" in data
        assert "positions" in data
        assert "colors" in data
    
    def test_get_particle_count(self, client):
        """Test getting particle counts."""
        response = client.get("/api/v1/particles/count")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "positive" in data
        assert "negative" in data
        assert data["total"] == data["positive"] + data["negative"]


# =============================================================================
# Export Tests
# =============================================================================

class TestExport:
    """Test export endpoints."""
    
    def test_export_csv(self, client):
        """Test CSV export."""
        response = client.post("/api/v1/export/csv")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "filename" in data
        assert data["particle_count"] > 0
