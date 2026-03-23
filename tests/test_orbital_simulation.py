"""
Tests for visualization/orbital_simulation.py
"""

import numpy as np
import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.orbital_simulation import (
    generate_debris_catalog,
    get_risk_distribution,
    get_altitude_distribution,
    get_collision_zones,
)


class TestGenerateDebrisCatalog:
    """Tests for debris catalog generation."""

    def test_correct_count(self):
        catalog = generate_debris_catalog(50)
        assert len(catalog) == 50

    def test_zero_count(self):
        catalog = generate_debris_catalog(0)
        assert len(catalog) == 0

    def test_required_keys(self):
        catalog = generate_debris_catalog(5)
        required_keys = [
            "id", "name", "lat", "lon", "altitude", "velocity",
            "size", "risk_level", "inclination", "orbit_type",
            "angular_speed", "orbit_phase",
        ]
        for debris in catalog:
            for key in required_keys:
                assert key in debris, f"Missing key: {key}"

    def test_altitude_range(self):
        catalog = generate_debris_catalog(100)
        for d in catalog:
            assert 200 <= d["altitude"] <= 2000

    def test_velocity_range(self):
        catalog = generate_debris_catalog(100)
        for d in catalog:
            assert 5.0 <= d["velocity"] <= 8.0

    def test_lat_lon_range(self):
        catalog = generate_debris_catalog(100)
        for d in catalog:
            assert -90 <= d["lat"] <= 90
            assert -180 <= d["lon"] <= 180

    def test_valid_orbit_types(self):
        catalog = generate_debris_catalog(100)
        valid_types = {"LEO", "MEO", "HEO"}
        for d in catalog:
            assert d["orbit_type"] in valid_types

    def test_valid_risk_levels(self):
        catalog = generate_debris_catalog(100)
        valid_risks = {"Low", "Medium", "High"}
        for d in catalog:
            assert d["risk_level"] in valid_risks

    def test_unique_ids(self):
        catalog = generate_debris_catalog(100)
        ids = [d["id"] for d in catalog]
        assert len(ids) == len(set(ids)), "Debris IDs should be unique"

    def test_orbit_type_matches_altitude(self):
        catalog = generate_debris_catalog(200)
        for d in catalog:
            if d["altitude"] < 600:
                assert d["orbit_type"] == "LEO"
            elif d["altitude"] < 1200:
                assert d["orbit_type"] == "MEO"
            else:
                assert d["orbit_type"] == "HEO"


class TestGetRiskDistribution:
    def test_all_categories_present(self):
        catalog = generate_debris_catalog(50)
        dist = get_risk_distribution(catalog)
        assert "High" in dist
        assert "Medium" in dist
        assert "Low" in dist

    def test_counts_sum_to_total(self):
        catalog = generate_debris_catalog(80)
        dist = get_risk_distribution(catalog)
        assert sum(dist.values()) == len(catalog)

    def test_empty_catalog(self):
        dist = get_risk_distribution([])
        assert dist == {"High": 0, "Medium": 0, "Low": 0}


class TestGetAltitudeDistribution:
    def test_all_categories_present(self):
        catalog = generate_debris_catalog(50)
        dist = get_altitude_distribution(catalog)
        assert "LEO" in dist
        assert "MEO" in dist
        assert "HEO" in dist

    def test_counts_sum_to_total(self):
        catalog = generate_debris_catalog(80)
        dist = get_altitude_distribution(catalog)
        assert sum(dist.values()) == len(catalog)


class TestGetCollisionZones:
    def test_returns_list(self):
        catalog = generate_debris_catalog(50)
        zones = get_collision_zones(catalog)
        assert isinstance(zones, list)

    def test_zone_has_required_keys(self):
        catalog = generate_debris_catalog(150)
        zones = get_collision_zones(catalog)
        if zones:  # may be empty with small datasets
            for zone in zones:
                assert "center_altitude" in zone
                assert "debris_count" in zone
                assert "risk" in zone

    def test_zone_debris_count_at_least_three(self):
        catalog = generate_debris_catalog(150)
        zones = get_collision_zones(catalog)
        for zone in zones:
            assert zone["debris_count"] >= 3

    def test_empty_catalog(self):
        zones = get_collision_zones([])
        assert zones == []
