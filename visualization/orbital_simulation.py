"""
Space Debris AI — Orbital Simulation Data Generator
Generates simulated debris objects with orbital parameters.
"""

import numpy as np
import random
from typing import List, Dict


# Debris name prefixes for variety
DEBRIS_PREFIXES = [
    "COSMOS", "FENGYUN", "IRIDIUM", "SL-", "CZ-", "DELTA", "ATLAS",
    "BREEZE", "TITAN", "ARIANE", "VEGA", "PROTON", "ZENIT", "PEGASUS",
]


def generate_debris_catalog(count: int = 150) -> List[Dict]:
    """
    Generate a catalog of simulated space debris objects.

    Each debris object has:
        - id: unique identifier
        - name: human-readable name
        - lat: latitude (-90 to 90)
        - lon: longitude (-180 to 180)
        - altitude: orbital altitude in km (200 - 2000)
        - velocity: orbital velocity in km/s (6.5 - 7.8)
        - size: estimated size in cm
        - risk_level: Low / Medium / High
        - inclination: orbital inclination (degrees)
        - orbit_type: LEO / MEO / HEO
    """
    debris_list = []

    for i in range(count):
        altitude = random.uniform(200, 2000)

        # Determine orbit type
        if altitude < 600:
            orbit_type = "LEO"
        elif altitude < 1200:
            orbit_type = "MEO"
        else:
            orbit_type = "HEO"

        # Velocity inversely related to altitude (simplified)
        velocity = round(7.8 - (altitude - 200) * 0.0007, 2)

        # Size affects risk
        size = round(random.uniform(0.5, 50.0), 1)
        density_factor = random.random()

        if size > 20 or density_factor > 0.7:
            risk_level = "High"
        elif size > 5 or density_factor > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        prefix = random.choice(DEBRIS_PREFIXES)
        debris_id = f"DEB-{i+1:04d}"
        name = f"{prefix}-{random.randint(1000, 9999)} Debris"

        debris_list.append({
            "id": debris_id,
            "name": name,
            "lat": round(random.uniform(-90, 90), 2),
            "lon": round(random.uniform(-180, 180), 2),
            "altitude": round(altitude, 1),
            "velocity": velocity,
            "size": size,
            "risk_level": risk_level,
            "inclination": round(random.uniform(0, 98), 1),
            "orbit_type": orbit_type,
            "angular_speed": round(random.uniform(0.001, 0.02), 4),
            "orbit_phase": round(random.uniform(0, 2 * np.pi), 4),
        })

    return debris_list


def get_risk_distribution(debris_list: List[Dict]) -> Dict:
    """Get count of debris by risk level."""
    dist = {"High": 0, "Medium": 0, "Low": 0}
    for d in debris_list:
        dist[d["risk_level"]] = dist.get(d["risk_level"], 0) + 1
    return dist


def get_altitude_distribution(debris_list: List[Dict]) -> Dict:
    """Get count of debris by orbit type."""
    dist = {"LEO": 0, "MEO": 0, "HEO": 0}
    for d in debris_list:
        dist[d["orbit_type"]] = dist.get(d["orbit_type"], 0) + 1
    return dist


def get_collision_zones(debris_list: List[Dict], proximity_km: float = 50.0) -> List[Dict]:
    """Identify clusters of debris that could represent collision zones."""
    zones = []
    checked = set()

    for i, d1 in enumerate(debris_list):
        if i in checked:
            continue
        cluster = [d1]
        for j, d2 in enumerate(debris_list[i + 1:], start=i + 1):
            if j in checked:
                continue
            alt_diff = abs(d1["altitude"] - d2["altitude"])
            if alt_diff < proximity_km:
                cluster.append(d2)
                checked.add(j)

        if len(cluster) >= 3:
            avg_alt = np.mean([d["altitude"] for d in cluster])
            zones.append({
                "center_altitude": round(avg_alt, 1),
                "debris_count": len(cluster),
                "risk": "High" if len(cluster) >= 6 else "Medium",
            })
            checked.add(i)

    return zones
