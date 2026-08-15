"""
Generates realistic synthetic demo data that mirrors the structure of
METR-LA (207 sensors) and PeMS-BAY (325 sensors) datasets.

The generated data includes:
  - Traffic speed time series with realistic daily/weekly patterns
  - Sensor location coordinates (approximate LA area for METR-LA)
  - Adjacency matrix based on sensor distances
  - Uber Movement-style zone travel times

This script can be run standalone:
    python scripts/generate_demo_data.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── METR-LA Sensor Locations (approximate real locations in LA) ─────────
# Using a grid within the LA freeway network bounding box
LA_LAT_MIN, LA_LAT_MAX = 33.90, 34.17
LA_LON_MIN, LA_LON_MAX = -118.50, -118.10

# PeMS-BAY approximate locations in Bay Area
BAY_LAT_MIN, BAY_LAT_MAX = 37.20, 37.85
BAY_LON_MIN, BAY_LON_MAX = -122.50, -121.80


def generate_sensor_locations(n_sensors: int, lat_range: tuple, lon_range: tuple,
                              prefix: str = "SENSOR") -> pd.DataFrame:
    """Generate random sensor locations within a geographic bounding box."""
    np.random.seed(42 if prefix == "METR" else 123)
    lats = np.random.uniform(lat_range[0], lat_range[1], n_sensors)
    lons = np.random.uniform(lon_range[0], lon_range[1], n_sensors)
    sensor_ids = [f"{prefix}_{i:04d}" for i in range(n_sensors)]

    return pd.DataFrame({
        "sensor_id": sensor_ids,
        "latitude": lats,
        "longitude": lons,
        "source": prefix.lower().replace("_", "-"),
    })


def generate_india_mp_sensor_locations(n_sensors: int = 250) -> pd.DataFrame:
    """
    Generate sensor locations mapped to famous real roads in Madhya Pradesh:
      - Indore: MG Road, AB Road, Vijay Nagar Square, Ring Road, Palasia, Rajwada, Super Corridor
      - Bhopal: VIP Road, Hoshangabad Road, MP Nagar, New Market, Hamidia Road, BHEL Bypass, Kolar Road
      - Ujjain: Mahakal Lok Corridor, Dewas Road, Freeganj Main Rd, Nanakheda Circle
      - Gwalior: City Center, Lashkar Market, Maharaj Bada
      - Jabalpur: Narmada Road, Wright Town, Civic Center
    """
    np.random.seed(777)
    records = []

    indore_roads = [
        "MG Road (Mahatma Gandhi Rd)", "AB Road (Agra-Bombay Rd)", "Vijay Nagar Square (BRTS)",
        "Ring Road (Radisson Square)", "Palasia Square", "Rajwada Palace Circle",
        "Super Corridor (Airport Rd)", "Indore Bypass Highway", "Bhawarkua Square", "Khajrana Road"
    ]
    bhopal_roads = [
        "VIP Road (Upper Lake Drive)", "Hoshangabad Road Highway", "MP Nagar Zone-1",
        "MP Nagar Zone-2", "New Market Circle", "Hamidia Road (Station Rd)",
        "BHEL Bypass (Chetak Bridge)", "Kolar Road Corridor", "Ayodhya Bypass", "Bairagarh Road"
    ]
    ujjain_roads = [
        "Mahakal Lok Corridor Road", "Dewas Road Highway", "Freeganj Main Market Rd",
        "Nanakheda Bus Stand Circle", "Kshipra Bridge Ghat Road", "Agar Road Corridor"
    ]
    gwalior_roads = [
        "City Center Arterial Rd", "Lashkar Main Market", "Maharaj Bada Circle", "Gwalior Airport Road"
    ]
    jabalpur_roads = [
        "Narmada Road Highway", "Wright Town Circle", "Civic Center Market", "Gorakhpur Main Rd"
    ]

    clusters = [
        {"city": "Bhopal", "prefix": "BPL", "lat": 23.2599, "lon": 77.4126, "count": 70, "spread": 0.06, "roads": bhopal_roads},
        {"city": "Indore", "prefix": "IND", "lat": 22.7196, "lon": 75.8577, "count": 80, "spread": 0.06, "roads": indore_roads},
        {"city": "Gwalior", "prefix": "GWL", "lat": 26.2183, "lon": 78.1828, "count": 35, "spread": 0.04, "roads": gwalior_roads},
        {"city": "Jabalpur", "prefix": "JBP", "lat": 23.1815, "lon": 79.9864, "count": 35, "spread": 0.04, "roads": jabalpur_roads},
        {"city": "Ujjain", "prefix": "UJN", "lat": 23.1765, "lon": 75.7885, "count": 30, "spread": 0.03, "roads": ujjain_roads},
    ]

    for cluster in clusters:
        lats = np.random.normal(cluster["lat"], cluster["spread"] / 2, cluster["count"])
        lons = np.random.normal(cluster["lon"], cluster["spread"] / 2, cluster["count"])
        roads_list = cluster["roads"]
        for i in range(cluster["count"]):
            road_name = roads_list[i % len(roads_list)]
            sec_num = (i // len(roads_list)) + 1
            full_road_name = f"{road_name} (Sec {sec_num})" if sec_num > 1 else road_name
            records.append({
                "sensor_id": f"MP_{cluster['prefix']}_{i:03d}",
                "road_name": full_road_name,
                "city": cluster["city"],
                "latitude": round(float(lats[i]), 6),
                "longitude": round(float(lons[i]), 6),
                "source": "india-mp",
            })

    return pd.DataFrame(records)


def generate_adjacency_matrix(locations: pd.DataFrame, threshold_km: float = 5.0) -> np.ndarray:
    """
    Build an adjacency matrix based on geographic distance between sensors.
    Edges connect sensors within `threshold_km` of each other.
    Uses Haversine approximation.
    """
    n = len(locations)
    lats = np.radians(locations["latitude"].values)
    lons = np.radians(locations["longitude"].values)

    # Pairwise distance using Haversine
    lat_diff = lats[:, None] - lats[None, :]
    lon_diff = lons[:, None] - lons[None, :]
    a = np.sin(lat_diff / 2) ** 2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(lon_diff / 2) ** 2
    distances_km = 2 * 6371 * np.arcsin(np.sqrt(a))

    # Gaussian kernel weighted adjacency
    sigma = threshold_km / 2
    adj = np.exp(-(distances_km ** 2) / (2 * sigma ** 2))
    # Zero out self-loops and distant connections
    np.fill_diagonal(adj, 0)
    adj[distances_km > threshold_km] = 0

    return adj


def generate_traffic_speeds(n_sensors: int, n_days: int = 14,
                            interval_minutes: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic traffic speed data with realistic patterns:
      - Daily pattern: slow during rush hours, fast at night
      - Weekly pattern: weekdays vs weekends
      - Per-sensor variation: different baseline speeds
      - Noise: random fluctuations
      - Occasional congestion events
    """
    np.random.seed(seed)

    timestamps_per_day = (24 * 60) // interval_minutes
    total_timestamps = timestamps_per_day * n_days

    start_time = pd.Timestamp("2024-03-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=total_timestamps, freq=f"{interval_minutes}min")

    # Base speeds per sensor (40-70 mph range)
    base_speeds = np.random.uniform(40, 70, n_sensors)

    # Time-of-day pattern (normalized 0-1, lower = more congested)
    hours = np.array([t.hour + t.minute / 60 for t in timestamps])

    # Morning rush: 7-9 AM, Evening rush: 5-7 PM
    morning_dip = np.exp(-((hours - 8) ** 2) / (2 * 0.8 ** 2)) * 0.35
    evening_dip = np.exp(-((hours - 17.5) ** 2) / (2 * 1.0 ** 2)) * 0.40
    midday_slight = np.exp(-((hours - 12.5) ** 2) / (2 * 1.5 ** 2)) * 0.10
    time_factor = 1.0 - morning_dip - evening_dip - midday_slight

    # Weekend factor (less congestion)
    day_of_week = np.array([t.dayofweek for t in timestamps])
    weekend_mask = (day_of_week >= 5).astype(float)
    weekend_factor = 1.0 + weekend_mask * 0.15  # 15% faster on weekends

    # Build speed matrix: (timestamps, sensors)
    speeds = np.zeros((total_timestamps, n_sensors))
    for s in range(n_sensors):
        sensor_base = base_speeds[s]
        sensor_noise = np.random.normal(0, 2.5, total_timestamps)

        # Some sensors are more congestion-prone
        congestion_sensitivity = np.random.uniform(0.7, 1.3)

        speed_profile = sensor_base * (time_factor * congestion_sensitivity) * weekend_factor + sensor_noise

        # Add occasional random congestion events (5% of time)
        congestion_events = np.random.random(total_timestamps) < 0.05
        speed_profile[congestion_events] *= np.random.uniform(0.3, 0.6, congestion_events.sum())

        # Clip to realistic range
        speed_profile = np.clip(speed_profile, 5, 75)
        speeds[:, s] = speed_profile

    # Build DataFrame
    sensor_ids = [f"SENSOR_{s:04d}" for s in range(n_sensors)]
    df = pd.DataFrame(speeds, index=timestamps, columns=sensor_ids)
    df.index.name = "timestamp"

    return df


def generate_uber_movement_data(n_zones: int = 50, n_days: int = 14, seed: int = 99) -> pd.DataFrame:
    """
    Generate Uber Movement-style zone-to-zone travel time data.
    Aggregated at hourly intervals.
    """
    np.random.seed(seed)
    start_time = pd.Timestamp("2024-03-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=n_days * 24, freq="1h")

    records = []
    # Generate travel times for a subset of zone pairs
    zone_pairs = []
    for i in range(n_zones):
        # Each zone connects to 3-5 neighbors
        n_neighbors = np.random.randint(3, 6)
        neighbors = np.random.choice([z for z in range(n_zones) if z != i], n_neighbors, replace=False)
        for j in neighbors:
            zone_pairs.append((i, j))

    for ts in timestamps:
        hour = ts.hour
        # Rush hour slowdown
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_travel = 25  # minutes
        elif 22 <= hour or hour <= 5:
            base_travel = 10
        else:
            base_travel = 15

        for src, dst in zone_pairs:
            travel_time = base_travel + np.random.normal(0, 3)
            travel_time = max(5, travel_time)
            records.append({
                "timestamp": ts,
                "source_zone": f"ZONE_{src:03d}",
                "destination_zone": f"ZONE_{dst:03d}",
                "mean_travel_time_minutes": round(travel_time, 1),
            })

    return pd.DataFrame(records)


def main():
    """Generate all demo data files."""
    demo_dir = PROJECT_ROOT / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Smart Traffic Forecasting — Demo Data Generator")
    print("=" * 60)

    # ── METR-LA style data ──
    print("\n[1/6] Generating METR-LA sensor locations (207 sensors)...")
    metr_locations = generate_sensor_locations(
        207, (LA_LAT_MIN, LA_LAT_MAX), (LA_LON_MIN, LA_LON_MAX), prefix="METR"
    )
    metr_locations.to_csv(demo_dir / "metr_la_sensors.csv", index=False)
    print(f"      Saved → {demo_dir / 'metr_la_sensors.csv'}")

    print("[2/6] Generating METR-LA traffic speeds (207 sensors × 14 days × 5min)...")
    metr_speeds = generate_traffic_speeds(207, n_days=14, interval_minutes=5, seed=42)
    metr_speeds.columns = metr_locations["sensor_id"].tolist()
    metr_speeds.to_hdf(demo_dir / "demo_metr_la.h5", key="df", mode="w")
    print(f"      {metr_speeds.shape[0]} timestamps × {metr_speeds.shape[1]} sensors")
    print(f"      Saved → {demo_dir / 'demo_metr_la.h5'}")

    print("[3/6] Generating METR-LA adjacency matrix...")
    metr_adj = generate_adjacency_matrix(metr_locations, threshold_km=8.0)
    with open(demo_dir / "metr_la_adj.pkl", "wb") as f:
        pickle.dump(metr_adj, f)
    n_edges = (metr_adj > 0).sum()
    print(f"      {n_edges} edges in adjacency matrix")
    print(f"      Saved → {demo_dir / 'metr_la_adj.pkl'}")

    # ── PeMS-BAY style data ──
    print("\n[4/6] Generating PeMS-BAY sensor locations (325 sensors)...")
    pems_locations = generate_sensor_locations(
        325, (BAY_LAT_MIN, BAY_LAT_MAX), (BAY_LON_MIN, BAY_LON_MAX), prefix="PEMS"
    )
    pems_locations.to_csv(demo_dir / "pems_bay_sensors.csv", index=False)
    print(f"      Saved → {demo_dir / 'pems_bay_sensors.csv'}")

    print("[5/6] Generating PeMS-BAY traffic speeds (325 sensors × 14 days × 5min)...")
    pems_speeds = generate_traffic_speeds(325, n_days=14, interval_minutes=5, seed=123)
    pems_speeds.columns = pems_locations["sensor_id"].tolist()
    pems_speeds.to_hdf(demo_dir / "demo_pems_bay.h5", key="df", mode="w")
    print(f"      {pems_speeds.shape[0]} timestamps × {pems_speeds.shape[1]} sensors")
    print(f"      Saved → {demo_dir / 'demo_pems_bay.h5'}")

    # ── Uber Movement style data ──
    print("\n[6/7] Generating Uber Movement travel times (50 zones × 14 days)...")
    uber_data = generate_uber_movement_data(n_zones=50, n_days=14)
    uber_data.to_csv(demo_dir / "demo_uber_movement.csv", index=False)
    print(f"      {len(uber_data)} records")
    print(f"      Saved → {demo_dir / 'demo_uber_movement.csv'}")

    # ── India MP Traffic Network data ──
    print("\n[7/7] Generating India MP sensor locations (250 sensors across Bhopal, Indore, Gwalior, Jabalpur, Ujjain)...")
    mp_locations = generate_india_mp_sensor_locations(250)
    mp_locations.to_csv(demo_dir / "india_mp_sensors.csv", index=False)
    print(f"      Saved → {demo_dir / 'india_mp_sensors.csv'}")

    print("      Generating India MP traffic speeds (250 sensors × 14 days × 5min)...")
    mp_speeds = generate_traffic_speeds(250, n_days=14, interval_minutes=5, seed=777)
    mp_speeds.columns = mp_locations["sensor_id"].tolist()
    mp_speeds.to_hdf(demo_dir / "demo_india_mp.h5", key="df", mode="w")
    print(f"      {mp_speeds.shape[0]} timestamps × {mp_speeds.shape[1]} sensors")
    print(f"      Saved → {demo_dir / 'demo_india_mp.h5'}")

    print("      Generating India MP adjacency matrix...")
    mp_adj = generate_adjacency_matrix(mp_locations, threshold_km=15.0)
    with open(demo_dir / "india_mp_adj.pkl", "wb") as f:
        pickle.dump(mp_adj, f)
    print(f"      Saved → {demo_dir / 'india_mp_adj.pkl'}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Demo data generation complete!")
    print(f"  Output directory: {demo_dir}")
    print("=" * 60)
    print(f"\n  Files generated:")
    for f in sorted(demo_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name:30s}  {size_mb:6.2f} MB")


if __name__ == "__main__":
    main()
