"""
Fix geographic features (elevation, coast distance) in training data
using real Open-Meteo elevation API, add missing extreme-climate cities,
and retrain models.
"""
import numpy as np
import pandas as pd
import requests
import time
import os
import sys
from math import radians, cos, sin, asin, sqrt

from meteostat import Point, Daily
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "weather_data.csv")

# Additional extreme-climate cities to improve global coverage
EXTRA_CITIES = [
    # Saharan / hyper-arid
    ("Arlit", 18.7406, 7.3853),
    ("In Salah", 27.1928, 2.4680),
    ("Tamanrasset", 22.7851, 5.5228),
    ("Agadez", 16.9739, 7.9911),
    ("Atar", 20.5169, -13.0499),
    ("Nouakchott", 18.0735, -15.9582),
    ("Faya-Largeau", 17.9268, 19.1081),

    # Central Asian steppe / continental
    ("Ashgabat", 37.9601, 58.3261),
    ("Tashkent", 41.2995, 69.2401),
    ("Astana", 51.1694, 71.4491),
    ("Bishkek", 42.8746, 74.5698),
    ("Urumqi", 43.8256, 87.6168),

    # Inland Australia
    ("Toowoomba", -27.5598, 151.9507),
    ("Broken Hill", -31.9505, 141.4538),
    ("Longreach", -23.4363, 144.2500),
    ("Tennant Creek", -19.6497, 134.1910),
    ("Kalgoorlie", -30.7489, 121.4731),

    # Tropical wet / rainforest
    ("Iquitos", -3.7491, -73.2538),
    ("Libreville", 0.4162, 9.4673),
    ("Douala", 4.0511, 9.7679),
    ("Port Moresby", -9.4438, 147.1803),

    # High altitude
    ("Quito", -0.1807, -78.4678),   # already exists but ensure it's there
    ("Cusco", -13.5320, -71.9675),
    ("Addis Ababa", 9.0250, 38.7469),  # already exists
    ("Mexico City", 19.4326, -99.1332),  # already exists
]

START = datetime(2019, 1, 1)
END = datetime(2024, 12, 31)

# ---- Coastline points (denser set for better coast distance) ----
COAST_POINTS = [
    # Africa coasts
    (14.7, -17.5), (6.5, 3.4), (5.6, -0.2), (-6.8, 39.2), (-34.0, 18.4),
    (33.6, -7.6), (36.8, 10.2), (30.0, 31.2), (4.0, 9.8), (0.4, 9.5),
    (6.1, 1.2), (-4.3, 15.3), (-1.3, 36.8), (-12.0, 49.3), (-26.0, 32.6),
    # Europe coasts
    (51.5, -0.1), (48.9, 2.3), (41.0, 29.0), (38.7, -9.1), (37.9, 23.7),
    (59.9, 10.8), (60.2, 25.0), (64.1, -22.0), (53.3, -6.3), (55.7, 12.6),
    (43.3, 5.4), (40.4, -3.7), (41.9, 12.5), (45.4, 12.3),
    # Asia coasts
    (19.1, 72.9), (13.1, 80.3), (22.3, 114.2), (31.2, 121.5), (35.7, 139.7),
    (37.6, 127.0), (25.0, 121.5), (1.3, 103.8), (-6.2, 106.8), (13.8, 100.5),
    (14.6, 121.0), (10.8, 106.6), (6.9, 79.9), (21.0, 105.8), (3.1, 101.7),
    (25.2, 55.3), (23.6, 58.4), (24.9, 67.0), (23.8, 90.4), (34.5, 69.2),
    # Americas coasts
    (40.7, -74.0), (25.8, -80.2), (34.1, -118.2), (37.8, -122.4), (47.6, -122.3),
    (49.3, -123.1), (61.2, -150.0), (-22.9, -43.2), (-23.6, -46.6), (-34.6, -58.4),
    (-33.4, -70.7), (-12.0, -77.0), (10.5, -67.0), (23.1, -82.4), (19.4, -99.1),
    (29.8, -95.4), (14.6, -90.5),
    # Oceania coasts
    (-33.9, 151.2), (-37.8, 145.0), (-27.5, 153.0), (-31.9, 115.9), (-12.5, 130.8),
    (-36.8, 174.8), (-41.3, 174.8), (-43.5, 172.6),
    # Middle East coasts
    (25.3, 51.5), (31.8, 35.2), (32.1, 34.8), (36.2, 36.2),
    # Arctic coasts
    (69.6, 19.0), (69.0, 33.1), (63.7, -68.5),
]


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))


def compute_coast_distance(lat, lon):
    """Compute minimum distance to any known coast point."""
    min_dist = 99999
    for clat, clon in COAST_POINTS:
        d = haversine(lat, lon, clat, clon)
        min_dist = min(min_dist, d)
    return min(min_dist, 2000)


def get_real_elevations(lats, lons, batch_size=50):
    """Get real elevation from Open-Meteo for multiple coordinates."""
    elevations = {}
    for i in range(0, len(lats), batch_size):
        batch_lats = lats[i:i+batch_size]
        batch_lons = lons[i:i+batch_size]
        lat_str = ','.join(f"{l:.4f}" for l in batch_lats)
        lon_str = ','.join(f"{l:.4f}" for l in batch_lons)
        try:
            r = requests.get(
                f'https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}',
                timeout=15
            )
            r.raise_for_status()
            data = r.json()
            for j, elev in enumerate(data['elevation']):
                key = (round(batch_lats[j], 4), round(batch_lons[j], 4))
                elevations[key] = max(0, elev)  # clamp to 0 (no negative elevation)
        except Exception as e:
            print(f"   [WARN] Elevation batch failed: {e}")
        time.sleep(0.5)
    return elevations


def fetch_humidity_for_city(lat, lon, retries=3):
    """Fetch daily humidity from Open-Meteo for a city."""
    all_data = []
    for year in range(2019, 2025):
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
            "daily": "relative_humidity_2m_mean",
            "timezone": "auto",
        }
        for attempt in range(retries):
            try:
                resp = requests.get("https://archive-api.open-meteo.com/v1/archive",
                                    params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(30 * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json().get("daily", {})
                dates = data.get("time", [])
                hum = data.get("relative_humidity_2m_mean", [])
                if dates:
                    all_data.extend(zip(dates, hum))
                break
            except:
                time.sleep(5)
        time.sleep(1.5)

    if not all_data:
        return None
    df = pd.DataFrame(all_data, columns=["date", "humidity"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    print("=" * 65)
    print("[*] Geographic Feature Fix + Extra Cities")
    print("=" * 65)

    # ---- Step 1: Load existing data ----
    df = pd.read_csv(DATA_PATH)
    print(f"[*] Loaded {len(df):,} rows")

    # ---- Step 2: Fetch data for extra cities ----
    print(f"\n[*] Fetching data for {len(EXTRA_CITIES)} extra cities...")
    new_dfs = []
    for i, (name, lat, lon) in enumerate(EXTRA_CITIES, 1):
        # Check if this city is already in the data
        exists = len(df[(df['latitude'].round(1) == round(lat, 1)) &
                        (df['longitude'].round(1) == round(lon, 1))]) > 0
        if exists:
            print(f"   [{i}/{len(EXTRA_CITIES)}] {name} - already in dataset, skipping")
            continue

        print(f"   [{i}/{len(EXTRA_CITIES)}] Fetching {name} ({lat}, {lon})...")
        location = Point(lat, lon)
        data = Daily(location, START, END)
        met_df = data.fetch()

        if met_df.empty or len(met_df) < 100:
            print(f"      [WARN] No/insufficient data for {name}")
            continue

        met_df = met_df.reset_index()

        # Temperature
        met_df["temperature"] = met_df["tavg"]
        mask = met_df["temperature"].isna() & met_df["tmin"].notna() & met_df["tmax"].notna()
        met_df.loc[mask, "temperature"] = (met_df.loc[mask, "tmin"] + met_df.loc[mask, "tmax"]) / 2
        met_df = met_df.dropna(subset=["temperature"])

        if len(met_df) == 0:
            continue

        met_df["latitude"] = round(lat, 4)
        met_df["longitude"] = round(lon, 4)
        met_df["month"] = pd.to_datetime(met_df["time"]).dt.month
        met_df["day_of_year"] = pd.to_datetime(met_df["time"]).dt.dayofyear
        met_df["wind_speed"] = met_df["wspd"].fillna(10.0)
        met_df["rain"] = (met_df["prcp"].fillna(0) > 0.5).astype(int)
        met_df["elevation"] = 200  # placeholder, will be fixed below
        met_df["distance_to_coast"] = compute_coast_distance(lat, lon)

        # Fetch real humidity
        hum_df = fetch_humidity_for_city(lat, lon)
        if hum_df is not None:
            hum_df["day_of_year"] = hum_df["date"].dt.dayofyear
            hum_df["month"] = hum_df["date"].dt.month
            hum_lookup = dict(zip(
                zip(hum_df["month"], hum_df["day_of_year"]),
                hum_df["humidity"]
            ))
            met_df["humidity"] = met_df.apply(
                lambda r: hum_lookup.get((r["month"], r["day_of_year"]), np.nan), axis=1
            )
            met_df["humidity"] = met_df["humidity"].fillna(met_df["humidity"].median())
        else:
            met_df["humidity"] = 50.0  # fallback

        result = met_df[["latitude", "longitude", "month", "day_of_year",
                          "elevation", "distance_to_coast",
                          "temperature", "humidity", "wind_speed", "rain"]].copy()
        new_dfs.append(result)
        t_range = f"{met_df['temperature'].min():.1f}C - {met_df['temperature'].max():.1f}C"
        print(f"      OK {len(result):,} days | Temp: {t_range}")

    if new_dfs:
        extra = pd.concat(new_dfs, ignore_index=True)
        df = pd.concat([df, extra], ignore_index=True)
        print(f"   Added {len(extra):,} new rows from extra cities")
    print(f"   Total rows: {len(df):,}")

    # ---- Step 3: Fix elevation with real API data ----
    print(f"\n[*] Fetching real elevation for all unique coords...")
    unique_coords = df.groupby(["latitude", "longitude"]).size().reset_index()
    lats = unique_coords["latitude"].tolist()
    lons = unique_coords["longitude"].tolist()
    elevations = get_real_elevations(lats, lons)
    print(f"   Got elevation for {len(elevations)} coordinates")

    # Apply real elevations
    patched_elev = 0
    for idx in range(len(df)):
        key = (round(df.at[idx, "latitude"], 4), round(df.at[idx, "longitude"], 4))
        if key in elevations:
            df.at[idx, "elevation"] = elevations[key]
            patched_elev += 1
    print(f"   Patched {patched_elev:,} elevation values")

    # ---- Step 4: Fix coast distance ----
    print(f"\n[*] Recomputing coast distances...")
    coast_cache = {}
    for idx in range(len(df)):
        key = (df.at[idx, "latitude"], df.at[idx, "longitude"])
        if key not in coast_cache:
            coast_cache[key] = compute_coast_distance(key[0], key[1])
        df.at[idx, "distance_to_coast"] = coast_cache[key]
    print(f"   Updated {len(df):,} coast distance values")

    # ---- Step 5: Validate samples ----
    print(f"\n[*] Sample elevations and coast distances:")
    samples = [
        ("Delhi", 28.6, 77.2), ("Arlit", 18.7, 7.4), ("Toowoomba", -27.6, 152.0),
        ("Timbuktu", 16.8, -3.0), ("Singapore", 1.4, 103.8), ("Denver", 39.7, -105.0)
    ]
    for name, lat, lon in samples:
        subset = df[(df["latitude"].between(lat - 0.5, lat + 0.5)) &
                     (df["longitude"].between(lon - 0.5, lon + 0.5))]
        if len(subset) > 0:
            elev = subset["elevation"].iloc[0]
            coast = subset["distance_to_coast"].iloc[0]
            print(f"   {name:15s}: elev={elev:.0f}m, coast={coast:.0f}km")

    # ---- Step 6: Save ----
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"\n[*] Saved {len(df):,} rows to {DATA_PATH}")
    print(">> Next: Run train_model.py to retrain")


if __name__ == "__main__":
    main()
