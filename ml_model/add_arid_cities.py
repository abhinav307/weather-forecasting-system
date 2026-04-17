"""
Add arid Middle Eastern cities to fix Iranian/Central Asian predictions.
"""
import numpy as np
import pandas as pd
import requests
import time
import os
from math import radians, cos, sin, asin, sqrt
from meteostat import Point, Daily
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "weather_data.csv")

# Arid cities that are missing from our training data
EXTRA_CITIES = [
    # Iran - arid plateau
    ("Isfahan", 32.6546, 51.6680),
    ("Yazd", 31.8974, 54.3569),
    ("Kerman", 30.2839, 57.0834),
    ("Shiraz", 29.5918, 52.5837),
    ("Tabriz", 38.0800, 46.2919),
    ("Mashhad", 36.2605, 59.6168),

    # Central Asia arid
    ("Dushanbe", 38.5598, 68.7740),
    ("Samarkand", 39.6270, 66.9750),

    # Middle East arid
    ("Riyadh", 24.7136, 46.6753),
    ("Kuwait City", 29.3759, 47.9774),
    ("Muscat", 23.5880, 58.3829),
    ("Amman", 31.9454, 35.9284),  # check if exists
    ("Doha", 25.2854, 51.5310),

    # North Africa arid
    ("Tripoli", 32.8872, 13.1913),
    ("Benghazi", 32.1194, 20.0868),
]

START = datetime(2019, 1, 1)
END = datetime(2024, 12, 31)

COAST_POINTS = [
    (14.7, -17.5), (6.5, 3.4), (5.6, -0.2), (-6.8, 39.2), (-34.0, 18.4),
    (33.6, -7.6), (36.8, 10.2), (30.0, 31.2), (4.0, 9.8), (0.4, 9.5),
    (-26.0, 32.6), (-1.3, 36.8),
    (51.5, -0.1), (48.9, 2.3), (41.0, 29.0), (38.7, -9.1), (37.9, 23.7),
    (59.9, 10.8), (60.2, 25.0), (64.1, -22.0), (53.3, -6.3), (55.7, 12.6),
    (43.3, 5.4), (40.4, -3.7), (41.9, 12.5), (45.4, 12.3),
    (19.1, 72.9), (13.1, 80.3), (22.3, 114.2), (31.2, 121.5), (35.7, 139.7),
    (37.6, 127.0), (25.0, 121.5), (1.3, 103.8), (-6.2, 106.8), (13.8, 100.5),
    (14.6, 121.0), (10.8, 106.6), (6.9, 79.9), (21.0, 105.8), (3.1, 101.7),
    (25.2, 55.3), (23.6, 58.4), (24.9, 67.0), (23.8, 90.4),
    (40.7, -74.0), (25.8, -80.2), (34.1, -118.2), (37.8, -122.4), (47.6, -122.3),
    (49.3, -123.1), (61.2, -150.0), (-22.9, -43.2), (-23.6, -46.6), (-34.6, -58.4),
    (-33.4, -70.7), (-12.0, -77.0), (10.5, -67.0), (23.1, -82.4), (29.8, -95.4),
    (-33.9, 151.2), (-37.8, 145.0), (-27.5, 153.0), (-31.9, 115.9), (-12.5, 130.8),
    (-36.8, 174.8), (-41.3, 174.8),
    (25.3, 51.5), (31.8, 35.2), (32.1, 34.8),
    (69.6, 19.0), (69.0, 33.1), (63.7, -68.5),
]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def coast_dist(lat, lon):
    return min(min(haversine(lat, lon, cl, co) for cl, co in COAST_POINTS), 2000)

def get_elevation(lat, lon):
    try:
        r = requests.get(f'https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}', timeout=10)
        return max(0, r.json()['elevation'][0])
    except:
        return 200

def fetch_humidity(lat, lon):
    all_data = []
    for year in range(2019, 2025):
        params = {"latitude": lat, "longitude": lon,
                  "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
                  "daily": "relative_humidity_2m_mean", "timezone": "auto"}
        for attempt in range(3):
            try:
                resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(30); continue
                resp.raise_for_status()
                d = resp.json().get("daily", {})
                if d.get("time"):
                    all_data.extend(zip(d["time"], d["relative_humidity_2m_mean"]))
                break
            except: time.sleep(5)
        time.sleep(1.5)
    if not all_data: return None
    df = pd.DataFrame(all_data, columns=["date", "humidity"])
    df["date"] = pd.to_datetime(df["date"])
    return df

def fetch_wind(lat, lon):
    all_data = []
    for year in range(2019, 2025):
        params = {"latitude": lat, "longitude": lon,
                  "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
                  "daily": "wind_speed_10m_max", "timezone": "auto"}
        for attempt in range(3):
            try:
                resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(30); continue
                resp.raise_for_status()
                d = resp.json().get("daily", {})
                if d.get("time"):
                    all_data.extend(zip(d["time"], d["wind_speed_10m_max"]))
                break
            except: time.sleep(5)
        time.sleep(1.5)
    if not all_data: return None
    df = pd.DataFrame(all_data, columns=["date", "wind_speed"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    print("=" * 60)
    print("[*] Adding arid Middle Eastern cities")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"[*] Current dataset: {len(df):,} rows")

    new_dfs = []
    for i, (name, lat, lon) in enumerate(EXTRA_CITIES, 1):
        exists = len(df[(df['latitude'].round(1) == round(lat, 1)) &
                        (df['longitude'].round(1) == round(lon, 1))]) > 0
        if exists:
            print(f"[{i}/{len(EXTRA_CITIES)}] {name} - already exists, skipping")
            continue

        print(f"[{i}/{len(EXTRA_CITIES)}] Fetching {name} ({lat}, {lon})...")

        # Meteostat for temp + rain
        location = Point(lat, lon)
        met_df = Daily(location, START, END).fetch()
        if met_df.empty or len(met_df) < 100:
            print(f"   [WARN] No/insufficient Meteostat data")
            continue

        met_df = met_df.reset_index()
        met_df["temperature"] = met_df["tavg"]
        mask = met_df["temperature"].isna() & met_df["tmin"].notna() & met_df["tmax"].notna()
        met_df.loc[mask, "temperature"] = (met_df.loc[mask, "tmin"] + met_df.loc[mask, "tmax"]) / 2
        met_df = met_df.dropna(subset=["temperature"])
        if len(met_df) == 0: continue

        met_df["latitude"] = round(lat, 4)
        met_df["longitude"] = round(lon, 4)
        met_df["month"] = pd.to_datetime(met_df["time"]).dt.month
        met_df["day_of_year"] = pd.to_datetime(met_df["time"]).dt.dayofyear
        met_df["rain"] = (met_df["prcp"].fillna(0) > 0.5).astype(int)

        # Real elevation
        elev = get_elevation(lat, lon)
        met_df["elevation"] = elev
        met_df["distance_to_coast"] = coast_dist(lat, lon)

        # Real humidity from Open-Meteo
        hum_df = fetch_humidity(lat, lon)
        if hum_df is not None:
            hum_df["day_of_year"] = hum_df["date"].dt.dayofyear
            hum_df["month"] = hum_df["date"].dt.month
            hum_lookup = dict(zip(zip(hum_df["month"], hum_df["day_of_year"]), hum_df["humidity"]))
            met_df["humidity"] = met_df.apply(
                lambda r: hum_lookup.get((r["month"], r["day_of_year"]), np.nan), axis=1)
            met_df["humidity"] = met_df["humidity"].fillna(met_df["humidity"].median())
        else:
            met_df["humidity"] = 30.0

        # Real wind from Open-Meteo
        wind_df = fetch_wind(lat, lon)
        if wind_df is not None:
            wind_df["day_of_year"] = wind_df["date"].dt.dayofyear
            wind_df["month"] = wind_df["date"].dt.month
            wind_lookup = dict(zip(zip(wind_df["month"], wind_df["day_of_year"]), wind_df["wind_speed"]))
            met_df["wind_speed"] = met_df.apply(
                lambda r: wind_lookup.get((r["month"], r["day_of_year"]), np.nan), axis=1)
            met_df["wind_speed"] = met_df["wind_speed"].fillna(met_df["wind_speed"].median())
        else:
            met_df["wind_speed"] = met_df["wspd"].fillna(10.0)

        result = met_df[["latitude", "longitude", "month", "day_of_year",
                          "elevation", "distance_to_coast",
                          "temperature", "humidity", "wind_speed", "rain"]].copy()
        new_dfs.append(result)
        avg_h = result["humidity"].mean()
        t_range = f"{result['temperature'].min():.0f}-{result['temperature'].max():.0f}C"
        print(f"   OK {len(result):,} days | Temp: {t_range} | Avg Hum: {avg_h:.0f}%")

    if new_dfs:
        extra = pd.concat(new_dfs, ignore_index=True)
        df = pd.concat([df, extra], ignore_index=True)
        print(f"\n[*] Added {len(extra):,} new rows")

    df.to_csv(DATA_PATH, index=False)
    print(f"[*] Total: {len(df):,} rows -> {DATA_PATH}")
    print(">> Next: Run train_model.py")


if __name__ == "__main__":
    main()
