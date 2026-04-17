"""
Patch humidity in weather_data.csv with real data from Open-Meteo API.
Fetches relative_humidity_2m_mean for each unique city coordinate,
then replaces the estimated humidity with actual observed values.
"""

import numpy as np
import pandas as pd
import requests
import time
import os
import sys

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "weather_data.csv")


def fetch_humidity(lat, lon, start_year=2019, end_year=2024, retries=5):
    """Fetch daily humidity for a location, one year at a time."""
    all_data = []
    for year in range(start_year, end_year + 1):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "daily": "relative_humidity_2m_mean",
            "timezone": "auto",
        }
        for attempt in range(retries):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=20)
                if resp.status_code == 429:
                    wait = 30 * (2 ** attempt)
                    print(f"    [WAIT] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json().get("daily", {})
                dates = data.get("time", [])
                hum = data.get("relative_humidity_2m_mean", [])
                if dates:
                    all_data.extend(zip(dates, hum))
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"    [ERR] Failed for year {year}: {e}")
        time.sleep(1.5)  # Rate limit between year requests

    if not all_data:
        return None
    df = pd.DataFrame(all_data, columns=["date", "humidity"])
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


def main():
    print("=" * 60)
    print("[*] Patching humidity with real Open-Meteo data")
    print("=" * 60)

    # Load existing data
    df = pd.read_csv(DATA_PATH)
    print(f"[*] Loaded {len(df):,} rows from weather_data.csv")

    # Get unique city coordinates
    coords = df.groupby(["latitude", "longitude"]).size().reset_index()[["latitude", "longitude"]]
    print(f"[*] Found {len(coords)} unique city coordinates")

    # Fetch humidity for each coordinate
    humidity_lookup = {}
    for i, (_, row) in enumerate(coords.iterrows(), 1):
        lat, lon = row["latitude"], row["longitude"]
        print(f"\n[{i}/{len(coords)}] Fetching humidity for ({lat}, {lon})...")

        hum_df = fetch_humidity(lat, lon)
        if hum_df is not None:
            # Create lookup by (month, day_of_year)
            for _, hrow in hum_df.iterrows():
                key = (lat, lon, int(hrow["month"]), int(hrow["day_of_year"]))
                if pd.notna(hrow["humidity"]):
                    humidity_lookup[key] = hrow["humidity"]
            print(f"   OK {len(hum_df)} days of humidity data")
        else:
            print(f"   [WARN] No humidity data")

        time.sleep(2)  # Rate limit between cities

    print(f"\n[*] Total humidity records: {len(humidity_lookup):,}")

    # Patch humidity values
    patched = 0
    for idx in range(len(df)):
        key = (
            df.at[idx, "latitude"],
            df.at[idx, "longitude"],
            int(df.at[idx, "month"]),
            int(df.at[idx, "day_of_year"])
        )
        if key in humidity_lookup:
            df.at[idx, "humidity"] = round(humidity_lookup[key], 1)
            patched += 1

    print(f"[*] Patched {patched:,} / {len(df):,} humidity values ({patched/len(df)*100:.1f}%)")

    # Save
    df.to_csv(DATA_PATH, index=False)
    print(f"[*] Saved updated weather_data.csv")

    # Validate
    print("\n[*] Delhi humidity validation (after patch):")
    delhi = df[(df["latitude"].between(27, 30)) & (df["longitude"].between(76, 79))]
    if len(delhi) > 0:
        monthly_hum = delhi.groupby("month")["humidity"].mean()
        for m, h in monthly_hum.items():
            print(f"   Month {m}: {h:.1f}%")

    print("\n>> Next: Run train_model.py to retrain with real humidity data")


if __name__ == "__main__":
    main()
