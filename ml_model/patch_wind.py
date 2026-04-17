"""
Patch wind speed data with better Open-Meteo wind_speed_10m_max values
and replace Meteostat wind data for all cities.
"""
import numpy as np
import pandas as pd
import requests
import time
import os

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "weather_data.csv")


def fetch_wind(lat, lon, retries=3):
    """Fetch daily wind speed from Open-Meteo for a location."""
    all_data = []
    for year in range(2019, 2025):
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
            "daily": "wind_speed_10m_max",
            "timezone": "auto",
        }
        for attempt in range(retries):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(30 * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json().get("daily", {})
                dates = data.get("time", [])
                wind = data.get("wind_speed_10m_max", [])
                if dates:
                    all_data.extend(zip(dates, wind))
                break
            except:
                time.sleep(5)
        time.sleep(1.5)

    if not all_data:
        return None
    df = pd.DataFrame(all_data, columns=["date", "wind_speed"])
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


def main():
    print("=" * 60)
    print("[*] Patching wind speed with Open-Meteo wind_speed_10m_max")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"[*] Loaded {len(df):,} rows")

    # Get unique coordinates
    coords = df.groupby(["latitude", "longitude"]).size().reset_index()[["latitude", "longitude"]]
    print(f"[*] Found {len(coords)} unique city coordinates")

    wind_lookup = {}
    for i, (_, row) in enumerate(coords.iterrows(), 1):
        lat, lon = row["latitude"], row["longitude"]
        print(f"[{i}/{len(coords)}] Fetching wind for ({lat}, {lon})...")

        wind_df = fetch_wind(lat, lon)
        if wind_df is not None:
            for _, wrow in wind_df.iterrows():
                key = (lat, lon, int(wrow["month"]), int(wrow["day_of_year"]))
                if pd.notna(wrow["wind_speed"]):
                    wind_lookup[key] = wrow["wind_speed"]
            print(f"   OK {len(wind_df)} days")
        else:
            print(f"   [WARN] No wind data")

        time.sleep(2)

    print(f"\n[*] Total wind records: {len(wind_lookup):,}")

    # Patch wind values
    patched = 0
    for idx in range(len(df)):
        key = (
            df.at[idx, "latitude"],
            df.at[idx, "longitude"],
            int(df.at[idx, "month"]),
            int(df.at[idx, "day_of_year"])
        )
        if key in wind_lookup:
            df.at[idx, "wind_speed"] = round(wind_lookup[key], 1)
            patched += 1

    print(f"[*] Patched {patched:,} / {len(df):,} wind values ({patched/len(df)*100:.1f}%)")

    # Validate
    print("\n[*] Wind statistics after patch:")
    print(f"   Mean: {df['wind_speed'].mean():.1f} km/h")
    print(f"   Std:  {df['wind_speed'].std():.1f} km/h")
    print(f"   Min:  {df['wind_speed'].min():.1f}")
    print(f"   Max:  {df['wind_speed'].max():.1f}")

    # Delhi monthly wind
    delhi = df[(df["latitude"].between(27, 30)) & (df["longitude"].between(76, 79))]
    if len(delhi) > 0:
        print("\n[*] Delhi wind after patch:")
        for m in range(1, 13):
            mdata = delhi[delhi["month"] == m]["wind_speed"]
            if len(mdata) > 0:
                print(f"   Month {m:2d}: {mdata.mean():.1f} km/h (max daily: {mdata.max():.1f})")

    df.to_csv(DATA_PATH, index=False)
    print(f"\n[*] Saved to {DATA_PATH}")
    print(">> Next: Run train_model.py to retrain")


if __name__ == "__main__":
    main()
