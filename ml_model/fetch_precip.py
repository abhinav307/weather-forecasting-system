"""
Fetch monthly average precipitation (mm) for all training cities
and add to climate normals for KNN-based rainfall estimation.
"""
import json
import os
import requests
import time
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
NORMALS_PATH = os.path.join(MODELS_DIR, 'climate_normals.json')

def fetch_monthly_precip(lat, lon):
    """Fetch daily precipitation from Open-Meteo archive, return monthly totals."""
    monthly_rain = {}
    for year in range(2019, 2025):
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
            "daily": "precipitation_sum,wind_speed_10m_max",
            "timezone": "auto",
        }
        for attempt in range(3):
            try:
                resp = requests.get("https://archive-api.open-meteo.com/v1/archive",
                                     params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(30 * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json().get("daily", {})
                dates = data.get("time", [])
                precip = data.get("precipitation_sum", [])
                wind = data.get("wind_speed_10m_max", [])
                for d, p, w in zip(dates, precip, wind):
                    m = int(d.split("-")[1])
                    if m not in monthly_rain:
                        monthly_rain[m] = {"precip_days": [], "wind_vals": []}
                    if p is not None:
                        monthly_rain[m]["precip_days"].append(p)
                    if w is not None:
                        monthly_rain[m]["wind_vals"].append(w)
                break
            except Exception as e:
                time.sleep(5)
        time.sleep(1.0)

    result = {}
    for m in range(1, 13):
        if m in monthly_rain and monthly_rain[m]["precip_days"]:
            daily_precip = monthly_rain[m]["precip_days"]
            # Group by year-month to get monthly totals, then average
            days_per_month = len(daily_precip) / 6  # ~6 years
            total_precip = sum(daily_precip)
            avg_monthly_mm = total_precip / 6  # average over 6 years

            # Mean wind (average of daily max values ÷ gust factor)
            wind_vals = monthly_rain[m]["wind_vals"]
            mean_wind = np.mean(wind_vals) if wind_vals else 10.0

            result[m] = {
                "rainfall_mm": round(avg_monthly_mm, 1),
                "mean_wind_max": round(mean_wind, 1),
            }
    return result


def main():
    print("=" * 60)
    print("[*] Fetching real precipitation + wind data for climate normals")
    print("=" * 60)

    # Load existing normals
    with open(NORMALS_PATH, 'r') as f:
        normals = json.load(f)
    print(f"[*] Loaded normals for {len(normals)} cities")

    updated = 0
    total = len(normals)
    for i, (key, monthly) in enumerate(normals.items(), 1):
        lat, lon = map(float, key.split(","))
        print(f"[{i}/{total}] ({lat}, {lon})...", end=" ", flush=True)

        precip_data = fetch_monthly_precip(lat, lon)
        if precip_data:
            for m_str, m_data in monthly.items():
                m = int(m_str)
                if m in precip_data:
                    m_data["rainfall_mm"] = precip_data[m]["rainfall_mm"]
                    m_data["mean_wind_max"] = precip_data[m]["mean_wind_max"]
            updated += 1
            # Show sample
            if 7 in precip_data:
                print(f"Jul rain={precip_data[7]['rainfall_mm']}mm, wind_max={precip_data[7]['mean_wind_max']}kph")
            else:
                print("OK")
        else:
            print("[WARN] No data")

        time.sleep(1.5)

    # Save updated normals
    with open(NORMALS_PATH, 'w') as f:
        json.dump(normals, f)
    print(f"\n[*] Updated {updated}/{total} cities with real precipitation + wind data")
    print(f"[*] Saved -> {NORMALS_PATH}")

    # Validate: Print some cities
    print("\n[*] Sample validation:")
    test_cities = [
        ("Delhi", "28.6139,77.209"),
        ("Pune-ish", "18.5204,73.8567"),
    ]
    for name, key in test_cities:
        if key in normals:
            print(f"\n  {name}:")
            for m in [1, 4, 7, 10]:
                ms = str(m)
                if ms in normals[key]:
                    d = normals[key][ms]
                    rain_mm = d.get("rainfall_mm", "N/A")
                    wind = d.get("mean_wind_max", "N/A")
                    print(f"    Month {m}: rain={rain_mm}mm, wind_max={wind}kph, temp={d['temperature']}C")


if __name__ == "__main__":
    main()
