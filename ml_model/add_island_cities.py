"""
Add Pacific Island and other underrepresented tropical oceanic cities
to fix predictions for remote island locations like Fiji.
"""
import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'weather_data.csv')

# Cities that need representation: Pacific Islands, Caribbean islands, Indian Ocean
NEW_CITIES = [
    # Pacific Islands
    ("Suva", -18.14, 178.44),
    ("Nadi", -17.77, 177.95),
    ("Apia", -13.83, -171.76),  # Samoa
    ("Noumea", -22.28, 166.46),  # New Caledonia
    ("Honiara", -9.43, 160.03),  # Solomon Islands
    
    # More Caribbean/Atlantic
    ("Kingston", 18.0, -76.8),   # Jamaica
    ("Port-au-Prince", 18.54, -72.34),  # Haiti
    ("San Juan", 18.47, -66.11),  # Puerto Rico
    
    # Indian Ocean  
    ("Antananarivo", -18.91, 47.52),  # Madagascar
    ("Mauritius", -20.16, 57.50),  # Mauritius
    
    # Tropical South America
    ("Belem", -1.46, -48.50),  # Amazon mouth
    ("Manaus", -3.12, -60.02),  # Already have? check
    
    # SE Asia islands
    ("Cebu", 10.31, 123.89),  # Philippines
]


def fetch_open_meteo(lat, lon, city_name):
    """Fetch historical daily weather from Open-Meteo Archive API."""
    all_rows = []
    
    for year in range(2019, 2025):
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                     "windspeed_10m_max,relative_humidity_2m_mean",
            "timezone": "auto"
        }
        
        for attempt in range(3):
            try:
                resp = requests.get("https://archive-api.open-meteo.com/v1/archive",
                                   params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(30)
                    continue
                resp.raise_for_status()
                data = resp.json().get("daily", {})
                
                dates = data.get("time", [])
                tmax = data.get("temperature_2m_max", [])
                tmin = data.get("temperature_2m_min", [])
                precip = data.get("precipitation_sum", [])
                wind = data.get("windspeed_10m_max", [])
                hum = data.get("relative_humidity_2m_mean", [])
                
                for i, d in enumerate(dates):
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    tx = tmax[i] if i < len(tmax) else None
                    tn = tmin[i] if i < len(tmin) else None
                    pr = precip[i] if i < len(precip) else None
                    ws = wind[i] if i < len(wind) else None
                    hu = hum[i] if i < len(hum) else None
                    
                    if tx is not None and tn is not None:
                        tavg = (tx + tn) / 2.0
                        rain = 1 if (pr is not None and pr > 0.5) else 0
                        all_rows.append({
                            'date': d,
                            'latitude': lat,
                            'longitude': lon,
                            'temperature': round(tavg, 1),
                            'humidity': round(hu, 1) if hu else 60.0,
                            'wind_speed': round(ws, 1) if ws else 10.0,
                            'month': dt.month,
                            'day_of_year': dt.timetuple().tm_yday,
                            'rain': rain,
                            'city': city_name,
                        })
                break
            except Exception as e:
                print(f"    Retry {attempt+1}: {e}")
                time.sleep(5)
        
        time.sleep(1.5)
    
    return all_rows


def main():
    print("=" * 60)
    print("[*] Adding Pacific Island & Tropical Oceanic cities")
    print("=" * 60)
    
    df = pd.read_csv(CSV_PATH)
    print(f"[*] Current dataset: {len(df):,} rows")
    
    existing_coords = set()
    for _, row in df.drop_duplicates(subset=['latitude', 'longitude']).iterrows():
        existing_coords.add((round(row['latitude'], 2), round(row['longitude'], 2)))
    
    total_added = 0
    for i, (name, lat, lon) in enumerate(NEW_CITIES, 1):
        key = (round(lat, 2), round(lon, 2))
        if key in existing_coords:
            print(f"[{i}/{len(NEW_CITIES)}] {name} - already exists, skipping")
            continue
        
        print(f"[{i}/{len(NEW_CITIES)}] Fetching {name} ({lat}, {lon})...", end=" ", flush=True)
        rows = fetch_open_meteo(lat, lon, name)
        
        if rows:
            new_df = pd.DataFrame(rows)
            temps = new_df['temperature']
            hums = new_df['humidity']
            print(f"OK {len(rows)} days | Temp: {temps.min():.0f}-{temps.max():.0f}C | Avg Hum: {hums.mean():.0f}%")
            df = pd.concat([df, new_df], ignore_index=True)
            total_added += len(rows)
        else:
            print("[WARN] No data")
        
        time.sleep(2)
    
    if total_added > 0:
        df.to_csv(CSV_PATH, index=False)
        print(f"\n[*] Added {total_added:,} new rows")
        print(f"[*] Total: {len(df):,} rows -> {CSV_PATH}")
        print("> Next: Run train_model.py")
    else:
        print("\n[*] No new cities added")


if __name__ == "__main__":
    main()
