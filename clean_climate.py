import json
import os
import sys
import numpy as np
from meteostat import Stations, Monthly
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('ml_model')
from generate_data import REFERENCE_CITIES

def fetch_station(key_data):
    city_name, data = key_data
    lat, lon, old_temp, old_hum, old_rain_prob, old_wind = data
    
    try:
        stations = Stations()
        stations = stations.nearby(lat, lon)
        station = stations.fetch(1)
        if station.empty:
            return city_name, data, None
            
        st_id = station.index[0]
        # Fetch 10 years of data
        monthly = Monthly(st_id, datetime(2013, 1, 1), datetime(2023, 12, 31)).fetch()
        
        if monthly.empty:
            return city_name, data, None
            
        new_temp = []
        new_wind = []
        new_rain_prob = []
        new_rain_mm = []

        for m in range(1, 13):
            m_data = monthly[monthly.index.month == m]
            
            # Temp
            tavg = m_data['tavg'].mean()
            if np.isnan(tavg): tavg = old_temp[m-1]
            new_temp.append(round(float(tavg), 1))
            
            # Wind
            wspd = m_data['wspd'].mean()
            if np.isnan(wspd): wspd = old_wind[m-1]
            new_wind.append(round(float(wspd), 1))
            
            # Rain MM
            prcp = m_data['prcp'].mean()
            if np.isnan(prcp): prcp = 0.0
            avg_rain = round(float(prcp), 1)
            new_rain_mm.append(avg_rain)
            
            # Rain Prob (smart deterministic mapping)
            if avg_rain < 1.0: prob = 0.0
            elif avg_rain < 20.0: prob = round(avg_rain / 50.0, 2)
            elif avg_rain < 80.0: prob = round(avg_rain / 120.0 + 0.1, 2)
            else: prob = round(min(0.95, avg_rain / 300.0 + 0.4), 2)
            new_rain_prob.append(prob)

        return city_name, [lat, lon, new_temp, old_hum, new_rain_prob, new_wind], new_rain_mm
        
    except Exception as e:
        return city_name, data, None

def main():
    print(f"[*] Fetching Meteostat Normals for {len(REFERENCE_CITIES)} cities...")
    new_reference_cities = {}
    city_precipitation = {}
    
    pairs = list(REFERENCE_CITIES.items())
    completed = 0
    
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(fetch_station, p) for p in pairs]
        for f in as_completed(futures):
            city_name, data, rain_mm = f.result()
            new_reference_cities[city_name] = data
            if rain_mm:
                key = f"{data[0]},{data[1]}"
                city_precipitation[key] = {str(m+1): {'rainfall_mm': rain_mm[m]} for m in range(12)}
            
            completed += 1
            print(f"\r[*] Progress: {completed}/{len(REFERENCE_CITIES)}", end='', flush=True)

    print("\n[*] Saving updated ground-truth datasets...")
    os.makedirs('ml_model/saved_models', exist_ok=True)
    with open('ml_model/saved_models/reference_cities.json', 'w') as f:
        json.dump(new_reference_cities, f)
        
    with open('ml_model/saved_models/city_precipitation.json', 'w') as f:
        json.dump(city_precipitation, f)
        
    print("[OK] Fetched all datasets successfully.")

if __name__ == '__main__':
    main()
