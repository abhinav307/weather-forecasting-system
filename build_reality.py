import requests
import json
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('ml_model')
from generate_data import REFERENCE_CITIES

def fetch_climate(key_data):
    city_name, data = key_data
    lat, lon, old_temp, old_hum, old_rain_prob, old_wind = data
    
    params = {
        'latitude': lat, 
        'longitude': lon, 
        'start_date': '2010-01-01', 
        'end_date': '2019-12-31', 
        'models': 'CMCC_CM2_VHR4', 
        'daily': 'precipitation_sum,temperature_2m_max,wind_speed_10m_max'
    }
    
    try:
        resp = requests.get('https://climate-api.open-meteo.com/v1/climate', params=params, timeout=30)
        resp.raise_for_status()
        daily = resp.json()['daily']
        
        times = daily['time']
        precip = daily['precipitation_sum_CMCC_CM2_VHR4']
        temp = daily.get('temperature_2m_max_CMCC_CM2_VHR4', [])
        wind = daily.get('wind_speed_10m_max_CMCC_CM2_VHR4', [])
        
        month_data = {m: {'precip': [], 'temp': [], 'wind': [], 'rain_days': 0, 'total': 0} for m in range(1, 13)}
        
        for i in range(len(times)):
            ymd = times[i].split('-')
            m = int(ymd[1])
            p = precip[i]
            t = temp[i] if i < len(temp) else None
            w = wind[i] if i < len(wind) else None
            
            if p is not None:
                month_data[m]['precip'].append(p)
                if p > 1.0:
                    month_data[m]['rain_days'] += 1
            if t is not None:
                month_data[m]['temp'].append(t)
            if w is not None:
                month_data[m]['wind'].append(w)
                
        # 10 years of data = 10 * 12 = 120 months.
        years = 10.0
        
        new_temp = []
        new_wind = []
        new_rain_prob = []
        new_rainfall_mm = []
        
        for m in range(1, 13):
            # Monthly average high temp
            avg_t = round(sum(month_data[m]['temp']) / len(month_data[m]['temp']), 1) if month_data[m]['temp'] else old_temp[m-1]
            new_temp.append(avg_t)
            
            # Monthly average max wind
            avg_w = round(sum(month_data[m]['wind']) / len(month_data[m]['wind']), 1) if month_data[m]['wind'] else old_wind[m-1]
            new_wind.append(avg_w)
            
            # Rain probability (fraction of rainy days > 1.0mm)
            total_days = len(month_data[m]['precip'])
            prob = round(month_data[m]['rain_days'] / total_days, 2) if total_days > 0 else old_rain_prob[m-1]
            new_rain_prob.append(prob)
            
            # Rainfall mm per month (total precipitation over 10 years / 10)
            avg_rain_mm = round(sum(month_data[m]['precip']) / years, 1)
            new_rainfall_mm.append(avg_rain_mm)
            
        return city_name, [lat, lon, new_temp, old_hum, new_rain_prob, new_wind], new_rainfall_mm
            
    except Exception as e:
        print(f"Error fetching {city_name}: {e}")
        return city_name, data, None

def main():
    print(f"[*] Fetching 10-year Climate Models for {len(REFERENCE_CITIES)} cities...")
    new_reference_cities = {}
    city_precipitation = {}
    
    pairs = list(REFERENCE_CITIES.items())
    completed = 0
    
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(fetch_climate, p) for p in pairs]
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
