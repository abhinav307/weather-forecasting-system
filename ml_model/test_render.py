import requests, time

print("Testing Render deployment...")
print("=" * 50)

cities = [
    ("Mumbai", 19.07, 72.87, 7, "~28C, ~85%"),
    ("Delhi", 28.61, 77.21, 7, "~31C, ~77%"),
    ("Pune", 18.52, 73.85, 7, "~25C, ~85%"),
    ("London", 51.51, -0.13, 7, "~17C, ~74%"),
]

for name, lat, lon, month, expected in cities:
    try:
        r = requests.get(
            f'https://weather-forecasting-system-o3lu.onrender.com/api/predict',
            params={'lat': lat, 'lon': lon, 'month': month},
            timeout=120
        )
        d = r.json()
        if 'error' in d:
            print(f"{name}: ERROR - {d['error']}")
            continue
        p = d['predictions']
        t = p['temperature']['value']
        h = p['humidity']['value']
        w = p['wind_speed']['value']
        rn = p['rain']['probability']
        mm = p['rain'].get('rainfall_mm', 0)
        print(f"{name:>8}: temp={t:.1f}C  hum={h:.1f}%  wind={w:.1f}kph  rain={rn:.1f}%  rainfall={mm:.1f}mm")
        print(f"          Expected: {expected}")
    except Exception as e:
        print(f"{name}: FAILED - {e}")
