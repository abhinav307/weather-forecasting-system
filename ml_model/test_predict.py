import requests
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

cities = [
    ("Cuba (Havana)", 23.0131, -80.8329),
    ("Pune, India", 18.5204, 73.8567),
    ("Qom, Iran", 34.64, 51.69),
    ("Delhi", 28.6139, 77.209),
]

for city, lat, lon in cities:
    print(f"\n{'='*60}")
    print(f" {city} ({lat}, {lon})")
    print(f"{'='*60}")
    print(f"{'Mon':>4} {'Temp':>7} {'Hum':>7} {'Wind':>7} {'Rain%':>7} {'RainMM':>8}")
    print("-" * 48)
    for m in range(1, 13):
        r = requests.get('http://localhost:5000/api/predict',
                         params={'lat': lat, 'lon': lon, 'month': m})
        data = r.json()
        if 'error' in data:
            print(f"  ERROR: {data['error']}")
            break
        p = data['predictions']
        t = p['temperature']['value']
        h = p['humidity']['value']
        w = p['wind_speed']['value']
        rn = p['rain']['probability']
        mm = p['rain'].get('rainfall_mm', 0)
        print(f"{months[m-1]:>4} {t:>6.1f}C {h:>5.1f}% {w:>5.1f}kph {rn:>5.1f}% {mm:>6.1f}mm")
