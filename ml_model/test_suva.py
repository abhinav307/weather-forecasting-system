import requests
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
lat, lon = -18.14, 178.44
print("Suva, Fiji (-18.14, 178.44)")
print(f"{'Mon':>4} {'Temp':>7} {'Hum':>7} {'Wind':>7} {'Rain%':>7} {'RainMM':>8}")
print("-" * 48)
for m in range(1, 13):
    r = requests.get('http://localhost:5000/api/predict', params={'lat': lat, 'lon': lon, 'month': m})
    d = r.json()
    p = d['predictions']
    t = p['temperature']['value']
    h = p['humidity']['value']
    w = p['wind_speed']['value']
    rn = p['rain']['probability']
    mm = p['rain'].get('rainfall_mm', 0)
    print(f"{months[m-1]:>4} {t:>6.1f}C {h:>5.1f}% {w:>5.1f}kph {rn:>5.1f}% {mm:>6.1f}mm")

print()
print("REAL Suva data:")
print("  Jan: 27C, 83%, rain ~350mm (wet season)")
print("  Apr: 26C, 82%, rain ~330mm")
print("  Jul: 23C, 78%, rain ~120mm (dry season)")
print("  Oct: 25C, 78%, rain ~170mm")
print("  Annual: ~3000mm total")
