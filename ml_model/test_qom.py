import requests
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

print("Qom, Iran (34.64, 51.69)")
print(f"{'Mon':>4} {'Temp':>7} {'Hum':>7} {'Wind':>7} {'Rain':>7}")
print("-" * 38)
for m in range(1, 13):
    r = requests.get('http://localhost:5000/api/predict',
                     params={'lat': 34.64, 'lon': 51.69, 'month': m})
    data = r.json()
    if 'error' in data:
        print(f"  ERROR: {data['error']}")
        break
    p = data['predictions']
    t = p['temperature']['value']
    h = p['humidity']['value']
    w = p['wind_speed']['value']
    rn = p['rain']['probability']
    print(f"{months[m-1]:>4} {t:>6.1f}C {h:>5.1f}% {w:>5.1f}kph {rn:>5.1f}%")

# Real Qom data for comparison
print("\nReal Qom data (approx):")
print("  Jan: 5C, 50%, rain ~25mm")
print("  Apr: 20C, 30%, rain ~15mm")
print("  Jul: 37C, 15%, rain ~0mm")
print("  Oct: 20C, 30%, rain ~5mm")
