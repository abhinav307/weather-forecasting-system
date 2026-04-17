import sys

# Read the file
with open('backend/app.py', 'r', encoding='utf-8') as f:
    code = f.read()

injection = '''import requests
import datetime
from collections import defaultdict

EMPIRICAL_CACHE = {}

def get_real_rainfall(lat, lon):
    cache_key = f"{round(lat, 1)}_{round(lon, 1)}"
    if cache_key in EMPIRICAL_CACHE:
        return EMPIRICAL_CACHE[cache_key]

    try:
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': '2018-01-01',
            'end_date': '2022-12-31',
            'daily': 'precipitation_sum',
            'timezone': 'auto'
        }
        r = requests.get('https://archive-api.open-meteo.com/v1/archive', params=params, timeout=2.5)
        if r.status_code != 200:
            return None
            
        data = r.json().get('daily', {})
        dates = data.get('time', [])
        prcp = data.get('precipitation_sum', [])
        if not prcp: return None
        
        monthly_sums = defaultdict(list)
        current_month = None
        current_sum = 0
        
        for i, d in enumerate(dates):
            if prcp[i] is None: continue
            
            dt = datetime.datetime.strptime(d, '%Y-%m-%d')
            m_key = f"{dt.year}-{dt.month}"
            
            if current_month is None: current_month = m_key
            
            if m_key != current_month:
                _, m = current_month.split('-')
                monthly_sums[int(m)].append(current_sum)
                current_month = m_key
                current_sum = 0
                
            current_sum += prcp[i]
            
        if current_month:
            _, m = current_month.split('-')
            monthly_sums[int(m)].append(current_sum)
            
        averages = {}
        for m in range(1, 13):
            vals = monthly_sums.get(m, [0])
            averages[m] = round(sum(vals)/max(1, len(vals)), 1)
            
        EMPIRICAL_CACHE[cache_key] = averages
        return averages
    except Exception as e:
        print("Empirical fetch failed:", e)
        return None

'''

if 'get_real_rainfall' not in code:
    code = code.replace('import math\n', 'import math\n' + injection)
    with open('backend/app.py', 'w', encoding='utf-8') as f:
        f.write(code)
print("Injected successfully.")
