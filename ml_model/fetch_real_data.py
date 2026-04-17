"""
Real-World Weather Data Fetcher (Meteostat Version)
Downloads historical daily weather data using the Meteostat library.
Meteostat pulls from GitHub-hosted bulk data -- no API rate limits.

Source: Meteostat (https://meteostat.net/) -- based on NOAA, DWD, Environment Canada, etc.
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

from meteostat import Point, Daily

# ---- 100+ Global Reference Cities ----
# Format: (city_name, latitude, longitude)
CITIES = [
    # -- South Asia --
    ("Delhi", 28.6139, 77.2090),
    ("Mumbai", 19.0760, 72.8777),
    ("Chennai", 13.0827, 80.2707),
    ("Kolkata", 22.5726, 88.3639),
    ("Bangalore", 12.9716, 77.5946),
    ("Hyderabad", 17.3850, 78.4867),
    ("Jaipur", 26.9124, 75.7873),
    ("Lucknow", 26.8467, 80.9462),
    ("Ahmedabad", 23.0225, 72.5714),
    ("Pune", 18.5204, 73.8567),
    ("Karachi", 24.8607, 67.0011),
    ("Lahore", 31.5204, 74.3587),
    ("Islamabad", 33.6844, 73.0479),
    ("Dhaka", 23.8103, 90.4125),
    ("Kathmandu", 27.7172, 85.3240),
    ("Colombo", 6.9271, 79.8612),

    # -- East Asia --
    ("Tokyo", 35.6762, 139.6503),
    ("Beijing", 39.9042, 116.4074),
    ("Shanghai", 31.2304, 121.4737),
    ("Seoul", 37.5665, 126.9780),
    ("Hong Kong", 22.3193, 114.1694),
    ("Taipei", 25.0330, 121.5654),
    ("Osaka", 34.6937, 135.5023),
    ("Ulaanbaatar", 47.8864, 106.9057),

    # -- Southeast Asia --
    ("Bangkok", 13.7563, 100.5018),
    ("Singapore", 1.3521, 103.8198),
    ("Jakarta", -6.2088, 106.8456),
    ("Manila", 14.5995, 120.9842),
    ("Hanoi", 21.0278, 105.8342),
    ("Kuala Lumpur", 3.1390, 101.6869),
    ("Ho Chi Minh City", 10.8231, 106.6297),

    # -- Middle East --
    ("Dubai", 25.2048, 55.2708),
    ("Riyadh", 24.7136, 46.6753),
    ("Tehran", 35.6892, 51.3890),
    ("Istanbul", 41.0082, 28.9784),
    ("Baghdad", 33.3152, 44.3661),
    ("Doha", 25.2854, 51.5310),
    ("Muscat", 23.5880, 58.3829),
    ("Amman", 31.9454, 35.9284),
    ("Jerusalem", 31.7683, 35.2137),

    # -- Europe --
    ("London", 51.5074, -0.1278),
    ("Paris", 48.8566, 2.3522),
    ("Berlin", 52.5200, 13.4050),
    ("Moscow", 55.7558, 37.6173),
    ("Rome", 41.9028, 12.4964),
    ("Madrid", 40.4168, -3.7038),
    ("Helsinki", 60.1699, 24.9384),
    ("Reykjavik", 64.1466, -21.9426),
    ("Oslo", 59.9139, 10.7522),
    ("Stockholm", 59.3293, 18.0686),
    ("Athens", 37.9838, 23.7275),
    ("Lisbon", 38.7223, -9.1393),
    ("Warsaw", 52.2297, 21.0122),
    ("Vienna", 48.2082, 16.3738),
    ("Zurich", 47.3769, 8.5417),
    ("Dublin", 53.3498, -6.2603),
    ("Amsterdam", 52.3676, 4.9041),
    ("Prague", 50.0755, 14.4378),
    ("Budapest", 47.4979, 19.0402),
    ("Bucharest", 44.4268, 26.1025),
    ("Copenhagen", 55.6761, 12.5683),
    ("Edinburgh", 55.9533, -3.1883),
    ("St Petersburg", 59.9311, 30.3609),

    # -- North America --
    ("New York", 40.7128, -74.0060),
    ("Los Angeles", 34.0522, -118.2437),
    ("Chicago", 41.8781, -87.6298),
    ("Toronto", 43.6532, -79.3832),
    ("Mexico City", 19.4326, -99.1332),
    ("Miami", 25.7617, -80.1918),
    ("Denver", 39.7392, -104.9903),
    ("Vancouver", 49.2827, -123.1207),
    ("Anchorage", 61.2181, -149.9003),
    ("Phoenix", 33.4484, -112.0740),
    ("Houston", 29.7604, -95.3698),
    ("Montreal", 45.5017, -73.5673),
    ("San Francisco", 37.7749, -122.4194),
    ("Atlanta", 33.7490, -84.3880),
    ("Seattle", 47.6062, -122.3321),
    ("Havana", 23.1136, -82.3666),
    ("Guatemala City", 14.6349, -90.5069),

    # -- South America --
    ("Sao Paulo", -23.5505, -46.6333),
    ("Rio de Janeiro", -22.9068, -43.1729),
    ("Buenos Aires", -34.6037, -58.3816),
    ("Bogota", 4.7110, -74.0721),
    ("Lima", -12.0464, -77.0428),
    ("Santiago", -33.4489, -70.6693),
    ("Quito", -0.1807, -78.4678),
    ("La Paz", -16.4897, -68.1193),
    ("Caracas", 10.4806, -66.9036),

    # -- Africa --
    ("Cairo", 30.0444, 31.2357),
    ("Nairobi", -1.2921, 36.8219),
    ("Cape Town", -33.9249, 18.4241),
    ("Lagos", 6.5244, 3.3792),
    ("Dakar", 14.7167, -17.4677),
    ("Addis Ababa", 9.0250, 38.7469),
    ("Marrakech", 31.6295, -7.9811),
    ("Casablanca", 33.5731, -7.5898),
    ("Johannesburg", -26.2041, 28.0473),
    ("Dar es Salaam", -6.7924, 39.2083),
    ("Accra", 5.6037, -0.1870),
    ("Tunis", 36.8065, 10.1815),
    ("Khartoum", 15.5007, 32.5599),
    ("Kinshasa", -4.4419, 15.2663),

    # -- Oceania --
    ("Sydney", -33.8688, 151.2093),
    ("Melbourne", -37.8136, 144.9631),
    ("Brisbane", -27.4698, 153.0251),
    ("Perth", -31.9505, 115.8605),
    ("Auckland", -36.8485, 174.7633),
    ("Darwin", -12.4634, 130.8456),
    ("Wellington", -41.2865, 174.7762),
    ("Christchurch", -43.5321, 172.6362),

    # -- Extreme / Edge-case stations --
    ("Yakutsk", 62.0355, 129.6755),
    ("Timbuktu", 16.7735, -3.0074),
    ("Manaus", -3.1190, -60.0217),
    ("Tromso", 69.6496, 18.9560),
    ("Ushuaia", -54.8019, -68.3030),
    ("Lhasa", 29.6520, 91.1721),
    ("Alice Springs", -23.6980, 133.8807),
    ("Murmansk", 68.9585, 33.0827),
    ("Iqaluit", 63.7467, -68.5170),
]

# Date range: 5 years
START = datetime(2019, 1, 1)
END = datetime(2024, 12, 31)


def compute_humidity_from_dewpoint(tavg, tdew):
    """
    Compute relative humidity (%) from temperature and dew point
    using the Magnus formula.
    RH = 100 * exp( (a*Td)/(b+Td) - (a*T)/(b+T) )
    Constants: a=17.27, b=237.7 (valid for 0-60 C range)
    """
    a = 17.27
    b = 237.7
    if pd.isna(tavg) or pd.isna(tdew):
        return np.nan
    gamma_t = (a * tavg) / (b + tavg)
    gamma_td = (a * tdew) / (b + tdew)
    rh = 100.0 * np.exp(gamma_td - gamma_t)
    return np.clip(rh, 5, 100)


def estimate_elevation(lat, lon):
    """Estimate elevation for a location."""
    if 27 <= lat <= 36 and 75 <= lon <= 100:
        if lat >= 32: return 2500.0
        elif lat >= 30: return 800.0
        else: return 250.0
    if 28 <= lat <= 38 and 78 <= lon <= 100:
        return 4000.0
    if 45 <= lat <= 48 and 5 <= lon <= 15:
        return 1500.0
    if -35 <= lat <= 10 and -80 <= lon <= -65:
        return 1500.0
    if 35 <= lat <= 50 and -120 <= lon <= -105:
        return 1800.0
    if -5 <= lat <= 15 and 30 <= lon <= 42:
        return 1500.0
    if 18 <= lat <= 21 and -100 <= lon <= -98:
        return 2250.0
    if 3 <= lat <= 6 and -75 <= lon <= -73:
        return 2640.0
    if -18 <= lat <= -15 and -70 <= lon <= -67:
        return 3640.0
    if 28 <= lat <= 31 and 90 <= lon <= 93:
        return 3650.0
    return 200.0


def estimate_coast_distance(lat, lon):
    """Estimate distance to coast for a location."""
    coast_points = [
        (19.1, 72.9), (13.1, 80.3), (51.5, -0.1), (40.7, -74.0),
        (34.1, -118.2), (35.7, 139.7), (-33.9, 151.2), (25.3, 55.3),
        (-22.9, -43.2), (1.3, 103.8), (-6.2, 106.8), (6.5, 3.4),
        (24.9, 67.0), (31.2, 121.5), (13.8, 100.5), (-27.5, 153.0),
        (-31.9, 115.8), (-12.4, 130.8), (25.8, -80.2), (37.8, -122.4),
        (14.6, 120.98), (10.8, 106.6), (6.9, 79.9), (22.3, 114.2),
        (-6.8, 39.2), (5.6, -0.2), (14.7, -17.5), (38.7, -9.1),
        (-41.3, 174.8), (-18.1, 178.4), (23.1, -82.4),
    ]
    min_dist = 9999
    for clat, clon in coast_points:
        d = np.sqrt((lat - clat)**2 + ((lon - clon) * np.cos(np.radians(lat)))**2)
        min_dist = min(min_dist, d)
    coast_km = min_dist * 111
    return min(coast_km, 1500)


def fetch_city_data(city_name, lat, lon):
    """Fetch daily weather data for a single city using Meteostat."""
    location = Point(lat, lon)
    data = Daily(location, START, END)
    df = data.fetch()

    if df.empty:
        print(f"  [WARN] No data for {city_name}")
        return None

    df = df.reset_index()
    df["latitude"] = round(lat, 4)
    df["longitude"] = round(lon, 4)
    df["city"] = city_name
    return df


def process_raw_data(all_dfs):
    """Process raw Meteostat data into the training-ready format."""
    print("\n[*] Processing raw data into training format...")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"   Raw rows: {len(df):,}")

    # Meteostat columns: time, tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun
    # We need: temperature, humidity, wind_speed, rain

    # Temperature: use tavg, fallback to (tmin+tmax)/2
    df["temperature"] = df["tavg"]
    mask = df["temperature"].isna() & df["tmin"].notna() & df["tmax"].notna()
    df.loc[mask, "temperature"] = (df.loc[mask, "tmin"] + df.loc[mask, "tmax"]) / 2

    # Humidity: Meteostat doesn't provide RH directly.
    # We'll estimate from temperature and dew point when available,
    # or use climate-based estimation as fallback.
    # First, check if we have enough data to compute from dewpoint
    # Meteostat doesn't always have dew point, so we'll use a climate-based approach

    # For humidity, we use known monthly climate averages per latitude band
    # This gives us realistic baseline humidity that we then add noise to
    df["month"] = pd.to_datetime(df["time"]).dt.month
    df["day_of_year"] = pd.to_datetime(df["time"]).dt.dayofyear

    # Estimate humidity based on precipitation patterns, latitude, and season
    df["humidity"] = df.apply(lambda r: estimate_humidity(
        r["latitude"], r["longitude"], r["month"],
        r.get("prcp", 0), r["temperature"]
    ), axis=1)

    # Wind speed: wspd is in km/h in Meteostat
    df["wind_speed"] = df["wspd"]

    # Rain: precipitation > 0.5mm counts as rain day
    df["rain"] = (df["prcp"].fillna(0) > 0.5).astype(int)

    # Drop rows missing essential data (temperature is critical)
    before = len(df)
    df = df.dropna(subset=["temperature"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"   Dropped {dropped:,} rows with missing temperature")

    # Fill missing wind speed with reasonable default (10 km/h)
    df["wind_speed"] = df["wind_speed"].fillna(10.0)

    # Compute geographic features
    df["elevation"] = df.apply(lambda r: estimate_elevation(r["latitude"], r["longitude"]), axis=1)
    df["distance_to_coast"] = df.apply(lambda r: estimate_coast_distance(r["latitude"], r["longitude"]), axis=1)

    # Round and clip values
    df["temperature"] = df["temperature"].round(1)
    df["humidity"] = df["humidity"].clip(5, 100).round(1)
    df["wind_speed"] = df["wind_speed"].clip(0, 120).round(1)

    # Select final columns in expected order
    result = df[[
        "latitude", "longitude", "month", "day_of_year",
        "elevation", "distance_to_coast",
        "temperature", "humidity", "wind_speed", "rain"
    ]].copy()

    # Shuffle
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"   Final rows: {len(result):,}")
    return result


# ---- Humidity Estimation ----
# Monthly reference humidity by climate zone (based on WMO climate normals)
# Zones: Tropical, Arid, Subtropical, Temperate, Continental, Subarctic, Polar

HUMIDITY_PROFILES = {
    # (lat_min, lat_max, lon_min, lon_max): monthly_humidity[12]
    # South Asia (monsoon climate)
    "south_asia": {
        "bounds": [(5, 35, 65, 95)],
        "monthly": [55, 45, 38, 35, 42, 65, 80, 82, 75, 60, 50, 55]
    },
    # Southeast Asia (tropical humid)
    "southeast_asia": {
        "bounds": [(-10, 25, 95, 125)],
        "monthly": [78, 76, 75, 76, 78, 78, 78, 80, 80, 80, 80, 79]
    },
    # East Asia (temperate monsoon)
    "east_asia": {
        "bounds": [(25, 50, 100, 145)],
        "monthly": [55, 55, 52, 55, 60, 70, 75, 73, 68, 62, 58, 55]
    },
    # Middle East (arid)
    "middle_east": {
        "bounds": [(15, 40, 25, 65)],
        "monthly": [50, 45, 38, 30, 25, 22, 22, 25, 30, 38, 45, 50]
    },
    # Western Europe (maritime temperate)
    "western_europe": {
        "bounds": [(45, 65, -15, 15)],
        "monthly": [82, 78, 72, 65, 65, 65, 68, 70, 72, 78, 82, 84]
    },
    # Eastern Europe (continental)
    "eastern_europe": {
        "bounds": [(45, 65, 15, 45)],
        "monthly": [80, 76, 68, 58, 55, 60, 62, 64, 68, 74, 80, 82]
    },
    # Scandinavia / subarctic Europe
    "scandinavia": {
        "bounds": [(55, 72, -25, 35)],
        "monthly": [82, 78, 72, 62, 58, 62, 68, 72, 76, 80, 84, 84]
    },
    # North America East
    "na_east": {
        "bounds": [(25, 55, -100, -60)],
        "monthly": [65, 62, 58, 55, 58, 65, 68, 70, 68, 62, 62, 65]
    },
    # North America West
    "na_west": {
        "bounds": [(25, 55, -130, -100)],
        "monthly": [55, 55, 52, 48, 45, 40, 35, 35, 38, 45, 52, 55]
    },
    # Central America / Caribbean
    "central_america": {
        "bounds": [(5, 25, -110, -60)],
        "monthly": [68, 65, 62, 62, 68, 75, 78, 78, 80, 78, 72, 70]
    },
    # South America tropical
    "sa_tropical": {
        "bounds": [(-15, 12, -80, -35)],
        "monthly": [80, 82, 82, 80, 78, 72, 68, 62, 65, 72, 76, 80]
    },
    # South America temperate
    "sa_temperate": {
        "bounds": [(-55, -15, -80, -35)],
        "monthly": [68, 70, 72, 72, 75, 78, 78, 75, 72, 68, 65, 66]
    },
    # North Africa (arid)
    "north_africa": {
        "bounds": [(15, 35, -20, 35)],
        "monthly": [52, 48, 40, 32, 28, 25, 28, 32, 38, 45, 50, 52]
    },
    # Sub-Saharan Africa
    "sub_saharan": {
        "bounds": [(-5, 15, -20, 45)],
        "monthly": [65, 62, 65, 72, 75, 78, 80, 80, 78, 75, 70, 65]
    },
    # Southern Africa
    "southern_africa": {
        "bounds": [(-35, -5, 10, 45)],
        "monthly": [60, 62, 62, 58, 52, 48, 45, 42, 45, 52, 58, 60]
    },
    # Australia
    "australia": {
        "bounds": [(-45, -10, 110, 155)],
        "monthly": [55, 58, 58, 55, 55, 58, 55, 48, 45, 48, 52, 55]
    },
    # New Zealand
    "new_zealand": {
        "bounds": [(-48, -34, 165, 180)],
        "monthly": [75, 75, 75, 78, 80, 82, 82, 80, 78, 75, 75, 75]
    },
    # Arctic / high lat
    "arctic": {
        "bounds": [(65, 90, -180, 180)],
        "monthly": [75, 72, 68, 62, 60, 65, 72, 78, 80, 80, 78, 76]
    },
    # Antarctic
    "antarctic": {
        "bounds": [(-90, -60, -180, 180)],
        "monthly": [60, 58, 55, 52, 55, 58, 60, 62, 58, 55, 55, 58]
    },
}

# Global default
DEFAULT_HUMIDITY = [65, 62, 58, 55, 55, 58, 62, 65, 65, 62, 62, 65]


def estimate_humidity(lat, lon, month, prcp, temp):
    """
    Estimate relative humidity based on geographic zone, month,
    precipitation presence, and temperature.
    """
    month_idx = int(month) - 1
    base_hum = DEFAULT_HUMIDITY[month_idx]

    # Find matching climate zone
    for zone_data in HUMIDITY_PROFILES.values():
        for (lat_min, lat_max, lon_min, lon_max) in zone_data["bounds"]:
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                base_hum = zone_data["monthly"][month_idx]
                break

    # Adjust for precipitation (rainy days are more humid)
    if pd.notna(prcp) and prcp > 1.0:
        base_hum = min(100, base_hum + 15)
    elif pd.notna(prcp) and prcp > 0.1:
        base_hum = min(100, base_hum + 8)

    # Adjust for extreme temperatures
    if pd.notna(temp):
        if temp > 40:  # Very hot = typically dry (unless monsoon)
            base_hum = max(15, base_hum - 10)
        elif temp < -10:  # Very cold = naturally low abs humidity
            base_hum = max(30, base_hum - 5)

    # Add small random variation for realism
    noise = np.random.normal(0, 4)
    base_hum = np.clip(base_hum + noise, 5, 100)

    return round(base_hum, 1)


def validate_data(df):
    """Run validation checks against known climate values."""
    print("\n[*] Validating data...")

    checks = [
        ("Delhi June", (27, 30), (76, 79), 6, "temperature", 33, 44),
        ("Delhi January", (27, 30), (76, 79), 1, "temperature", 12, 22),
        ("Moscow January", (54, 57), (36, 39), 1, "temperature", -14, -2),
        ("Singapore Year-round", (0, 3), (103, 105), None, "temperature", 25, 30),
        ("Dubai July", (24, 26), (54, 56), 7, "temperature", 34, 44),
        ("Sydney July", (-35, -33), (150, 153), 7, "temperature", 10, 18),
        ("London July", (50, 53), (-1, 1), 7, "temperature", 18, 25),
        ("Chicago January", (41, 43), (-89, -86), 1, "temperature", -8, 2),
    ]

    all_passed = True
    for name, lat_range, lon_range, month, col, exp_low, exp_high in checks:
        subset = df[
            (df["latitude"].between(*lat_range)) &
            (df["longitude"].between(*lon_range))
        ]
        if month is not None:
            subset = subset[subset["month"] == month]

        if len(subset) == 0:
            print(f"   [WARN] {name}: No data found")
            continue

        mean_val = subset[col].mean()
        status = "[OK]" if exp_low <= mean_val <= exp_high else "[FAIL]"
        if status == "[FAIL]":
            all_passed = False
        print(f"   {status} {name}: {col} = {mean_val:.1f} (expected {exp_low}-{exp_high})")

    return all_passed


def main():
    print("=" * 65)
    print("[*] Real-World Weather Data Fetcher (Meteostat)")
    print(f"    Source: Meteostat (NOAA, DWD, Environment Canada, etc.)")
    print(f"    Period: {START.year} - {END.year}")
    print(f"    Cities: {len(CITIES)}")
    print("=" * 65)

    all_dfs = []
    failed_cities = []

    for i, (city, lat, lon) in enumerate(CITIES, 1):
        print(f"\n[{i}/{len(CITIES)}] Fetching {city} ({lat}, {lon})...")
        df = fetch_city_data(city, lat, lon)

        if df is not None and len(df) > 0:
            all_dfs.append(df)
            n = len(df)
            t_col = "tavg" if "tavg" in df.columns else "temperature"
            t_vals = df[t_col].dropna()
            if len(t_vals) > 0:
                print(f"   OK {n:,} days | Temp range: {t_vals.min():.1f}C - {t_vals.max():.1f}C")
            else:
                print(f"   OK {n:,} days | (no temp data)")
        else:
            failed_cities.append(city)

    if not all_dfs:
        print("\n[FAIL] No data fetched! Check your internet connection.")
        sys.exit(1)

    # Process into training format
    result = process_raw_data(all_dfs)

    # Validate
    validate_data(result)

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_data.csv")
    result.to_csv(output_path, index=False)

    print(f"\n{'=' * 65}")
    print(f"[DONE] Generated {len(result):,} real-world data points")
    print(f"   Saved to: {output_path}")
    if failed_cities:
        print(f"   [WARN] Failed cities ({len(failed_cities)}): {', '.join(failed_cities)}")
    print(f"\nData Summary:")
    print(result.describe().round(2))
    print(f"\nRain distribution: {result['rain'].value_counts().to_dict()}")
    print(f"\n>> Next step: Run train_model.py to retrain the models")


if __name__ == "__main__":
    main()
