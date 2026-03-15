"""
Weather Forecasting API Server
Flask backend that serves ML model predictions and the frontend.
"""

import os
import json
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, 'ml_model', 'saved_models')
FRONTEND_DIR = os.path.join(PROJECT_DIR, 'frontend')

# ── Flask App ──────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# ── Load Models ────────────────────────────────────────────────────
print("🔄 Loading ML models...")
models = {
    'temperature': joblib.load(os.path.join(MODELS_DIR, 'temperature_model.joblib')),
    'humidity': joblib.load(os.path.join(MODELS_DIR, 'humidity_model.joblib')),
    'wind_speed': joblib.load(os.path.join(MODELS_DIR, 'wind_speed_model.joblib')),
    'rain': joblib.load(os.path.join(MODELS_DIR, 'rain_model.joblib')),
}
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))

with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'r') as f:
    model_metrics = json.load(f)

print("✅ All models loaded successfully!")


# ══════════════════════════════════════════════════════════════════════
# Meteorological Formulas (scientifically correct)
# ══════════════════════════════════════════════════════════════════════

def compute_feels_like(temp_c, humidity, wind_kmh):
    """
    Compute 'feels like' temperature using:
    - NWS Heat Index (Rothfusz regression) when T >= 27°C
    - Environment Canada Wind Chill when T <= 10°C and wind > 4.8 km/h
    - Actual temperature otherwise
    Capped at 55°C (real-world meteorological limit).
    """
    # ── Heat Index (when hot & humid) ──
    if temp_c >= 27 and humidity >= 40:
        # Convert to Fahrenheit for NWS formula
        T = temp_c * 9 / 5 + 32
        RH = humidity

        # Step 1: Simple formula first (NWS standard procedure)
        HI = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + RH * 0.094)

        # Step 2: If simple average with T exceeds 80°F, use Rothfusz
        if (HI + T) / 2 >= 80:
            HI = (
                -42.379
                + 2.04901523 * T
                + 10.14333127 * RH
                - 0.22475541 * T * RH
                - 0.00683783 * T * T
                - 0.05481717 * RH * RH
                + 0.00122874 * T * T * RH
                + 0.00085282 * T * RH * RH
                - 0.00000199 * T * T * RH * RH
            )

            # Adjustment for low humidity
            if RH < 13 and 80 < T < 112:
                adj = -((13 - RH) / 4) * ((17 - abs(T - 95)) / 17) ** 0.5
                HI += adj

            # Adjustment for high humidity
            if RH > 85 and 80 < T < 87:
                adj = ((RH - 85) / 10) * ((87 - T) / 5)
                HI += adj

        # Convert back to Celsius
        feels_c = (HI - 32) * 5 / 9

        # Cap at 55°C — real-world meteorological limit
        # Apply soft damping above 50°C to avoid extreme spikes
        if feels_c > 50:
            feels_c = 50 + (feels_c - 50) * 0.3
        feels_c = min(feels_c, 55.0)

        return round(feels_c, 1)

    # ── Wind Chill (when cold & windy) ──
    if temp_c <= 10 and wind_kmh > 4.8:
        WC = (
            13.12
            + 0.6215 * temp_c
            - 11.37 * (wind_kmh ** 0.16)
            + 0.3965 * temp_c * (wind_kmh ** 0.16)
        )
        return round(WC, 1)

    # ── Neutral zone: feels like actual temperature ──
    return round(temp_c, 1)


def compute_dew_point(temp_c, humidity):
    """
    Magnus approximation for dew point.
    Td = (b * α) / (a - α)   where  α = (a * T) / (b + T) + ln(RH/100)
    Constants: a = 17.27, b = 237.7 (valid for 0–60°C)
    Dew point MUST be <= temperature.
    """
    a = 17.27
    b = 237.7
    RH = max(1, min(100, humidity))  # clamp 1-100
    alpha = (a * temp_c) / (b + temp_c) + np.log(RH / 100.0)
    dew_pt = (b * alpha) / (a - alpha)
    # Physical constraint: dew point cannot exceed temperature
    return round(min(dew_pt, temp_c), 1)


def compute_uv_index(lat, month, rain_prob, humidity):
    """
    UV Index estimate based on:
    - Solar zenith angle (latitude + month/season)
    - Cloud cover proxy (rain probability)
    - Atmospheric moisture attenuation (humidity)
    """
    # Solar declination (approximate)
    day_of_year = (month - 1) * 30 + 15
    declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))

    # Solar elevation angle at noon
    solar_elevation = 90 - abs(lat - declination)
    solar_elevation = max(0, solar_elevation)

    # Base UV from solar elevation (clear sky max ~12)
    base_uv = (solar_elevation / 90) * 12

    # Cloud attenuation (rain = more clouds)
    cloud_factor = 1 - rain_prob * 0.7

    # Humidity attenuation (more moisture scatters UV slightly)
    humidity_factor = 1 - max(0, (humidity - 50)) / 500

    uv = base_uv * cloud_factor * humidity_factor
    return max(0, min(12, round(uv)))


def compute_visibility(humidity, rain_prob, wind_speed):
    """
    Visibility estimate in km based on:
    - Humidity (fog/haze when > 90%)
    - Rain probability (rain reduces visibility)
    - Wind (high wind = dust in arid areas)
    """
    base_vis = 20.0  # Clear day max ~20 km

    # Humidity reduction (fog formation > 85%)
    if humidity > 95:
        base_vis *= 0.15  # Dense fog
    elif humidity > 90:
        base_vis *= 0.35  # Fog
    elif humidity > 85:
        base_vis *= 0.55  # Mist
    elif humidity > 75:
        base_vis *= 0.75  # Haze
    else:
        base_vis *= 0.95

    # Rain reduction
    if rain_prob > 0.7:
        base_vis *= 0.3
    elif rain_prob > 0.5:
        base_vis *= 0.5
    elif rain_prob > 0.3:
        base_vis *= 0.7

    # Wind-blown dust (high wind + low humidity)
    if wind_speed > 40 and humidity < 30:
        base_vis *= 0.4

    return round(max(0.5, min(20, base_vis)), 1)


def compute_pressure(lat, month, temperature):
    """
    Sea-level pressure estimate based on:
    - Standard atmosphere (1013.25 hPa)
    - Latitude (higher pressure in subtropics ~30°)
    - Seasonal variation
    - Temperature influence
    """
    # Base pressure
    p = 1013.25

    # Subtropical high (30° lat) vs equatorial/polar low
    lat_effect = 5 * np.cos(np.radians((abs(lat) - 30) * 3))
    p += lat_effect

    # Seasonal variation (stronger at high latitudes)
    seasonal = 3 * np.cos(2 * np.pi * month / 12) * (abs(lat) / 90)
    if lat < 0:
        seasonal = -seasonal  # Opposite seasons in Southern Hemisphere
    p += seasonal

    # Temperature influence (hot = lower pressure)
    temp_effect = -(temperature - 15) * 0.15
    p += temp_effect

    # Add small noise for realism
    return round(p, 1)


def validate_weather(temp, humidity, wind, rain_prob, feels_like, dew_point):
    """
    Global sanity-check — detects physically impossible weather combinations.
    Returns dict of any issues found.
    """
    issues = []

    # 1. Dew point MUST be <= temperature
    if dew_point > temp + 0.5:
        issues.append(f"Dew point ({dew_point}°C) exceeds temperature ({temp}°C)")

    # 2. If hot + humid, feels-like MUST be >= temperature
    if temp >= 27 and humidity >= 40 and feels_like < temp - 1:
        issues.append(f"Feel-like ({feels_like}°C) too low for hot+humid conditions ({temp}°C, {humidity}%)")

    # 3. If cold + windy, feels-like MUST be <= temperature
    if temp <= 10 and wind > 5 and feels_like > temp + 1:
        issues.append(f"Feels-like ({feels_like}°C) too high for cold+windy ({temp}°C, {wind} km/h)")

    # 4. Humidity bounds
    if humidity < 0 or humidity > 100:
        issues.append(f"Humidity out of range: {humidity}%")

    # 5. Wind speed bounds
    if wind < 0 or wind > 200:
        issues.append(f"Wind speed out of range: {wind} km/h")

    # 6. Temperature bounds (Earth records: -89.2°C to 56.7°C)
    if temp < -90 or temp > 60:
        issues.append(f"Temperature out of Earth range: {temp}°C")

    return {'valid': len(issues) == 0, 'issues': issues}



def estimate_elevation(lat, lon):
    """Estimate elevation matching the training data generator."""
    # Himalayas (narrow band — NOT all of north India)
    if 27 <= lat <= 36 and 75 <= lon <= 100:
        if lat >= 32:
            return 2500
        elif lat >= 30:
            return 800
        else:
            return 250  # Plains: Delhi, Lucknow, Jaipur, etc.
    # Alps
    if 45 <= lat <= 48 and 5 <= lon <= 15:
        return 1500
    # Andes
    if -35 <= lat <= 10 and -80 <= lon <= -65:
        return 1500
    # Rockies
    if 35 <= lat <= 50 and -120 <= lon <= -105:
        return 1800
    # Tibet
    if 28 <= lat <= 38 and 78 <= lon <= 100:
        return 4000
    # East African highlands
    if -5 <= lat <= 5 and 30 <= lon <= 40:
        return 1200
    return 200  # Default lowland


def estimate_coast_distance(lat, lon):
    """Estimate coast distance matching the training data generator."""
    coast_points = [
        (19.1, 72.9), (13.1, 80.3), (51.5, -0.1), (40.7, -74.0),
        (34.1, -118.2), (35.7, 139.7), (-33.9, 151.2), (25.3, 55.3),
        (-22.9, -43.2), (1.3, 103.8), (-6.2, 106.8), (6.5, 3.4),
        (24.9, 67.0), (31.2, 121.5), (13.8, 100.5),
    ]
    min_dist = 9999
    for clat, clon in coast_points:
        d = np.sqrt((lat - clat)**2 + ((lon - clon) * np.cos(np.radians(lat)))**2)
        min_dist = min(min_dist, d)
    coast_km = min_dist * 111
    return min(coast_km, 1500)


def prepare_features(lat, lon, month):
    """Prepare the feature vector for prediction."""
    day_of_year = (month - 1) * 30 + 15  # mid-month approximation
    
    elevation = estimate_elevation(lat, lon)
    distance_to_coast = estimate_coast_distance(lat, lon)
    
    # Build feature vector matching training features
    features = np.array([[
        lat, lon, month, day_of_year, elevation, distance_to_coast,
        # Engineered features
        np.sin(2 * np.pi * month / 12),      # month_sin
        np.cos(2 * np.pi * month / 12),      # month_cos
        np.sin(2 * np.pi * day_of_year / 365),  # day_sin
        np.cos(2 * np.pi * day_of_year / 365),  # day_cos
        abs(lat),                              # abs_latitude
        1 if lat >= 0 else 0                   # is_northern
    ]])
    
    return scaler.transform(features)


# ── API Endpoints ──────────────────────────────────────────────────

@app.route('/api/predict', methods=['GET'])
def predict():
    """Predict weather for a given location and month."""
    try:
        lat = float(request.args.get('lat', 0))
        lon = float(request.args.get('lon', 0))
        month = int(request.args.get('month', datetime.now().month))
        
        # Validate inputs
        if not (-90 <= lat <= 90):
            return jsonify({'error': 'Latitude must be between -90 and 90'}), 400
        if not (-180 <= lon <= 180):
            return jsonify({'error': 'Longitude must be between -180 and 180'}), 400
        if not (1 <= month <= 12):
            return jsonify({'error': 'Month must be between 1 and 12'}), 400
        
        # Prepare features
        X = prepare_features(lat, lon, month)
        
        # Make predictions
        temperature = float(models['temperature'].predict(X)[0])
        humidity = float(models['humidity'].predict(X)[0])
        wind_speed = float(models['wind_speed'].predict(X)[0])
        rain_prediction = int(models['rain'].predict(X)[0])
        rain_probability = float(models['rain'].predict_proba(X)[0][1])
        
        # Determine weather condition
        if rain_probability > 0.7:
            condition = 'Heavy Rain'
            icon = '🌧️'
        elif rain_probability > 0.5:
            condition = 'Rain'
            icon = '🌦️'
        elif rain_probability > 0.3:
            condition = 'Partly Cloudy'
            icon = '⛅'
        elif temperature > 30:
            condition = 'Hot & Sunny'
            icon = '🔥'
        elif temperature > 20:
            condition = 'Sunny'
            icon = '☀️'
        elif temperature > 10:
            condition = 'Mild'
            icon = '🌤️'
        elif temperature > 0:
            condition = 'Cold'
            icon = '❄️'
        else:
            condition = 'Freezing'
            icon = '🥶'
        
        # Additional derived metrics — scientifically correct formulas
        feels_like = compute_feels_like(temperature, humidity, wind_speed)
        dew_point = compute_dew_point(temperature, humidity)
        uv_index = compute_uv_index(lat, month, rain_probability, humidity)
        visibility = compute_visibility(humidity, rain_probability, wind_speed)
        pressure = compute_pressure(lat, month, temperature)
        
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        result = {
            'location': {
                'latitude': round(lat, 4),
                'longitude': round(lon, 4),
            },
            'month': month,
            'month_name': month_names[month],
            'predictions': {
                'temperature': {
                    'value': round(temperature, 1),
                    'unit': '°C',
                    'feels_like': round(feels_like, 1)
                },
                'humidity': {
                    'value': round(max(5, min(100, humidity)), 1),
                    'unit': '%'
                },
                'wind_speed': {
                    'value': round(max(0, wind_speed), 1),
                    'unit': 'km/h'
                },
                'rain': {
                    'prediction': 'Rain' if rain_prediction else 'No Rain',
                    'probability': round(rain_probability * 100, 1),
                    'unit': '%'
                }
            },
            'condition': {
                'text': condition,
                'icon': icon
            },
            'additional': {
                'dew_point': round(dew_point, 1),
                'uv_index': uv_index,
                'visibility': visibility,
                'pressure': pressure
            },
            'validation': validate_weather(
                temperature, humidity, wind_speed,
                rain_probability, feels_like, dew_point
            )
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid parameter: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model metrics and training information."""
    return jsonify({
        **model_metrics,
        'api': {
            'version': '1.0.0',
            'endpoints': [
                {
                    'path': '/api/predict',
                    'method': 'GET',
                    'description': 'Get weather predictions for a location',
                    'parameters': [
                        {'name': 'lat', 'type': 'float', 'required': True, 'description': 'Latitude (-90 to 90)'},
                        {'name': 'lon', 'type': 'float', 'required': True, 'description': 'Longitude (-180 to 180)'},
                        {'name': 'month', 'type': 'int', 'required': False, 'description': 'Month (1-12, defaults to current)'}
                    ],
                    'example': '/api/predict?lat=28.6139&lon=77.2090&month=6'
                },
                {
                    'path': '/api/model-info',
                    'method': 'GET',
                    'description': 'Get model details and performance metrics',
                    'parameters': []
                }
            ]
        }
    })


@app.route('/api/forecast', methods=['GET'])
def forecast():
    """Get a 12-month forecast for a location."""
    try:
        lat = float(request.args.get('lat', 0))
        lon = float(request.args.get('lon', 0))
        
        if not (-90 <= lat <= 90):
            return jsonify({'error': 'Latitude must be between -90 and 90'}), 400
        if not (-180 <= lon <= 180):
            return jsonify({'error': 'Longitude must be between -180 and 180'}), 400
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        forecasts = []
        for m in range(1, 13):
            X = prepare_features(lat, lon, m)
            temp = float(models['temperature'].predict(X)[0])
            hum = float(models['humidity'].predict(X)[0])
            wind = float(models['wind_speed'].predict(X)[0])
            rain_prob = float(models['rain'].predict_proba(X)[0][1])
            
            forecasts.append({
                'month': m,
                'month_name': month_names[m - 1],
                'temperature': round(temp, 1),
                'humidity': round(max(5, min(100, hum)), 1),
                'wind_speed': round(max(0, wind), 1),
                'rain_probability': round(rain_prob * 100, 1)
            })
        
        return jsonify({
            'location': {'latitude': round(lat, 4), 'longitude': round(lon, 4)},
            'forecasts': forecasts
        })
    
    except ValueError as e:
        return jsonify({'error': f'Invalid parameter: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Forecast failed: {str(e)}'}), 500


# ── Serve Frontend ─────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)


# ── Main ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n🌦️  Weather Forecasting API")
    print("=" * 40)
    print("📍 Server:  http://localhost:5000")
    print("📡 API:     http://localhost:5000/api/predict?lat=28.6&lon=77.2&month=6")
    print("📊 Info:    http://localhost:5000/api/model-info")
    print("🌐 UI:      http://localhost:5000")
    print("=" * 40)
    app.run(debug=True, port=5000)
