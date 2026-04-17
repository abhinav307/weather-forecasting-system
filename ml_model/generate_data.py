"""
Weather Data Generator — V3 (Monthly Seasonal Variation)
Generates synthetic weather data calibrated against real-world climate averages.
All variables (temp, humidity, rain, wind) now use monthly arrays for proper
seasonal variation globally.
"""

import numpy as np
import pandas as pd
import os
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, 'saved_models')
PRECIP_PATH = os.path.join(MODELS_DIR, 'city_precipitation.json')

def load_city_precip():
    if os.path.exists(PRECIP_PATH):
        with open(PRECIP_PATH, 'r') as f:
            return json.load(f)
    return {}

city_precip = load_city_precip()

np.random.seed(42)

NUM_SAMPLES = 30000

# ── Reference climate data for calibration ──────────────────────────
# Format: (lat, lon, monthly_temps[12], monthly_humidity[12], monthly_rain_prob[12], monthly_wind[12])
# Temperatures = Average Daytime Highs (°C)
# Humidity = Monthly average relative humidity (%)
# Rain Prob = Monthly probability of rain on any given day (0-1)
# Wind = Monthly average wind speed (km/h)
REFERENCE_CITIES = {
    # ── Asia ──
    'Delhi': (28.6, 77.2,
              [21, 24, 30, 36, 40, 39, 35, 34, 34, 33, 28, 23],
              [40, 35, 28, 22, 25, 45, 75, 80, 70, 45, 35, 40],
              [0.05, 0.05, 0.08, 0.05, 0.10, 0.30, 0.60, 0.55, 0.35, 0.10, 0.03, 0.03],
              [10, 12, 14, 16, 18, 22, 20, 16, 14, 10, 8, 8]),
    'Mumbai': (19.1, 72.9,
               [31, 31, 33, 33, 34, 32, 30, 30, 30, 32, 33, 32],
               [60, 58, 62, 65, 68, 82, 88, 87, 83, 72, 62, 60],
               [0.02, 0.02, 0.02, 0.02, 0.08, 0.70, 0.85, 0.75, 0.50, 0.15, 0.05, 0.02],
               [12, 12, 12, 14, 16, 22, 24, 20, 16, 12, 10, 10]),
    'Chennai': (13.1, 80.3,
                [29, 31, 33, 35, 38, 38, 36, 35, 34, 32, 30, 29],
                [70, 65, 62, 65, 60, 55, 60, 65, 70, 78, 82, 78],
                [0.10, 0.05, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.45, 0.55, 0.35],
                [14, 14, 12, 12, 16, 22, 22, 18, 14, 12, 14, 14]),
    'Kolkata': (22.6, 88.4,
                [26, 29, 34, 36, 36, 34, 33, 33, 33, 32, 30, 27],
                [55, 45, 42, 50, 65, 78, 85, 85, 82, 72, 60, 55],
                [0.05, 0.08, 0.08, 0.12, 0.25, 0.50, 0.65, 0.60, 0.45, 0.20, 0.05, 0.03],
                [10, 14, 16, 20, 22, 22, 20, 18, 16, 12, 10, 8]),
    'Karachi': (24.9, 67.0,
                [26, 28, 32, 34, 35, 35, 33, 32, 32, 34, 31, 27],
                [50, 45, 40, 38, 45, 60, 72, 75, 68, 48, 42, 48],
                [0.03, 0.03, 0.03, 0.02, 0.02, 0.08, 0.25, 0.20, 0.08, 0.02, 0.02, 0.02],
                [12, 14, 16, 20, 24, 26, 24, 20, 16, 12, 10, 10]),
    'Lahore': (31.5, 74.3,
               [20, 23, 28, 35, 39, 40, 36, 35, 35, 33, 27, 21],
               [55, 50, 40, 30, 25, 35, 70, 75, 60, 42, 45, 55],
               [0.08, 0.10, 0.10, 0.08, 0.08, 0.15, 0.50, 0.45, 0.20, 0.05, 0.03, 0.05],
               [8, 10, 12, 14, 18, 22, 20, 16, 12, 8, 6, 6]),
    'Dhaka': (23.8, 90.4,
              [25, 28, 32, 34, 33, 32, 32, 32, 32, 31, 29, 26],
              [55, 45, 45, 58, 72, 82, 88, 87, 84, 76, 65, 58],
              [0.05, 0.08, 0.10, 0.20, 0.40, 0.60, 0.70, 0.65, 0.50, 0.25, 0.08, 0.03],
              [8, 12, 16, 20, 22, 24, 22, 20, 16, 12, 8, 6]),
    'Kathmandu': (27.7, 85.3,
                  [19, 21, 25, 28, 30, 29, 29, 29, 28, 26, 23, 20],
                  [65, 55, 45, 42, 52, 72, 85, 84, 78, 65, 58, 62],
                  [0.05, 0.08, 0.10, 0.15, 0.25, 0.50, 0.70, 0.65, 0.40, 0.10, 0.03, 0.03],
                  [6, 8, 10, 12, 14, 16, 14, 12, 10, 8, 6, 6]),
    'Jaipur': (26.9, 75.8,
               [22, 26, 32, 37, 40, 40, 34, 32, 34, 33, 29, 24],
               [35, 28, 22, 18, 20, 38, 68, 72, 55, 30, 25, 32],
               [0.03, 0.03, 0.02, 0.02, 0.05, 0.15, 0.45, 0.40, 0.15, 0.03, 0.02, 0.02],
               [8, 10, 12, 14, 18, 22, 18, 14, 12, 8, 6, 6]),
    'Lucknow': (26.8, 81.0,
                [22, 26, 32, 38, 40, 39, 33, 33, 33, 32, 28, 23],
                [55, 45, 30, 22, 25, 42, 78, 82, 72, 50, 42, 52],
                [0.05, 0.05, 0.05, 0.03, 0.08, 0.20, 0.55, 0.50, 0.30, 0.05, 0.02, 0.03],
                [8, 10, 12, 14, 18, 22, 18, 14, 12, 8, 6, 6]),
    'Tokyo': (35.7, 139.7,
              [10, 10, 14, 19, 23, 26, 30, 31, 27, 22, 17, 12],
              [50, 50, 55, 60, 65, 75, 78, 72, 70, 65, 58, 52],
              [0.15, 0.18, 0.30, 0.30, 0.30, 0.40, 0.35, 0.25, 0.35, 0.30, 0.20, 0.15],
              [14, 16, 16, 16, 14, 12, 14, 14, 14, 14, 14, 14]),
    'Beijing': (39.9, 116.4,
                [2, 5, 12, 20, 26, 30, 31, 30, 26, 19, 10, 4],
                [40, 38, 30, 32, 38, 52, 72, 75, 58, 48, 45, 42],
                [0.05, 0.08, 0.08, 0.12, 0.15, 0.25, 0.45, 0.40, 0.20, 0.10, 0.08, 0.05],
                [12, 16, 20, 22, 20, 16, 12, 10, 12, 14, 14, 12]),
    'Shanghai': (31.2, 121.5,
                 [8, 10, 14, 20, 25, 28, 32, 32, 28, 23, 17, 11],
                 [70, 68, 70, 68, 68, 78, 78, 75, 72, 68, 68, 68],
                 [0.25, 0.28, 0.35, 0.30, 0.30, 0.40, 0.35, 0.30, 0.28, 0.20, 0.18, 0.20],
                 [14, 16, 16, 16, 14, 14, 16, 14, 16, 14, 14, 14]),
    'Bangkok': (13.8, 100.5,
                [33, 34, 35, 35, 34, 33, 33, 33, 33, 33, 33, 32],
                [55, 58, 60, 65, 72, 75, 75, 78, 80, 78, 68, 58],
                [0.05, 0.08, 0.10, 0.18, 0.35, 0.40, 0.42, 0.48, 0.50, 0.40, 0.15, 0.05],
                [12, 14, 16, 16, 14, 14, 14, 14, 12, 10, 10, 10]),
    'Singapore': (1.3, 103.8,
                  [31, 31, 32, 32, 32, 31, 31, 31, 31, 32, 31, 31],
                  [82, 78, 80, 82, 82, 80, 80, 80, 80, 82, 85, 84],
                  [0.40, 0.30, 0.35, 0.40, 0.40, 0.35, 0.35, 0.38, 0.40, 0.45, 0.50, 0.50],
                  [10, 12, 12, 10, 10, 12, 12, 12, 10, 10, 10, 10]),
    'Jakarta': (-6.2, 106.8,
                [31, 31, 32, 33, 33, 33, 33, 33, 33, 33, 32, 32],
                [85, 82, 80, 78, 72, 68, 65, 62, 65, 72, 78, 82],
                [0.60, 0.55, 0.50, 0.40, 0.25, 0.15, 0.12, 0.10, 0.12, 0.25, 0.40, 0.55],
                [10, 10, 10, 10, 12, 14, 14, 16, 16, 12, 10, 10]),
    'Dubai': (25.3, 55.3,
              [24, 26, 29, 33, 38, 40, 41, 41, 39, 35, 31, 26],
              [62, 60, 55, 45, 40, 42, 48, 50, 52, 50, 55, 60],
              [0.05, 0.05, 0.08, 0.03, 0.01, 0.00, 0.00, 0.00, 0.00, 0.02, 0.03, 0.05],
              [14, 16, 18, 18, 16, 16, 16, 14, 14, 12, 12, 14]),
    'Riyadh': (24.7, 46.7,
               [20, 23, 28, 33, 39, 42, 43, 43, 40, 35, 28, 22],
               [40, 30, 25, 20, 12, 8, 8, 8, 10, 15, 25, 38],
               [0.05, 0.05, 0.08, 0.08, 0.02, 0.00, 0.00, 0.00, 0.00, 0.02, 0.03, 0.05],
               [12, 14, 16, 18, 16, 14, 14, 12, 12, 10, 10, 10]),

    # ── Europe ──
    'London': (51.5, -0.1,
               [8, 9, 11, 14, 17, 20, 23, 23, 20, 15, 11, 9],
               [82, 78, 72, 65, 65, 65, 68, 70, 72, 78, 82, 84],
               [0.48, 0.38, 0.35, 0.35, 0.32, 0.30, 0.30, 0.32, 0.35, 0.42, 0.48, 0.50],
               [16, 16, 14, 14, 12, 12, 12, 12, 14, 16, 16, 16]),
    'Paris': (48.9, 2.3,
              [7, 8, 12, 16, 20, 23, 25, 25, 21, 16, 11, 8],
              [85, 78, 72, 65, 65, 62, 60, 62, 68, 78, 82, 85],
              [0.35, 0.30, 0.30, 0.28, 0.30, 0.25, 0.22, 0.22, 0.25, 0.30, 0.35, 0.38],
              [16, 16, 16, 14, 14, 12, 12, 12, 12, 14, 14, 16]),
    'Berlin': (52.5, 13.4,
               [3, 5, 9, 14, 19, 22, 24, 24, 19, 14, 8, 4],
               [82, 78, 68, 60, 58, 62, 62, 64, 68, 75, 82, 84],
               [0.35, 0.30, 0.28, 0.25, 0.30, 0.35, 0.35, 0.32, 0.28, 0.28, 0.32, 0.35],
               [16, 16, 16, 14, 14, 14, 14, 14, 14, 14, 16, 16]),
    'Moscow': (55.8, 37.6,
               [-4, -3, 3, 11, 19, 23, 25, 23, 16, 8, 1, -3],
               [82, 78, 68, 58, 52, 58, 62, 65, 70, 78, 82, 85],
               [0.35, 0.28, 0.25, 0.25, 0.30, 0.35, 0.38, 0.35, 0.32, 0.35, 0.38, 0.38],
               [18, 18, 16, 14, 14, 12, 12, 12, 14, 16, 18, 18]),
    'Helsinki': (60.2, 25.0,
                 [-1, -2, 2, 8, 15, 19, 22, 20, 14, 8, 3, 0],
                 [85, 82, 72, 62, 55, 58, 62, 68, 75, 82, 85, 88],
                 [0.38, 0.30, 0.28, 0.25, 0.25, 0.30, 0.32, 0.35, 0.35, 0.38, 0.40, 0.40],
                 [18, 18, 16, 14, 14, 12, 12, 14, 16, 18, 18, 18]),
    'Reykjavik': (64.1, -21.9,
                  [3, 3, 3, 6, 9, 12, 14, 13, 10, 7, 4, 3],
                  [78, 76, 74, 72, 68, 72, 76, 78, 76, 78, 78, 80],
                  [0.50, 0.45, 0.45, 0.40, 0.35, 0.35, 0.35, 0.38, 0.42, 0.48, 0.50, 0.52],
                  [24, 22, 22, 20, 18, 16, 16, 18, 20, 22, 24, 24]),

    # ── North America ──
    'New York': (40.7, -74.0,
                 [4, 5, 10, 16, 22, 27, 29, 28, 24, 18, 12, 6],
                 [60, 58, 55, 55, 60, 65, 68, 70, 68, 62, 62, 62],
                 [0.30, 0.28, 0.32, 0.32, 0.32, 0.30, 0.30, 0.28, 0.25, 0.25, 0.28, 0.30],
                 [20, 20, 20, 18, 16, 14, 14, 14, 16, 18, 18, 20]),
    'Los Angeles': (34.1, -118.2,
                    [20, 20, 21, 23, 24, 26, 29, 29, 29, 26, 23, 20],
                    [55, 58, 60, 58, 62, 62, 58, 55, 55, 55, 50, 52],
                    [0.18, 0.18, 0.15, 0.08, 0.05, 0.02, 0.01, 0.02, 0.03, 0.08, 0.12, 0.15],
                    [12, 14, 14, 16, 14, 14, 12, 12, 12, 12, 12, 12]),
    'Chicago': (41.9, -87.6,
                [-1, 2, 8, 15, 21, 27, 29, 28, 24, 17, 9, 2],
                [68, 65, 60, 55, 55, 60, 62, 65, 62, 58, 65, 70],
                [0.30, 0.25, 0.30, 0.32, 0.32, 0.28, 0.28, 0.25, 0.25, 0.25, 0.30, 0.30],
                [20, 20, 22, 22, 18, 16, 14, 14, 16, 18, 20, 20]),
    'Toronto': (43.7, -79.4,
                [-1, 0, 5, 12, 19, 24, 27, 26, 22, 15, 8, 2],
                [70, 65, 60, 55, 55, 60, 62, 65, 65, 62, 68, 72],
                [0.35, 0.30, 0.30, 0.32, 0.28, 0.25, 0.25, 0.25, 0.28, 0.30, 0.35, 0.35],
                [20, 18, 18, 18, 16, 14, 14, 14, 16, 18, 18, 20]),
    'Anchorage': (61.2, -150.0,
                  [-5, -3, 1, 7, 13, 17, 19, 18, 13, 5, -1, -4],
                  [68, 65, 58, 50, 48, 55, 62, 68, 72, 72, 70, 70],
                  [0.25, 0.22, 0.18, 0.15, 0.15, 0.20, 0.30, 0.38, 0.40, 0.35, 0.28, 0.25],
                  [14, 14, 12, 14, 14, 14, 12, 12, 14, 16, 16, 14]),
    'Mexico City': (19.4, -99.1,
                    [22, 24, 26, 27, 28, 26, 24, 24, 24, 23, 22, 21],
                    [38, 32, 28, 30, 42, 62, 68, 68, 70, 58, 45, 40],
                    [0.05, 0.05, 0.05, 0.10, 0.30, 0.55, 0.60, 0.58, 0.55, 0.30, 0.10, 0.05],
                    [12, 14, 16, 16, 14, 12, 12, 12, 10, 12, 12, 12]),

    # ── South America ──
    'São Paulo': (-23.5, -46.6,
                  [27, 28, 27, 25, 23, 22, 22, 23, 24, 25, 26, 27],
                  [78, 76, 76, 72, 70, 68, 62, 58, 62, 70, 72, 76],
                  [0.55, 0.50, 0.42, 0.28, 0.18, 0.15, 0.12, 0.12, 0.20, 0.30, 0.40, 0.50],
                  [12, 12, 12, 12, 14, 14, 16, 16, 16, 14, 12, 12]),
    'Rio': (-22.9, -43.2,
            [31, 31, 30, 28, 26, 25, 25, 26, 26, 27, 28, 30],
            [78, 76, 78, 76, 72, 68, 65, 62, 68, 75, 76, 78],
            [0.40, 0.35, 0.35, 0.25, 0.18, 0.12, 0.10, 0.10, 0.18, 0.28, 0.35, 0.40],
            [14, 14, 14, 14, 16, 16, 18, 18, 16, 14, 14, 14]),
    'Bogota': (4.7, -74.0,
               [19, 19, 19, 19, 19, 19, 18, 19, 19, 19, 19, 19],
               [72, 68, 72, 78, 78, 72, 65, 62, 68, 78, 80, 75],
               [0.20, 0.22, 0.35, 0.48, 0.45, 0.18, 0.12, 0.15, 0.25, 0.45, 0.42, 0.28],
               [10, 12, 12, 10, 10, 14, 16, 16, 14, 10, 10, 10]),

    # ── Africa ──
    'Cairo': (30.0, 31.2,
              [19, 21, 24, 28, 32, 35, 35, 35, 33, 30, 25, 21],
              [55, 50, 42, 35, 32, 30, 35, 38, 42, 48, 52, 55],
              [0.08, 0.05, 0.05, 0.03, 0.02, 0.00, 0.00, 0.00, 0.00, 0.02, 0.05, 0.08],
              [14, 16, 18, 18, 16, 16, 14, 14, 14, 14, 12, 14]),
    'Nairobi': (-1.3, 36.8,
                [26, 26, 26, 25, 24, 23, 22, 23, 25, 26, 24, 24],
                [55, 50, 55, 70, 72, 60, 55, 52, 50, 58, 72, 65],
                [0.15, 0.12, 0.30, 0.55, 0.45, 0.12, 0.08, 0.10, 0.10, 0.30, 0.50, 0.25],
                [12, 14, 14, 12, 10, 14, 16, 18, 16, 12, 10, 12]),
    'Cape Town': (-33.9, 18.4,
                  [26, 27, 25, 23, 20, 18, 18, 18, 19, 21, 23, 25],
                  [48, 50, 52, 60, 68, 72, 75, 72, 65, 55, 50, 48],
                  [0.05, 0.05, 0.08, 0.18, 0.30, 0.38, 0.42, 0.35, 0.25, 0.15, 0.08, 0.05],
                  [22, 22, 20, 16, 14, 14, 16, 18, 20, 22, 22, 22]),
    'Lagos': (6.5, 3.4,
              [32, 33, 33, 32, 31, 29, 28, 28, 29, 31, 32, 32],
              [72, 72, 75, 78, 80, 85, 85, 82, 82, 80, 78, 72],
              [0.08, 0.12, 0.22, 0.30, 0.45, 0.62, 0.55, 0.40, 0.52, 0.48, 0.18, 0.08],
              [10, 12, 14, 14, 14, 16, 18, 18, 16, 12, 10, 10]),

    # ── Oceania ──
    'Sydney': (-33.9, 151.2,
               [26, 26, 25, 23, 20, 17, 16, 18, 20, 22, 24, 26],
               [62, 65, 62, 60, 58, 58, 55, 48, 48, 52, 58, 60],
               [0.30, 0.35, 0.35, 0.28, 0.25, 0.22, 0.18, 0.15, 0.18, 0.22, 0.28, 0.28],
               [16, 16, 14, 14, 16, 18, 18, 20, 18, 16, 16, 16]),
    'Melbourne': (-37.8, 145.0,
                  [26, 26, 24, 21, 17, 14, 14, 15, 17, 20, 22, 24],
                  [48, 50, 52, 58, 62, 65, 65, 60, 55, 52, 50, 48],
                  [0.18, 0.15, 0.18, 0.20, 0.22, 0.22, 0.22, 0.22, 0.25, 0.25, 0.22, 0.20],
                  [16, 16, 14, 16, 18, 18, 20, 20, 20, 18, 16, 16]),
    'Brisbane': (-27.5, 153.0,
                 [30, 30, 29, 27, 24, 22, 21, 22, 24, 26, 28, 29],
                 [62, 65, 62, 58, 55, 52, 48, 42, 42, 48, 55, 60],
                 [0.35, 0.38, 0.30, 0.22, 0.15, 0.12, 0.10, 0.08, 0.10, 0.18, 0.25, 0.30],
                 [16, 16, 14, 14, 14, 16, 16, 18, 18, 18, 16, 16]),
    'Perth': (-31.9, 115.8,
              [32, 32, 30, 26, 22, 19, 18, 19, 20, 23, 27, 29],
              [38, 38, 42, 50, 58, 65, 68, 65, 58, 48, 42, 38],
              [0.05, 0.05, 0.08, 0.15, 0.30, 0.42, 0.48, 0.40, 0.28, 0.18, 0.10, 0.05],
              [18, 18, 16, 16, 18, 20, 22, 22, 20, 18, 18, 18]),
    'Darwin': (-12.4, 130.8,
               [32, 31, 32, 33, 32, 31, 31, 31, 33, 33, 33, 33],
               [78, 80, 78, 65, 50, 42, 38, 38, 42, 52, 62, 72],
               [0.60, 0.62, 0.55, 0.25, 0.05, 0.02, 0.02, 0.02, 0.05, 0.15, 0.35, 0.50],
               [12, 12, 10, 10, 12, 14, 16, 18, 18, 16, 14, 12]),
    'Auckland': (-36.8, 174.7,
                 [24, 24, 23, 20, 18, 16, 15, 15, 16, 18, 20, 22],
                 [72, 72, 72, 75, 78, 80, 80, 78, 75, 72, 72, 72],
                 [0.25, 0.22, 0.25, 0.28, 0.35, 0.38, 0.40, 0.38, 0.32, 0.30, 0.28, 0.25],
                 [16, 16, 16, 16, 18, 20, 20, 20, 20, 18, 16, 16]),
}


def haversine_dist(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth in km."""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def get_interpolated_value(lat, lon, value_idx, month_idx=None):
    """
    Generic inverse-distance-weighted interpolation from reference cities.
    value_idx: which index in the reference tuple to interpolate
    month_idx: if the target value is a list (e.g. monthly data)
    """
    ref_list = list(REFERENCE_CITIES.values())
    weights = []
    values = []

    for ref_city in ref_list:
        rlat, rlon = ref_city[0], ref_city[1]
        dist = haversine_dist(lat, lon, rlat, rlon)
        dist = max(dist, 10.0)  # Min 10km to avoid division by zero

        # Power parameter p=4 for sharper regional boundaries
        w = 1.0 / (dist ** 4)
        weights.append(w)

        target_val = ref_city[value_idx]
        if month_idx is not None:
            target_val = target_val[month_idx]

        values.append(target_val)

    total_w = sum(weights)
    return sum(w * v for w, v in zip(weights, values)) / total_w


def get_rainfall_for_location(lat, lon, month_idx):
    if not city_precip: return 0.0
    min_dist = float('inf')
    best_rain = 0.0
    m_str = str(month_idx + 1)
    
    for k, monthly in city_precip.items():
        base_lat, base_lon = map(float, k.split(","))
        dist = (lat - base_lat)**2 + (lon - base_lon)**2
        if dist < min_dist:
            min_dist = dist
            if m_str in monthly:
                best_rain = monthly[m_str].get('rainfall_mm', 0.0)
    return best_rain


def get_temperature_for_location(lat, lon, month_idx):
    return get_interpolated_value(lat, lon, 2, month_idx)

def get_humidity_for_location(lat, lon, month_idx):
    return get_interpolated_value(lat, lon, 3, month_idx)

def get_rain_prob_for_location(lat, lon, month_idx):
    return get_interpolated_value(lat, lon, 4, month_idx)

def get_wind_for_location(lat, lon, month_idx):
    return get_interpolated_value(lat, lon, 5, month_idx)


def estimate_elevation(lat, lon):
    """Rough but more accurate elevation estimation."""
    if 27 <= lat <= 36 and 75 <= lon <= 100:
        if lat >= 32: return np.random.uniform(1500, 4000)
        elif lat >= 30: return np.random.uniform(300, 1500)
        else: return np.random.uniform(150, 400)
    if 45 <= lat <= 48 and 5 <= lon <= 15:
        return np.random.uniform(800, 2500)
    if -35 <= lat <= 10 and -80 <= lon <= -65:
        return np.random.uniform(500, 3000)
    if 35 <= lat <= 50 and -120 <= lon <= -105:
        return np.random.uniform(1000, 3000)
    if 28 <= lat <= 38 and 78 <= lon <= 100:
        return np.random.uniform(3000, 5000)
    if -5 <= lat <= 5 and 30 <= lon <= 40:
        return np.random.uniform(500, 2000)
    return np.random.uniform(0, 500)


def estimate_coast_distance(lat, lon):
    """Rough distance-to-coast heuristic."""
    coast_points = [
        (19.1, 72.9), (13.1, 80.3), (51.5, -0.1), (40.7, -74.0),
        (34.1, -118.2), (35.7, 139.7), (-33.9, 151.2), (25.3, 55.3),
        (-22.9, -43.2), (1.3, 103.8), (-6.2, 106.8), (6.5, 3.4),
        (24.9, 67.0), (31.2, 121.5), (13.8, 100.5), (-27.5, 153.0),
        (-31.9, 115.8), (-12.4, 130.8),
    ]
    min_dist = 9999
    for clat, clon in coast_points:
        d = np.sqrt((lat - clat)**2 + ((lon - clon) * np.cos(np.radians(lat)))**2)
        min_dist = min(min_dist, d)
    coast_km = min_dist * 111
    return min(coast_km, 1500)


def generate_weather_data(n=NUM_SAMPLES):
    """Generate realistic synthetic weather data with proper seasonal variation."""

    n_city = int(n * 0.6)
    n_random = n - n_city

    cities = list(REFERENCE_CITIES.values())

    # Samples near reference cities (with jitter)
    city_lats = []
    city_lons = []
    for i in range(n_city):
        city = cities[i % len(cities)]
        jitter_lat = np.random.normal(0, 3)
        jitter_lon = np.random.normal(0, 3)
        lat = np.clip(city[0] + jitter_lat, -65, 70)
        lon = np.clip(city[1] + jitter_lon, -180, 180)
        city_lats.append(lat)
        city_lons.append(lon)

    # Random worldwide samples
    random_lats = np.random.uniform(-60, 70, n_random)
    random_lons = np.random.uniform(-180, 180, n_random)

    all_lats = np.concatenate([city_lats, random_lats])
    all_lons = np.concatenate([city_lons, random_lons])

    months = np.random.randint(1, 13, n)
    day_of_year = (months - 1) * 30 + np.random.randint(1, 31, n)

    temperatures = np.zeros(n)
    humidities = np.zeros(n)
    wind_speeds = np.zeros(n)
    rain_labels = np.zeros(n, dtype=int)
    rainfall_mms = np.zeros(n)
    elevations = np.zeros(n)
    coast_distances = np.zeros(n)

    print("🌍 Generating weather data (this may take a moment)...")

    for i in range(n):
        lat = all_lats[i]
        lon = all_lons[i]
        month = months[i]
        month_idx = month - 1

        # ── Temperature ──
        base_temp = get_temperature_for_location(lat, lon, month_idx)
        elev = estimate_elevation(lat, lon)
        elevations[i] = elev
        elev_correction = -6.5 * max(0, (elev - 200)) / 1000
        coast_dist = estimate_coast_distance(lat, lon)
        coast_distances[i] = coast_dist
        temp = base_temp + elev_correction + np.random.normal(0, 2.0)
        temperatures[i] = round(temp, 1)

        # ── Humidity (monthly, with noise) ──
        base_hum = get_humidity_for_location(lat, lon, month_idx)
        hum = base_hum + np.random.normal(0, 5)
        humidities[i] = round(np.clip(hum, 8, 100), 1)

        # ── Wind Speed (monthly, with noise) ──
        base_wind = get_wind_for_location(lat, lon, month_idx)
        # Add coastal bonus
        if coast_dist < 200:
            base_wind += 3
        # Add elevation bonus
        base_wind += elev / 1000 * 1.5
        wind = base_wind + np.random.normal(0, 2.5)
        wind_speeds[i] = round(max(1, min(60, wind)), 1)

        # ── Rain Probability (0 or 1 label) ──
        rain_prob = get_rain_prob_for_location(lat, lon, month_idx)
        rain_prob = np.clip(rain_prob, 0.01, 0.95)
        is_rain = 1 if np.random.random() < rain_prob else 0
        rain_labels[i] = is_rain

        # ── Rainfall amounts (mm) ──
        base_rainfall = get_rainfall_for_location(lat, lon, month_idx)
        rainfall = base_rainfall * np.random.uniform(0.6, 1.4)
        if is_rain == 0:
            rainfall = 0.0
        elif rainfall == 0.0 and is_rain == 1:
            rainfall = np.random.uniform(1.0, 5.0)  # Trace amounts if it rained unexpectedly
        rainfall_mms[i] = round(max(0, rainfall), 1)

    df = pd.DataFrame({
        'latitude': np.round(all_lats, 4),
        'longitude': np.round(all_lons, 4),
        'month': months,
        'day_of_year': day_of_year,
        'elevation': np.round(elevations, 1),
        'distance_to_coast': np.round(coast_distances, 1),
        'temperature': temperatures,
        'humidity': humidities,
        'wind_speed': wind_speeds,
        'rain': rain_labels,
        'rainfall_mm': rainfall_mms
    })

    return df


if __name__ == '__main__':
    df = generate_weather_data()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weather_data.csv')
    df.to_csv(output_path, index=False)

    print(f"✅ Generated {len(df)} samples → {output_path}")
    print(f"\n📊 Data Summary:")
    print(df.describe().round(2))
    print(f"\n🌧️  Rain distribution: {df['rain'].value_counts().to_dict()}")

    # Validate with known cities
    print("\n🔍 Validation — Delhi (June, lat~28.6, lon~77.2):")
    delhi = df[(df['latitude'].between(27, 30)) & (df['longitude'].between(76, 79)) & (df['month'] == 6)]
    if len(delhi) > 0:
        print(f"   Temp: {delhi['temperature'].mean():.1f}°C (should be ~39°C)")
        print(f"   Humidity: {delhi['humidity'].mean():.1f}% (should be ~45%)")
        print(f"   Wind: {delhi['wind_speed'].mean():.1f} km/h (should vary seasonally)")

    print("\n🔍 Validation — Delhi (January):")
    delhi_jan = df[(df['latitude'].between(27, 30)) & (df['longitude'].between(76, 79)) & (df['month'] == 1)]
    if len(delhi_jan) > 0:
        print(f"   Temp: {delhi_jan['temperature'].mean():.1f}°C (should be ~21°C)")
        print(f"   Humidity: {delhi_jan['humidity'].mean():.1f}% (should be ~40%)")

    print("\n🔍 Validation — Brisbane (January, lat~-27.5, lon~153.0):")
    bris = df[(df['latitude'].between(-29, -26)) & (df['longitude'].between(151, 155)) & (df['month'] == 1)]
    if len(bris) > 0:
        print(f"   Temp: {bris['temperature'].mean():.1f}°C (should be ~30°C)")
        print(f"   Humidity: {bris['humidity'].mean():.1f}% (should be ~62%)")

    print("\n🔍 Validation — Brisbane (July):")
    bris_jul = df[(df['latitude'].between(-29, -26)) & (df['longitude'].between(151, 155)) & (df['month'] == 7)]
    if len(bris_jul) > 0:
        print(f"   Temp: {bris_jul['temperature'].mean():.1f}°C (should be ~21°C)")
        print(f"   Humidity: {bris_jul['humidity'].mean():.1f}% (should be ~48%)")
