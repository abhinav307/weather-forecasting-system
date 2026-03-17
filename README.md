# 🌦️ WeatherAI — AI-Powered Weather Forecasting

An intelligent weather prediction system using **Random Forest ML models** trained on global climate data from **38 reference cities**.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikit-learn)

**[🌐 Live Demo Here](https://weather-forecasting-system-o3lu.onrender.com)**

## ✨ Features

- 🌍 **Global Coverage** — Predicts weather for any latitude/longitude on Earth
- 🌡️ **4 ML Models** — Temperature, Humidity, Wind Speed, Rain Classification
- 📊 **Annual Trends** — 12-month charts and monthly data tables
- 🗺️ **Interactive Maps** — Leaflet.js with click-to-predict and search
- 📡 **REST API** — Full API with live testers in the docs page
- 🔬 **Correct Meteorology** — NWS Heat Index, Magnus dew point, Wind Chill formulas
- 🎨 **Modern UI** — Glassmorphism design, dark/light themes, smooth animations

## 🚀 Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data & train models (first time only)
python build.py

# 3. Run the server
python backend/app.py
```

Open **http://localhost:5000** in your browser.

## 📁 Project Structure

```
weather-forecasting-system/
├── backend/
│   └── app.py              # Flask API server
├── frontend/
│   ├── index.html           # Landing page
│   ├── forecast.html        # Weather forecast page
│   ├── annual.html          # Annual trends page
│   ├── api-docs.html        # API documentation
│   ├── model-info.html      # Model metrics
│   ├── styles.css           # Global styles
│   ├── shared.js            # Shared utilities
│   ├── forecast.js          # Forecast page logic
│   ├── annual.js            # Annual view logic
│   ├── api-docs.js          # API docs logic
│   └── model-info.js        # Model info logic
├── ml_model/
│   ├── generate_data.py     # Synthetic data generator (38 cities)
│   ├── train_model.py       # Model training pipeline
│   └── saved_models/        # Trained models (auto-generated)
├── build.py                 # Build script for deployment
├── render.yaml              # Render.com deployment config
├── requirements.txt         # Python dependencies
└── README.md
```

## 🌐 Deploy to Render (Free)

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your repo — Render reads `render.yaml` automatically
4. Click **Deploy** — models train during the build step (~5 min)

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict?lat=28.6&lon=77.2&month=6` | GET | Single prediction |
| `/api/forecast?lat=28.6&lon=77.2` | GET | 12-month forecast |
| `/api/model-info` | GET | Model metrics & details |

## 🧪 Model Performance

| Model | R² Score | MAE |
|-------|----------|------|
| Temperature | 0.91 | 2.1°C |
| Humidity | 0.64 | 5.0% |
| Wind Speed | 0.31 | 2.6 km/h |
| Rain (F1) | 0.29 | — |

## 📜 License

MIT
