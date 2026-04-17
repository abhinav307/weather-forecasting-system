"""
Weather Model Trainer v2 — Hybrid Architecture
================================================
Instead of a single global RandomForest, this uses:

1. Climate Normals Lookup Table: Pre-computed monthly averages per training city
2. KNN Interpolation: At prediction time, finds K nearest cities and
   uses inverse-distance-weighted interpolation of their REAL monthly data
3. XGBoost Residual Model: Learns to adjust the KNN baseline based on
   local geography (elevation, coast distance, etc.)

This hybrid approach gives:
- Accurate baselines from real nearby city data
- Proper seasonal patterns from actual observations
- Geographic refinements from the ML model
"""

import numpy as np
import pandas as pd
import json
import os
import joblib
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'weather_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

TARGETS = ['temperature', 'humidity', 'wind_speed', 'rain']


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))


def build_climate_normals(df):
    """
    Compute monthly climate normals per unique city coordinate.
    Returns a dict: {(lat, lon): {month: {temp, humidity, wind, rain_prob, elevation, coast_dist}}}
    """
    print("[*] Building climate normals lookup table...")
    normals = {}

    cities = df.groupby(['latitude', 'longitude'])
    for (lat, lon), city_df in cities:
        monthly = {}
        for m in range(1, 13):
            month_data = city_df[city_df['month'] == m]
            if len(month_data) == 0:
                continue
            monthly[m] = {
                'temperature': round(float(month_data['temperature'].mean()), 2),
                'humidity': round(float(month_data['humidity'].mean()), 2),
                'wind_speed': round(float(month_data['wind_speed'].mean()), 2),
                'rain_prob': round(float(month_data['rain'].mean()) * 100, 2),
                'elevation': float(month_data['elevation'].iloc[0]),
                'distance_to_coast': float(month_data['distance_to_coast'].iloc[0]),
            }
        if monthly:
            normals[(round(lat, 4), round(lon, 4))] = monthly

    print(f"   Built normals for {len(normals)} cities, 12 months each")
    return normals


def knn_interpolate(normals, lat, lon, month, elevation, k=5):
    """
    Find K nearest cities and compute inverse-distance-weighted climate for given month.
    Applies elevation-based temperature lapse rate correction.
    """
    distances = []
    for (clat, clon), monthly in normals.items():
        if month not in monthly:
            continue
        d = haversine(lat, lon, clat, clon)
        d = max(d, 1.0)  # Avoid division by zero
        distances.append((d, clat, clon, monthly[month]))

    distances.sort(key=lambda x: x[0])
    nearest = distances[:k]

    if not nearest:
        return {'temperature': 20, 'humidity': 50, 'wind_speed': 10, 'rain_prob': 20}

    # Inverse-distance weighting
    weights = [1.0 / (d ** 2) for d, _, _, _ in nearest]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    result = {}
    for key in ['temperature', 'humidity', 'wind_speed', 'rain_prob']:
        result[key] = sum(w * data[key] for w, (_, _, _, data) in zip(weights, nearest))

    # Temperature lapse rate correction: -6.5C per 1000m elevation difference
    weighted_elev = sum(w * data['elevation'] for w, (_, _, _, data) in zip(weights, nearest))
    elev_diff = elevation - weighted_elev
    result['temperature'] -= 6.5 * (elev_diff / 1000.0)

    return result


def add_knn_features(df, normals, k=5):
    """
    For each row in the dataset, compute KNN-interpolated values from
    OTHER cities (leave-one-out to avoid data leakage) and add as features.
    """
    print("[*] Computing KNN interpolation features (this may take a minute)...")

    knn_temp = np.zeros(len(df))
    knn_hum = np.zeros(len(df))
    knn_wind = np.zeros(len(df))
    knn_rain = np.zeros(len(df))
    knn_rainfall_mm = np.zeros(len(df))

    # Group by city for efficiency
    city_groups = df.groupby(['latitude', 'longitude']).groups

    for (lat, lon), indices in city_groups.items():
        # Leave-one-out: exclude this city from normals
        key = (round(lat, 4), round(lon, 4))
        filtered_normals = {k_: v for k_, v in normals.items() if k_ != key}

        # Get elevation for this city
        elev = df.loc[indices[0], 'elevation']

        # Compute KNN for each month
        month_cache = {}
        for idx in indices:
            m = int(df.loc[idx, 'month'])
            if m not in month_cache:
                month_cache[m] = knn_interpolate(filtered_normals, lat, lon, m, elev, k)

            result = month_cache[m]
            knn_temp[idx] = result['temperature']
            knn_hum[idx] = result['humidity']
            knn_wind[idx] = result['wind_speed']
            knn_rain[idx] = result['rain_prob']
            knn_rainfall_mm[idx] = result.get('rainfall_mm', 0.0)

    df['knn_temp'] = knn_temp
    df['knn_hum'] = knn_hum
    df['knn_wind'] = knn_wind
    df['knn_rain'] = knn_rain
    df['knn_rainfall_mm'] = knn_rainfall_mm

    print("   KNN features added")
    return df


def add_features(df):
    """Engineer features for the XGBoost refinement model."""
    df = df.copy()

    # Cyclical encoding of month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Cyclical encoding of day_of_year
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Geographic
    df['abs_latitude'] = np.abs(df['latitude'])
    df['is_northern'] = (df['latitude'] >= 0).astype(int)

    # Interaction features
    df['lat_x_month_sin'] = df['latitude'] * df['month_sin']
    df['lat_x_month_cos'] = df['latitude'] * df['month_cos']
    df['lon_x_month_sin'] = df['longitude'] * df['month_sin']
    df['lon_x_month_cos'] = df['longitude'] * df['month_cos']

    # Climate zone
    df['lat_band'] = pd.cut(
        df['abs_latitude'],
        bins=[0, 10, 23.5, 35, 55, 90],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(float)

    df['month_f'] = df['month'].astype(float)

    return df


def get_feature_columns():
    """Full list of features for the XGBoost model."""
    return [
        # Geographic
        'latitude', 'longitude', 'month', 'day_of_year',
        'elevation', 'distance_to_coast',
        # Cyclical
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        # Derived
        'abs_latitude', 'is_northern',
        'lat_x_month_sin', 'lat_x_month_cos',
        'lon_x_month_sin', 'lon_x_month_cos',
        'lat_band', 'month_f',
        # KNN interpolation features (KEY — brings real nearby data into model)
        'knn_temp', 'knn_hum', 'knn_wind', 'knn_rain', 'knn_rainfall_mm',
    ]


def train_xgb_regressor(X_train, X_test, y_train, y_test, target_name):
    """Train XGBoost regressor."""
    print(f"\n{'='*50}")
    print(f"[*] Training XGBoost: {target_name}")
    print(f"{'='*50}")

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R2:   {r2:.4f}")

    metrics = {
        'type': 'regression',
        'target': target_name,
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2_score': round(r2, 4),
        'algorithm': 'XGBoost',
        'n_estimators': 300,
        'max_depth': 8,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }

    return model, metrics


def train_xgb_classifier(X_train, X_test, y_train, y_test):
    """Train XGBoost classifier for rain prediction."""
    print(f"\n{'='*50}")
    print(f"[*] Training XGBoost: rain (classification)")
    print(f"{'='*50}")

    # Compute scale_pos_weight for imbalanced classes
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_weight,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric='logloss',
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Rain', 'Rain'])}")

    metrics = {
        'type': 'classification',
        'target': 'rain',
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'algorithm': 'XGBoost',
        'n_estimators': 300,
        'max_depth': 8,
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    }

    return model, metrics


def main():
    print("[*] Weather Model Training Pipeline v2 (Hybrid Architecture)")
    print("=" * 65)

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"[*] Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Step 1: Build climate normals lookup
    normals = build_climate_normals(df)

    # Save normals for use in backend prediction
    normals_serializable = {}
    for (lat, lon), monthly in normals.items():
        key = f"{lat},{lon}"
        normals_serializable[key] = monthly
    normals_path = os.path.join(MODELS_DIR, 'climate_normals.json')
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(normals_path, 'w') as f:
        json.dump(normals_serializable, f)
    print(f"   Saved normals -> {normals_path}")

    # Step 2: Add KNN interpolation features
    df = add_knn_features(df, normals, k=5)

    # Step 3: Add engineered features
    df = add_features(df)

    feature_cols = get_feature_columns()
    X = df[feature_cols]

    # Split
    X_train, X_test, indices_train, indices_test = train_test_split(
        X, df.index, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    all_metrics = {}

    # Train regression models
    for target in ['temperature', 'humidity', 'wind_speed', 'rainfall_mm']:
        y_train = df.loc[indices_train, target]
        y_test = df.loc[indices_test, target]

        model, metrics = train_xgb_regressor(
            X_train_scaled, X_test_scaled, y_train, y_test, target
        )

        model_path = os.path.join(MODELS_DIR, f'{target}_model.json')
        model.save_model(model_path)
        print(f"  Saved -> {model_path}")

        all_metrics[target] = metrics

    # Train rain classifier
    y_train = df.loc[indices_train, 'rain']
    y_test = df.loc[indices_test, 'rain']

    model, metrics = train_xgb_classifier(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    model_path = os.path.join(MODELS_DIR, 'rain_model.json')
    model.save_model(model_path)
    print(f"  Saved -> {model_path}")

    all_metrics['rain'] = metrics

    # Save scaler as JSON for cross-platform stability
    scaler_data = {
        'mean_': scaler.mean_.tolist(),
        'scale_': scaler.scale_.tolist()
    }
    scaler_path = os.path.join(MODELS_DIR, 'scaler.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler_data, f)
    print(f"\n  Scaler saved -> {scaler_path}")

    # Save metrics
    output = {
        'metrics': all_metrics,
        'features': {
            'feature_columns': feature_cols,
            'knn_features': ['knn_temp', 'knn_hum', 'knn_wind', 'knn_rain'],
        },
        'training_info': {
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'algorithm': 'XGBoost + KNN Interpolation (Hybrid)',
            'framework': 'xgboost + scikit-learn',
            'architecture': 'KNN climate-normal interpolation as features, '
                            'XGBoost for refinement based on local geography'
        }
    }

    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[*] Metrics saved -> {metrics_path}")
    print("\n[DONE] Training complete! Architecture: KNN Interpolation + XGBoost Hybrid")


if __name__ == '__main__':
    main()
