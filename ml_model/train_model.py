"""
Weather Model Trainer
Trains RandomForest models for weather prediction:
  - Temperature (regression)
  - Humidity (regression)
  - Wind Speed (regression)
  - Rain / No Rain (classification)
"""

import numpy as np
import pandas as pd
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'weather_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')


FEATURE_COLS = ['latitude', 'longitude', 'month', 'day_of_year', 'elevation', 'distance_to_coast']
REGRESSION_TARGETS = ['temperature', 'humidity', 'wind_speed']
CLASSIFICATION_TARGET = 'rain'


def load_data():
    """Load and prepare the weather dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"📂 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def add_features(df):
    """Engineer additional features for better predictions."""
    df = df.copy()
    
    # Cyclical encoding of month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Cyclical encoding of day_of_year
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Absolute latitude (distance from equator)
    df['abs_latitude'] = np.abs(df['latitude'])
    
    # Hemisphere indicator
    df['is_northern'] = (df['latitude'] >= 0).astype(int)
    
    return df


def get_feature_columns():
    """Return the full list of feature columns after engineering."""
    return FEATURE_COLS + [
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'abs_latitude', 'is_northern'
    ]


def train_regression_model(X_train, X_test, y_train, y_test, target_name):
    """Train a RandomForestRegressor for a given target."""
    print(f"\n{'='*50}")
    print(f"🔧 Training model: {target_name}")
    print(f"{'='*50}")
    
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²:   {r2:.4f}")
    
    metrics = {
        'type': 'regression',
        'target': target_name,
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2_score': round(r2, 4),
        'n_estimators': 150,
        'max_depth': 20,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return model, metrics


def train_classification_model(X_train, X_test, y_train, y_test):
    """Train a RandomForestClassifier for rain prediction."""
    print(f"\n{'='*50}")
    print(f"🔧 Training model: rain (classification)")
    print(f"{'='*50}")
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
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
        'n_estimators': 150,
        'max_depth': 20,
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    }
    
    return model, metrics


def main():
    print("🌦️  Weather Model Training Pipeline")
    print("=" * 60)
    
    # Load & prepare data
    df = load_data()
    df = add_features(df)
    
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    
    # Split data
    X_train, X_test, indices_train, indices_test = train_test_split(
        X, df.index, test_size=0.2, random_state=42
    )
    
    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    all_metrics = {}
    
    # Train regression models
    for target in REGRESSION_TARGETS:
        y_train = df.loc[indices_train, target]
        y_test = df.loc[indices_test, target]
        
        model, metrics = train_regression_model(
            X_train_scaled, X_test_scaled, y_train, y_test, target
        )
        
        model_path = os.path.join(MODELS_DIR, f'{target}_model.joblib')
        joblib.dump(model, model_path)
        print(f"  💾 Saved → {model_path}")
        
        all_metrics[target] = metrics
    
    # Train classification model
    y_train = df.loc[indices_train, CLASSIFICATION_TARGET]
    y_test = df.loc[indices_test, CLASSIFICATION_TARGET]
    
    model, metrics = train_classification_model(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    model_path = os.path.join(MODELS_DIR, 'rain_model.joblib')
    joblib.dump(model, model_path)
    print(f"  💾 Saved → {model_path}")
    
    all_metrics['rain'] = metrics
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"\n  💾 Scaler saved → {scaler_path}")
    
    # Save feature columns list
    feature_info = {
        'feature_columns': feature_cols,
        'original_features': FEATURE_COLS,
        'engineered_features': [c for c in feature_cols if c not in FEATURE_COLS]
    }
    
    # Save all metrics
    output = {
        'metrics': all_metrics,
        'features': feature_info,
        'training_info': {
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'algorithm': 'RandomForest',
            'framework': 'scikit-learn'
        }
    }
    
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📊 Metrics saved → {metrics_path}")
    print("\n✅ All models trained and saved successfully!")


if __name__ == '__main__':
    main()
