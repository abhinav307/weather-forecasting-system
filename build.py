#!/usr/bin/env python
"""
Build script for deployment — generates training data and trains ML models.
Run this during deployment build step if models don't exist yet.
"""

import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, 'ml_model', 'saved_models')

def main():
    # Check if models already exist
    required_files = [
        'temperature_model.json',
        'humidity_model.json',
        'wind_speed_model.json',
        'rain_model.json',
        'rainfall_mm_model.json',
        'scaler.json',
        'model_metrics.json'
    ]
    
    all_exist = all(
        os.path.exists(os.path.join(MODELS_DIR, f))
        for f in required_files
    )
    
    if all_exist:
        print("✅ Models already exist, skipping build.")
        return
    
    print("🔧 Models not found — generating data and training...")
    
    # Create saved_models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Step 1: Generate training data
    print("\n📊 Step 1/2: Generating training data...")
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_DIR, 'ml_model', 'generate_data.py')],
        cwd=PROJECT_DIR
    )
    if result.returncode != 0:
        print("❌ Data generation failed!")
        sys.exit(1)
    
    # Step 2: Train models
    print("\n🤖 Step 2/2: Training ML models...")
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_DIR, 'ml_model', 'train_model.py')],
        cwd=PROJECT_DIR
    )
    if result.returncode != 0:
        print("❌ Model training failed!")
        sys.exit(1)
    
    print("\n✅ Build complete! All models trained and saved.")


if __name__ == '__main__':
    main()
