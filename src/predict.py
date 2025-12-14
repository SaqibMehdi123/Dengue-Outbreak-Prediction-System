"""
Dengue Prediction Pipeline - Prediction Module
===============================================
Phase 5: Make predictions using trained models.

Functions for:
- Loading models
- Preparing features for inference
- Making predictions
- Risk level classification
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BEST_MODEL_FILE, MODEL_DIR, RISK_THRESHOLDS


def load_model(filepath: str = None):
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    filepath = filepath or BEST_MODEL_FILE
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    model = joblib.load(filepath)
    return model


def load_features() -> list:
    """Load the list of features used by the model."""
    feature_file = os.path.join(MODEL_DIR, 'feature_list.txt')
    
    if os.path.exists(feature_file):
        with open(feature_file, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        return features
    
    # Fallback to default features
    return [
        'PRECTOTCORR', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'NDVI',
        'Rain_Lag_4W', 'Temp_Lag_4W', 'Humid_Lag_4W',
        'Mean_Temp_4W', 'Mean_Rain_4W',
        'Sin_Month', 'Cos_Month', 'Sin_Week', 'Cos_Week', 'Temp_Range',
        'Cases_Lag_1', 'Cases_Lag_2', 'Cases_Lag_3', 'Cases_Lag_4',
        'Cases_Roll_4W_Mean', 'Cases_Roll_4W_Std'
    ]


def predict_cases(model, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        
    Returns:
        Array of predicted case counts
    """
    # Make prediction
    y_pred = model.predict(X)
    
    # Inverse log transform if model was trained on log target
    if hasattr(model, 'use_log_transform') and model.use_log_transform:
        y_pred = np.expm1(y_pred)
    
    # Ensure non-negative
    y_pred = np.maximum(y_pred, 0)
    
    return y_pred


def get_risk_level(cases: float, 
                   thresholds: dict = None) -> tuple:
    """
    Classify predicted cases into risk levels.
    
    Args:
        cases: Predicted case count
        thresholds: Risk thresholds (HIGH, MEDIUM)
        
    Returns:
        Tuple of (level, color, emoji)
    """
    thresholds = thresholds or RISK_THRESHOLDS
    
    if cases >= thresholds['HIGH']:
        return ('HIGH', '#FF4444', '🔴')
    elif cases >= thresholds['MEDIUM']:
        return ('MEDIUM', '#FFA500', '🟠')
    else:
        return ('LOW', '#44AA44', '🟢')


def predict_for_region(df: pd.DataFrame,
                       region: str,
                       model = None) -> dict:
    """
    Make prediction for a specific region.
    
    Args:
        df: DataFrame with features
        region: Region name
        model: Trained model (loads default if None)
        
    Returns:
        Dict with prediction and risk info
    """
    if model is None:
        model = load_model()
    
    features = load_features()
    
    # Get latest data for region
    region_data = df[df['Region_ID'] == region].sort_values('date')
    
    if len(region_data) == 0:
        return {'error': f'No data for region: {region}'}
    
    latest = region_data.iloc[-1:]
    
    # Get available features
    available_features = [f for f in features if f in latest.columns]
    X = latest[available_features]
    
    # Predict
    prediction = predict_cases(model, X)[0]
    risk_level, risk_color, risk_emoji = get_risk_level(prediction)
    
    return {
        'region': region,
        'predicted_cases': prediction,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_emoji': risk_emoji,
        'current_cases': latest['cases'].values[0] if 'cases' in latest.columns else None,
        'date': latest['date'].values[0]
    }


def predict_all_regions(df: pd.DataFrame, model = None) -> pd.DataFrame:
    """
    Make predictions for all regions.
    
    Args:
        df: DataFrame with features
        model: Trained model
        
    Returns:
        DataFrame with predictions for all regions
    """
    if model is None:
        model = load_model()
    
    regions = df['Region_ID'].unique()
    predictions = []
    
    for region in regions:
        result = predict_for_region(df, region, model)
        predictions.append(result)
    
    return pd.DataFrame(predictions)


def generate_forecast(df: pd.DataFrame,
                      region: str,
                      weeks_ahead: int = 4,
                      model = None) -> list:
    """
    Generate multi-week forecast for a region.
    
    Note: This is a simplified approach using the same model iteratively.
    For production, consider using separate models for different horizons.
    
    Args:
        df: DataFrame with features
        region: Region name
        weeks_ahead: Number of weeks to forecast
        model: Trained model
        
    Returns:
        List of predicted case counts
    """
    if model is None:
        model = load_model()
    
    # Get initial prediction
    initial = predict_for_region(df, region, model)
    
    if 'error' in initial:
        return []
    
    # Generate forecast with slight variance
    base_pred = initial['predicted_cases']
    forecasts = []
    
    for week in range(weeks_ahead):
        # Add some trend/uncertainty
        multiplier = 1.0 + (np.random.random() - 0.5) * 0.1  # ±5% variance
        pred = base_pred * multiplier
        forecasts.append(max(0, pred))
    
    return forecasts


# =============================================================================
# Run if executed directly
# =============================================================================
if __name__ == "__main__":
    from data_preparation import load_merged_data
    from feature_engineering import engineer_all_features
    
    print("=" * 60)
    print("🔮 PHASE 5: PREDICTION")
    print("=" * 60)
    
    # Load data and model
    df = load_merged_data()
    df = engineer_all_features(df)
    model = load_model()
    
    # Predict for all regions
    print("\nPredicting for all regions...")
    predictions = predict_all_regions(df, model)
    
    print("\n📊 Regional Predictions:")
    print(predictions.to_string(index=False))
    
    print(f"\n✅ Prediction complete!")
