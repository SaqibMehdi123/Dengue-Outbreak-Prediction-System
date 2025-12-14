"""
Dengue Prediction Pipeline - Feature Engineering
=================================================
Phase 2: Create ML features from raw data.

Feature Categories:
1. Lag Features - Delayed effects (rain today → cases in 4 weeks)
2. Rolling Features - Sustained conditions
3. Momentum Features - Outbreak trends
4. Interaction Features - Combined weather effects
5. Cyclical Features - Seasonality
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WEATHER_LAG_PERIODS, CASE_LAG_PERIODS, ROLLING_WINDOWS,
    PREDICTION_HORIZON, ENGINEERED_DATA_FILE
)


def create_lag_features(df: pd.DataFrame, 
                        weather_lags: list = None,
                        case_lags: list = None) -> pd.DataFrame:
    """
    Create lag features for weather and case variables.
    
    Biological rationale:
    - Rain 4 weeks ago → Mosquito breeding → Cases today
    - Mosquito lifecycle: 7-10 days egg to adult
    - Dengue incubation: 4-10 days in humans
    """
    weather_lags = weather_lags or WEATHER_LAG_PERIODS
    case_lags = case_lags or CASE_LAG_PERIODS
    
    print("   Creating lag features...")
    
    # Weather lags
    for lag in weather_lags:
        df[f'Rain_Lag_{lag}W'] = df.groupby('Region_ID')['PRECTOTCORR'].shift(lag)
        df[f'Temp_Lag_{lag}W'] = df.groupby('Region_ID')['T2M'].shift(lag)
        df[f'Humid_Lag_{lag}W'] = df.groupby('Region_ID')['RH2M'].shift(lag)
    
    # Case lags (auto-regressive)
    for lag in case_lags:
        df[f'Cases_Lag_{lag}'] = df.groupby('Region_ID')['cases'].shift(lag)
    
    print(f"      ✓ Weather lags: {weather_lags}")
    print(f"      ✓ Case lags: {case_lags}")
    
    return df


def create_rolling_features(df: pd.DataFrame,
                           windows: list = None) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    Rationale: A single hot day doesn't cause outbreak - need sustained conditions.
    """
    windows = windows or ROLLING_WINDOWS
    
    print("   Creating rolling features...")
    
    for window in windows:
        # Weather rolling stats
        df[f'Mean_Temp_{window}W'] = df.groupby('Region_ID')['T2M'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'Mean_Rain_{window}W'] = df.groupby('Region_ID')['PRECTOTCORR'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'Std_Rain_{window}W'] = df.groupby('Region_ID')['PRECTOTCORR'].transform(
            lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
        df[f'Max_Rain_{window}W'] = df.groupby('Region_ID')['PRECTOTCORR'].transform(
            lambda x: x.rolling(window, min_periods=1).max())
        
        # Case rolling stats
        df[f'Cases_Roll_{window}W_Mean'] = df.groupby('Region_ID')['cases'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'Cases_Roll_{window}W_Std'] = df.groupby('Region_ID')['cases'].transform(
            lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
        df[f'Cases_Roll_{window}W_Max'] = df.groupby('Region_ID')['cases'].transform(
            lambda x: x.rolling(window, min_periods=1).max())
    
    print(f"      ✓ Rolling windows: {windows}")
    
    return df


def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create momentum/trend features.
    
    Rationale: Is the outbreak growing or shrinking? Rate of change matters.
    """
    print("   Creating momentum features...")
    
    # Week-over-week change
    df['Cases_Diff_1'] = df.groupby('Region_ID')['cases'].diff(1)
    df['Cases_Diff_4'] = df.groupby('Region_ID')['cases'].diff(4)
    
    # Percentage change (clipped to prevent extreme values)
    df['Cases_Pct_1'] = df.groupby('Region_ID')['cases'].pct_change(1).fillna(0).clip(-10, 10)
    df['Cases_Pct_4'] = df.groupby('Region_ID')['cases'].pct_change(4).fillna(0).clip(-10, 10)
    
    # Ratio to rolling average (current vs typical)
    df['Cases_vs_4W_Avg'] = df['cases'] / (df['Cases_Roll_4W_Mean'] + 1)
    df['Cases_vs_8W_Avg'] = df['cases'] / (df['Cases_Roll_8W_Mean'] + 1)
    
    print(f"      ✓ Diff, pct_change, ratio features")
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between weather variables.
    
    Rationale: Rain + humidity TOGETHER create ideal breeding conditions.
    """
    print("   Creating interaction features...")
    
    # Current weather interactions
    df['Rain_x_Humidity'] = df['PRECTOTCORR'] * df['RH2M'] / 100
    df['Temp_x_Rain'] = df['T2M'] * df['PRECTOTCORR'] / 100
    df['Rain_x_NDVI'] = df['PRECTOTCORR'] * df['NDVI']
    
    # Lagged interactions
    if 'Rain_Lag_4W' in df.columns and 'Humid_Lag_4W' in df.columns:
        df['Rain4_x_Humid4'] = df['Rain_Lag_4W'] * df['Humid_Lag_4W'] / 100
    
    print(f"      ✓ Rain×Humidity, Temp×Rain, Rain×NDVI")
    
    return df


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical encoding for time features.
    
    Rationale: December (12) should be close to January (1) in feature space.
    Sin/cos encoding achieves this.
    """
    print("   Creating cyclical features...")
    
    df['Month'] = df['date'].dt.month
    df['Week_of_Year'] = df['date'].dt.isocalendar().week.astype(int)
    df['Year'] = df['date'].dt.year
    
    # Cyclical encoding
    df['Sin_Month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Cos_Month'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Sin_Week'] = np.sin(2 * np.pi * df['Week_of_Year'] / 52)
    df['Cos_Week'] = np.cos(2 * np.pi * df['Week_of_Year'] / 52)
    
    # Temperature range
    df['Temp_Range'] = df['T2M_MAX'] - df['T2M_MIN']
    
    print(f"      ✓ Sin/Cos encoding for month and week")
    
    return df


def create_target_variable(df: pd.DataFrame, 
                          horizon: int = None) -> pd.DataFrame:
    """
    Create the prediction target: cases N weeks ahead.
    
    Args:
        df: Input DataFrame
        horizon: Weeks ahead to predict (default from config)
    """
    horizon = horizon or PREDICTION_HORIZON
    
    print(f"   Creating target variable ({horizon}-week ahead)...")
    
    df['Cases_Target'] = df.groupby('Region_ID')['cases'].shift(-horizon)
    
    print(f"      ✓ Cases_Target = cases shifted by -{horizon} weeks")
    
    return df


def engineer_all_features(df: pd.DataFrame,
                         prediction_horizon: int = None) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    
    Args:
        df: Raw merged DataFrame
        prediction_horizon: Weeks ahead to predict
        
    Returns:
        DataFrame with all engineered features
    """
    print("=" * 60)
    print("⚙️ PHASE 2: FEATURE ENGINEERING")
    print("=" * 60)
    
    # Ensure proper sorting
    df = df.sort_values(['Region_ID', 'date']).reset_index(drop=True)
    
    # Create all features
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_momentum_features(df)
    df = create_interaction_features(df)
    df = create_cyclical_features(df)
    df = create_target_variable(df, prediction_horizon)
    
    # Drop rows with NaN (from lagging operations)
    original_len = len(df)
    df = df.dropna().reset_index(drop=True)
    
    print(f"\n   ✓ Dropped {original_len - len(df)} rows with NaN")
    print(f"   ✓ Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


def save_engineered_data(df: pd.DataFrame, filepath: str = None) -> None:
    """Save engineered dataset to CSV."""
    filepath = filepath or ENGINEERED_DATA_FILE
    df.to_csv(filepath, index=False)
    print(f"\n💾 Saved engineered data to: {filepath}")


def load_engineered_data(filepath: str = None) -> pd.DataFrame:
    """Load previously engineered dataset."""
    filepath = filepath or ENGINEERED_DATA_FILE
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


# =============================================================================
# Run if executed directly
# =============================================================================
if __name__ == "__main__":
    from data_preparation import run_data_preparation
    
    # Load merged data
    df = run_data_preparation()
    
    # Engineer features
    df = engineer_all_features(df)
    save_engineered_data(df)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Total features: {df.shape[1]}")
