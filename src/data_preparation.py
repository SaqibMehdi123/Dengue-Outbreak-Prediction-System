"""
Dengue Prediction Pipeline - Data Preparation
==============================================
Phase 1: Load, clean, and merge data from multiple sources.

Data Sources:
1. DOH Philippines - Dengue cases
2. NASA POWER API - Weather data  
3. MODIS Satellite - NDVI vegetation index
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, RAW_DENGUE_FILE, RAW_WEATHER_DIR, RAW_NDVI_FILE,
    MERGED_DATA_FILE, REGION_COORDINATES
)


def load_dengue_data(filepath: str = None) -> pd.DataFrame:
    """
    Load raw dengue case data from DOH Philippines.
    
    Args:
        filepath: Path to dengue CSV file (uses config if None)
        
    Returns:
        DataFrame with columns: Region_ID, date, cases
    """
    filepath = filepath or RAW_DENGUE_FILE
    
    print("📥 Loading dengue case data...")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Standardize region names
    if 'Region' in df.columns:
        df['Region_ID'] = df['Region'].str.upper().str.strip()
    elif 'Region_ID' not in df.columns:
        df['Region_ID'] = df['loc'].str.upper().str.strip()
    
    # Aggregate to weekly by region
    df_weekly = df.groupby(['Region_ID', pd.Grouper(key='date', freq='W')]).agg({
        'cases': 'sum'
    }).reset_index()
    
    print(f"   ✓ Loaded {len(df_weekly):,} weekly records from {df_weekly['date'].min().date()} to {df_weekly['date'].max().date()}")
    print(f"   ✓ Regions: {df_weekly['Region_ID'].nunique()}")
    
    return df_weekly


def load_weather_data(weather_dir: str = None) -> pd.DataFrame:
    """
    Load weather data from NASA POWER API CSVs.
    
    Args:
        weather_dir: Directory containing weather CSV files
        
    Returns:
        DataFrame with columns: Region_ID, date, PRECTOTCORR, T2M, T2M_MAX, T2M_MIN, RH2M
    """
    weather_dir = weather_dir or RAW_WEATHER_DIR
    
    print("🌤️ Loading weather data...")
    
    all_weather = []
    weather_files = [f for f in os.listdir(weather_dir) if f.endswith('.csv')]
    
    for filename in weather_files:
        filepath = os.path.join(weather_dir, filename)
        
        # Extract region name from filename
        region_name = filename.replace('weather_', '').replace('.csv', '')
        region_name = region_name.replace('_', ' ').upper()
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].astype(str).agg('-'.join, axis=1))
        df['Region_ID'] = region_name
        
        # Select relevant columns
        weather_cols = ['Region_ID', 'date', 'PRECTOTCORR', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M']
        available_cols = [c for c in weather_cols if c in df.columns]
        df = df[available_cols]
        
        # Resample to weekly
        df_weekly = df.groupby(['Region_ID', pd.Grouper(key='date', freq='W')]).agg({
            'PRECTOTCORR': 'sum',  # Total weekly rainfall
            'T2M': 'mean',          # Average temperature
            'T2M_MAX': 'max',       # Max temperature
            'T2M_MIN': 'min',       # Min temperature
            'RH2M': 'mean'          # Average humidity
        }).reset_index()
        
        all_weather.append(df_weekly)
    
    df_weather = pd.concat(all_weather, ignore_index=True)
    print(f"   ✓ Loaded weather for {len(weather_files)} regions")
    
    return df_weather


def load_ndvi_data(filepath: str = None) -> pd.DataFrame:
    """
    Load NDVI vegetation index from MODIS satellite data.
    
    Args:
        filepath: Path to NDVI CSV file
        
    Returns:
        DataFrame with columns: Region_ID, date, NDVI
    """
    filepath = filepath or RAW_NDVI_FILE
    
    print("🛰️ Loading NDVI satellite data...")
    
    if not os.path.exists(filepath):
        print("   ⚠️ NDVI file not found, skipping...")
        return None
        
    df = pd.read_csv(filepath)
    
    # Parse dates and aggregate
    if 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
    
    # NDVI processing varies by file format
    # Return placeholder if format doesn't match
    if 'NDVI' not in df.columns and 'MOD13Q1_061__250m_16_days_NDVI' not in df.columns:
        print("   ⚠️ NDVI column not found in expected format")
        return None
    
    print(f"   ✓ Loaded NDVI data")
    return df


def merge_all_sources(dengue_df: pd.DataFrame = None, 
                      weather_df: pd.DataFrame = None,
                      ndvi_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merge all data sources into a single dataset.
    
    Args:
        dengue_df: Dengue case data
        weather_df: Weather data
        ndvi_df: NDVI data (optional)
        
    Returns:
        Merged DataFrame
    """
    print("\n🔗 Merging data sources...")
    
    # Load data if not provided
    if dengue_df is None:
        dengue_df = load_dengue_data()
    if weather_df is None:
        weather_df = load_weather_data()
    
    # Standardize region names for merging
    dengue_df['Region_ID'] = dengue_df['Region_ID'].str.upper().str.strip()
    weather_df['Region_ID'] = weather_df['Region_ID'].str.upper().str.strip()
    
    # Merge dengue with weather
    df_merged = pd.merge(
        dengue_df,
        weather_df,
        on=['Region_ID', 'date'],
        how='inner'
    )
    
    print(f"   ✓ Merged dengue + weather: {len(df_merged):,} records")
    
    # Add NDVI if available
    if ndvi_df is not None and len(ndvi_df) > 0:
        df_merged = pd.merge(df_merged, ndvi_df, on=['Region_ID', 'date'], how='left')
        df_merged['NDVI'] = df_merged['NDVI'].fillna(df_merged['NDVI'].mean())
        print(f"   ✓ Added NDVI data")
    else:
        # Create placeholder NDVI if not available
        df_merged['NDVI'] = 0.5  # Neutral value
        print(f"   ⚠️ NDVI not available, using placeholder")
    
    return df_merged


def save_merged_data(df: pd.DataFrame, filepath: str = None) -> None:
    """Save merged dataset to CSV."""
    filepath = filepath or MERGED_DATA_FILE
    df.to_csv(filepath, index=False)
    print(f"\n💾 Saved merged data to: {filepath}")


def load_merged_data(filepath: str = None) -> pd.DataFrame:
    """Load previously merged dataset."""
    filepath = filepath or MERGED_DATA_FILE
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def run_data_preparation() -> pd.DataFrame:
    """
    Run the complete data preparation pipeline.
    
    Returns:
        Merged and cleaned DataFrame
    """
    print("=" * 60)
    print("📊 PHASE 1: DATA PREPARATION")
    print("=" * 60)
    
    # Check if merged data already exists
    if os.path.exists(MERGED_DATA_FILE):
        print(f"\n✓ Using existing merged data: {MERGED_DATA_FILE}")
        return load_merged_data()
    
    # Load and merge from scratch
    df = merge_all_sources()
    save_merged_data(df)
    
    return df


# =============================================================================
# Run if executed directly
# =============================================================================
if __name__ == "__main__":
    df = run_data_preparation()
    print(f"\n✅ Data preparation complete!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
