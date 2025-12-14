"""
Dengue Prediction Pipeline - Configuration
==========================================
Central configuration for paths, hyperparameters, and feature definitions.
"""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data files
RAW_DENGUE_FILE = os.path.join(DATA_DIR, 'philippines_dengue.csv')
RAW_WEATHER_DIR = os.path.join(DATA_DIR, 'weather')
RAW_NDVI_FILE = os.path.join(DATA_DIR, 'Philippines-Vegetation-MOD13Q1-061-results.csv')
MERGED_DATA_FILE = os.path.join(DATA_DIR, 'philippines_dengue_dataset_FINAL.csv')
ENGINEERED_DATA_FILE = os.path.join(DATA_DIR, 'dengue_dataset_engineered.csv')

# Model files
BEST_MODEL_FILE = os.path.join(MODEL_DIR, 'xgboost_best.joblib')
RF_MODEL_FILE = os.path.join(MODEL_DIR, 'random_forest.joblib')
RIDGE_MODEL_FILE = os.path.join(MODEL_DIR, 'ridge.joblib')

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
TRAIN_TEST_SPLIT_DATE = '2020-01-01'  # Everything before = train, after = test
PREDICTION_HORIZON = 2  # Weeks ahead to predict (2 = R²=0.80, 4 = R²=0.71)

# Region coordinates (for weather API calls)
REGION_COORDINATES = {
    'NATIONAL CAPITAL REGION': (14.5995, 120.9842),
    'REGION I-ILOCOS REGION': (16.0832, 120.6200),
    'REGION II-CAGAYAN VALLEY': (16.9754, 121.8107),
    'REGION III-CENTRAL LUZON': (15.4755, 120.5963),
    'REGION IV-A-CALABARZON': (14.1008, 121.0794),
    'REGION IV-B-MIMAROPA': (12.8797, 121.7740),
    'REGION V-BICOL REGION': (13.4210, 123.4137),
    'REGION VI-WESTERN VISAYAS': (10.7202, 122.5621),
    'REGION VII-CENTRAL VISAYAS': (9.8500, 123.8907),
    'REGION VIII-EASTERN VISAYAS': (11.2543, 125.0000),
    'REGION IX-ZAMBOANGA PENINSULA': (7.8044, 123.4390),
    'REGION X-NORTHERN MINDANAO': (8.0200, 124.6853),
    'REGION XI-DAVAO REGION': (7.3041, 126.0893),
    'REGION XII-SOCCSKSARGEN': (6.2706, 124.6857),
    'REGION XIII-CARAGA': (8.8015, 125.7407),
    'CAR': (17.3513, 121.1719),
    'BARMM': (6.9568, 124.2421),
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# Lag periods for weather features (in weeks)
WEATHER_LAG_PERIODS = [1, 2, 3, 4, 6, 8, 12]

# Lag periods for case features
CASE_LAG_PERIODS = [1, 2, 3, 4, 8, 12]

# Rolling window sizes (in weeks)
ROLLING_WINDOWS = [4, 8, 12]

# Columns to exclude from features
EXCLUDE_FROM_FEATURES = ['date', 'Region_ID', 'cases', 'Cases_Target', 'Month', 'Week_of_Year']

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
XGBOOST_PARAMS = {
    'n_estimators': 800,
    'learning_rate': 0.015,
    'max_depth': 10,
    'min_child_weight': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.02,
    'reg_lambda': 0.5,
    'gamma': 0.02,
    'random_state': 42,
    'n_jobs': -1
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_leaf': 5,
    'random_state': 42,
    'n_jobs': -1
}

RIDGE_PARAMS = {
    'alpha': 10.0,  # Regularization strength
}

# =============================================================================
# RISK LEVELS (for Streamlit app)
# =============================================================================
RISK_THRESHOLDS = {
    'HIGH': 200,
    'MEDIUM': 50
}
