"""
Dengue Prediction Pipeline - Model Training
============================================
Phase 3: Train and save ML models.

Models:
1. XGBoost (primary) - Gradient boosting, best performer
2. Random Forest - Ensemble baseline
3. Ridge Regression - Regularized linear baseline
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_TEST_SPLIT_DATE, EXCLUDE_FROM_FEATURES,
    XGBOOST_PARAMS, RANDOM_FOREST_PARAMS, RIDGE_PARAMS,
    MODEL_DIR, BEST_MODEL_FILE, RF_MODEL_FILE, RIDGE_MODEL_FILE
)


def prepare_train_test(df: pd.DataFrame,
                       split_date: str = None,
                       target_col: str = 'Cases_Target') -> tuple:
    """
    Split data into train/test sets using time-based split.
    
    Args:
        df: Feature-engineered DataFrame
        split_date: Date to split on (everything before = train)
        target_col: Name of target column
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    split_date = split_date or TRAIN_TEST_SPLIT_DATE
    
    print(f"   Splitting data at {split_date}...")
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Split by date
    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]
    
    # Get feature columns
    exclude = EXCLUDE_FROM_FEATURES + [target_col]
    features = [col for col in df.columns if col not in exclude and 'Cases_Target' not in col]
    
    X_train = train[features]
    X_test = test[features]
    y_train = train[target_col]
    y_test = test[target_col]
    
    print(f"      ✓ Train: {len(train):,} samples ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"      ✓ Test: {len(test):,} samples ({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"      ✓ Features: {len(features)}")
    
    return X_train, X_test, y_train, y_test, features


def train_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_val: pd.DataFrame = None,
                  y_val: pd.Series = None,
                  params: dict = None,
                  use_log_transform: bool = True) -> xgb.XGBRegressor:
    """
    Train XGBoost model with optimized hyperparameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        params: XGBoost hyperparameters
        use_log_transform: Whether to train on log-transformed target
        
    Returns:
        Trained XGBoost model
    """
    params = params or XGBOOST_PARAMS
    
    print("\n   Training XGBoost model...")
    
    # Log transform target (stabilizes variance)
    if use_log_transform:
        y_train_transformed = np.log1p(y_train)
        if y_val is not None:
            y_val_transformed = np.log1p(y_val)
    else:
        y_train_transformed = y_train
        y_val_transformed = y_val if y_val is not None else None
    
    # Initialize model
    model = xgb.XGBRegressor(**params)
    
    # Fit with or without validation set
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train_transformed,
            eval_set=[(X_val, y_val_transformed)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train_transformed, verbose=False)
    
    # Store log transform flag for prediction
    model.use_log_transform = use_log_transform
    
    print(f"      ✓ XGBoost trained: {params['n_estimators']} trees, depth={params['max_depth']}")
    
    return model


def train_random_forest(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       params: dict = None,
                       use_log_transform: bool = True) -> RandomForestRegressor:
    """
    Train Random Forest model.
    
    Args:
        X_train, y_train: Training data
        params: Random Forest hyperparameters
        use_log_transform: Whether to train on log-transformed target
        
    Returns:
        Trained Random Forest model
    """
    params = params or RANDOM_FOREST_PARAMS
    
    print("\n   Training Random Forest model...")
    
    # Log transform target
    if use_log_transform:
        y_train_transformed = np.log1p(y_train)
    else:
        y_train_transformed = y_train
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train_transformed)
    
    model.use_log_transform = use_log_transform
    
    print(f"      ✓ Random Forest trained: {params['n_estimators']} trees")
    
    return model


def train_ridge(X_train: pd.DataFrame,
                y_train: pd.Series,
                params: dict = None,
                use_log_transform: bool = True):
    """
    Train Ridge Regression model with standardization.
    Uses careful feature selection focusing on most predictive features.
    
    Args:
        X_train, y_train: Training data
        params: Ridge hyperparameters (optional, uses CV if not specified)
        use_log_transform: Whether to train on log-transformed target
        
    Returns:
        Trained Ridge model dict with model and metadata
    """
    from sklearn.linear_model import RidgeCV
    
    print("\n   Training Ridge Regression model...")
    
    # Log transform target
    if use_log_transform:
        y_train_transformed = np.log1p(y_train)
    else:
        y_train_transformed = y_train
    
    # Handle any remaining NaN or infinite values in features
    X_train_clean = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Select the best features for linear models
    # Focus on features with near-linear relationships to target
    selected_cols = []
    for col in X_train_clean.columns:
        # Include case-related features (strongest predictors)
        if 'Cases' in col:
            selected_cols.append(col)
        # Include cyclical features (good for seasonality)
        elif 'Sin' in col or 'Cos' in col:
            selected_cols.append(col)
        # Include rolling mean weather features
        elif 'Mean_' in col:
            selected_cols.append(col)
    
    # If not enough features, fallback to all
    if len(selected_cols) < 10:
        selected_cols = list(X_train_clean.columns)
    
    X_selected = X_train_clean[selected_cols]
    print(f"      ✓ Using {len(selected_cols)} selected features for Ridge")
    
    # Create a simple pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0))  # Default alpha
    ])
    
    # Find best alpha using RidgeCV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    alphas = np.logspace(-3, 5, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_scaled, y_train_transformed)
    
    best_alpha = ridge_cv.alpha_
    print(f"      ✓ Best alpha from CV: {best_alpha:.4f}")
    
    # Create final pipeline with best alpha
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=best_alpha))
    ])
    model.fit(X_selected, y_train_transformed)
    
    # Store metadata
    model.use_log_transform = use_log_transform
    model.selected_cols = selected_cols
    
    # Get R² on training data for feedback
    y_train_pred_log = model.predict(X_selected)
    from sklearn.metrics import r2_score
    train_r2 = r2_score(y_train_transformed, y_train_pred_log)
    print(f"      ✓ Ridge trained: alpha={best_alpha:.4f}, Train R²={train_r2:.4f}")
    
    return model


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, 
                        n_splits: int = 3) -> dict:
    """
    Perform time-series cross-validation.
    
    Returns:
        Dict with CV scores
    """
    print("\n   Running cross-validation...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Transform y if needed
    y_transformed = np.log1p(y) if hasattr(model, 'use_log_transform') and model.use_log_transform else y
    
    scores = cross_val_score(model, X, y_transformed, cv=tscv, scoring='neg_mean_absolute_error')
    
    results = {
        'cv_mae_mean': -scores.mean(),
        'cv_mae_std': scores.std()
    }
    
    print(f"      ✓ CV MAE: {results['cv_mae_mean']:.2f} (+/- {results['cv_mae_std']:.2f})")
    
    return results


def save_model(model, filepath: str = None, model_name: str = 'xgboost') -> None:
    """Save trained model to disk."""
    if filepath is None:
        filepath = BEST_MODEL_FILE if model_name == 'xgboost' else RF_MODEL_FILE
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"\n💾 Saved {model_name} model to: {filepath}")


def load_model(filepath: str = None) -> object:
    """Load a trained model from disk."""
    filepath = filepath or BEST_MODEL_FILE
    return joblib.load(filepath)


def train_all_models(df: pd.DataFrame) -> dict:
    """
    Train all models and return them with metadata.
    
    Args:
        df: Feature-engineered DataFrame
        
    Returns:
        Dict of trained models
    """
    print("=" * 60)
    print("🤖 PHASE 3: MODEL TRAINING")
    print("=" * 60)
    
    # Prepare data
    X_train, X_test, y_train, y_test, features = prepare_train_test(df)
    
    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    
    # Train Ridge Regression
    ridge_model = train_ridge(X_train, y_train)
    
    # Save models
    save_model(xgb_model, model_name='xgboost')
    save_model(rf_model, RF_MODEL_FILE, model_name='random_forest')
    save_model(ridge_model, RIDGE_MODEL_FILE, model_name='ridge')
    
    # Save feature list
    feature_file = os.path.join(MODEL_DIR, 'feature_list.txt')
    with open(feature_file, 'w') as f:
        f.write('\n'.join(features))
    print(f"💾 Saved feature list to: {feature_file}")
    
    return {
        'xgboost': xgb_model,
        'random_forest': rf_model,
        'ridge': ridge_model,
        'features': features,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


# =============================================================================
# Run if executed directly
# =============================================================================
if __name__ == "__main__":
    from data_preparation import run_data_preparation
    from feature_engineering import engineer_all_features
    
    # Load and prepare data
    df = run_data_preparation()
    df = engineer_all_features(df)
    
    # Train models
    results = train_all_models(df)
    
    print(f"\n✅ Model training complete!")
