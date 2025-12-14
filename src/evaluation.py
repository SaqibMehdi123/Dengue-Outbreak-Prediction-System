"""
Dengue Prediction Pipeline - Model Evaluation
==============================================
Phase 4: Evaluate model performance and analyze results.

Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

Analysis:
- Feature importance
- Error distribution
- Per-region performance
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray) -> dict:
    """
    Calculate regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dict with MAE, RMSE, R²
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics


def predict_with_model(model, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions, handling log transform if needed.
    Also handles Ridge's feature selection via selected_cols.
    """
    # Handle Ridge's selected columns
    if hasattr(model, 'selected_cols'):
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        available_cols = [c for c in model.selected_cols if c in X_clean.columns]
        X_input = X_clean[available_cols]
    else:
        X_input = X
    
    y_pred = model.predict(X_input)
    
    # Inverse log transform if model was trained on log target
    if hasattr(model, 'use_log_transform') and model.use_log_transform:
        y_pred = np.expm1(y_pred)
    
    # Ensure non-negative
    y_pred = np.maximum(y_pred, 0)
    
    return y_pred


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                  model_name: str = 'Model') -> dict:
    """
    Evaluate a single model.
    
    Returns:
        Dict with predictions and metrics
    """
    print(f"\n   Evaluating {model_name}...")
    
    y_pred = predict_with_model(model, X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    print(f"      ✓ MAE: {metrics['MAE']:.2f}")
    print(f"      ✓ RMSE: {metrics['RMSE']:.2f}")
    print(f"      ✓ R²: {metrics['R2']:.4f}")
    
    return {
        'predictions': y_pred,
        'metrics': metrics
    }


def compare_models(models_dict: dict,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        models_dict: Dict of {name: model}
        X_test, y_test: Test data
        
    Returns:
        DataFrame with comparison results
    """
    print("=" * 60)
    print("📊 PHASE 4: MODEL EVALUATION")
    print("=" * 60)
    
    results = []
    
    for name, model in models_dict.items():
        eval_result = evaluate_model(model, X_test, y_test, name)
        results.append({
            'Model': name,
            **eval_result['metrics']
        })
    
    # Add baseline (persistence model)
    baseline_pred = X_test['Cases_Lag_1'].values if 'Cases_Lag_1' in X_test.columns else y_test.shift(1).fillna(y_test.mean())
    baseline_metrics = calculate_metrics(y_test, baseline_pred)
    results.append({
        'Model': 'Baseline (Persistence)',
        **baseline_metrics
    })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('R2', ascending=False)
    
    print("\n📋 Model Comparison:")
    print(df_results.to_string(index=False))
    
    return df_results


def analyze_feature_importance(model, feature_names: list,
                              top_n: int = 20) -> pd.DataFrame:
    """
    Analyze and rank feature importance.
    
    Returns:
        DataFrame with feature importance
    """
    print("\n   Analyzing feature importance...")
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print("      ⚠️ Model doesn't have feature_importances_")
        return None
    
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print(f"\n   Top {top_n} Features:")
    for i, row in df_imp.head(top_n).iterrows():
        print(f"      {row['Feature']}: {row['Importance']:.4f}")
    
    # Categorize features
    weather_features = [f for f in feature_names if any(x in f for x in ['Rain', 'Temp', 'Humid', 'NDVI', 'PREC', 'T2M', 'RH'])]
    case_features = [f for f in feature_names if 'Cases' in f]
    
    weather_imp = df_imp[df_imp['Feature'].isin(weather_features)]['Importance'].sum()
    case_imp = df_imp[df_imp['Feature'].isin(case_features)]['Importance'].sum()
    
    print(f"\n   Feature Category Contribution:")
    print(f"      Weather/Satellite: {weather_imp:.1%}")
    print(f"      Case History: {case_imp:.1%}")
    
    return df_imp


def plot_predictions(y_true: np.ndarray, 
                    y_pred: np.ndarray,
                    title: str = 'Predicted vs Actual',
                    save_path: str = None) -> plt.Figure:
    """
    Create prediction visualization plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, edgecolors='black', linewidth=0.3)
    ax1.plot([0, max(y_true)], [0, max(y_true)], 'r--', lw=2, label='Perfect')
    ax1.set_xlabel('Actual Cases')
    ax1.set_ylabel('Predicted Cases')
    ax1.set_title(f'{title}: Scatter')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time series (first 100)
    ax2 = axes[1]
    n = min(100, len(y_true))
    ax2.plot(range(n), y_true[:n], 'b-', lw=2, label='Actual', alpha=0.8)
    ax2.plot(range(n), y_pred[:n], 'r--', lw=2, label='Predicted', alpha=0.8)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Cases')
    ax2.set_title(f'{title}: Time Series')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n💾 Saved plot to: {save_path}")
    
    return fig


def analyze_errors(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   df_test: pd.DataFrame = None) -> dict:
    """
    Analyze prediction errors.
    """
    print("\n   Analyzing errors...")
    
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    analysis = {
        'mean_error': errors.mean(),
        'mean_abs_error': abs_errors.mean(),
        'median_abs_error': np.median(abs_errors),
        'max_abs_error': abs_errors.max(),
        'std_error': errors.std()
    }
    
    print(f"      Mean Error: {analysis['mean_error']:.2f} (bias)")
    print(f"      Mean Absolute Error: {analysis['mean_abs_error']:.2f}")
    print(f"      Max Absolute Error: {analysis['max_abs_error']:.2f}")
    
    # Underpredictions (model missed outbreaks)
    underpredicted = np.where(errors > 100)[0]
    print(f"      Severe underpredictions (>100 cases): {len(underpredicted)}")
    
    return analysis


# =============================================================================
# Run if executed directly
# =============================================================================
if __name__ == "__main__":
    from data_preparation import run_data_preparation
    from feature_engineering import engineer_all_features
    from model_training import train_all_models
    
    # Full pipeline to evaluation
    df = run_data_preparation()
    df = engineer_all_features(df)
    training_results = train_all_models(df)
    
    # Evaluate
    comparison = compare_models(
        {'XGBoost': training_results['xgboost'], 'Random Forest': training_results['random_forest']},
        training_results['X_test'],
        training_results['y_test']
    )
    
    # Feature importance
    analyze_feature_importance(
        training_results['xgboost'],
        training_results['features']
    )
    
    print(f"\n✅ Evaluation complete!")
