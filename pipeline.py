"""
Dengue Prediction Pipeline - Main Orchestration
=================================================
This script runs the complete ML pipeline from data to model.

Usage:
    python pipeline.py              # Run full pipeline
    python pipeline.py --train      # Just train models
    python pipeline.py --predict    # Just make predictions
"""

import argparse
import os
import sys

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preparation import run_data_preparation, load_merged_data
from src.feature_engineering import engineer_all_features, save_engineered_data
from src.model_training import train_all_models, prepare_train_test
from src.evaluation import compare_models, analyze_feature_importance, plot_predictions
from src.predict import predict_all_regions, load_model

from config import MODEL_DIR, DATA_DIR


def run_full_pipeline(skip_data_prep: bool = False):
    """
    Run the complete ML pipeline.
    
    Phases:
    1. Data Preparation - Load and merge data sources
    2. Feature Engineering - Create ML features
    3. Model Training - Train XGBoost and RandomForest
    4. Evaluation - Calculate metrics and analyze
    5. Save - Persist models and results
    """
    print("\n" + "=" * 70)
    print("🦟 DENGUE PREDICTION PIPELINE")
    print("=" * 70)
    
    # =================================================================
    # PHASE 1: DATA PREPARATION
    # =================================================================
    if skip_data_prep:
        print("\n⏭️ Skipping data preparation (using existing data)...")
        df = load_merged_data()
    else:
        df = run_data_preparation()
    
    # =================================================================
    # PHASE 2: FEATURE ENGINEERING
    # =================================================================
    df = engineer_all_features(df)
    save_engineered_data(df)
    
    # =================================================================
    # PHASE 3: MODEL TRAINING
    # =================================================================
    training_results = train_all_models(df)
    
    # =================================================================
    # PHASE 4: EVALUATION
    # =================================================================
    models_to_compare = {
        'XGBoost': training_results['xgboost'],
        'Random Forest': training_results['random_forest'],
        'Ridge': training_results['ridge']
    }
    
    comparison = compare_models(
        models_to_compare,
        training_results['X_test'],
        training_results['y_test']
    )
    
    # Feature importance
    importance = analyze_feature_importance(
        training_results['xgboost'],
        training_results['features']
    )
    
    # Save comparison plot
    from src.evaluation import predict_with_model
    y_pred_xgb = predict_with_model(training_results['xgboost'], training_results['X_test'])
    plot_predictions(
        training_results['y_test'].values,
        y_pred_xgb,
        title='XGBoost',
        save_path=os.path.join(DATA_DIR, 'model_predictions.png')
    )
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)
    
    best_model = comparison.iloc[0]
    print(f"\n🏆 Best Model: {best_model['Model']}")
    print(f"   R² Score: {best_model['R2']:.4f}")
    print(f"   MAE: {best_model['MAE']:.2f}")
    
    print(f"\n📁 Outputs saved to:")
    print(f"   Models: {MODEL_DIR}")
    print(f"   Data: {DATA_DIR}")
    
    return {
        'comparison': comparison,
        'best_model': training_results['xgboost'],
        'features': training_results['features']
    }


def run_training_only():
    """Run only the training phase (assumes data exists)."""
    print("\n🚀 Running training only...")
    
    df = load_merged_data()
    df = engineer_all_features(df)
    
    training_results = train_all_models(df)
    
    print("\n✅ Training complete!")
    return training_results


def run_prediction_only():
    """Run only the prediction phase (assumes model exists)."""
    print("\n🔮 Running prediction only...")
    
    df = load_merged_data()
    df = engineer_all_features(df)
    model = load_model()
    
    predictions = predict_all_regions(df, model)
    
    print("\n📊 Predictions:")
    print(predictions.to_string(index=False))
    
    print("\n✅ Prediction complete!")
    return predictions


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dengue Prediction Pipeline')
    parser.add_argument('--train', action='store_true', help='Run training only')
    parser.add_argument('--predict', action='store_true', help='Run prediction only')
    parser.add_argument('--skip-data', action='store_true', help='Skip data preparation')
    
    args = parser.parse_args()
    
    if args.train:
        run_training_only()
    elif args.predict:
        run_prediction_only()
    else:
        run_full_pipeline(skip_data_prep=args.skip_data)
