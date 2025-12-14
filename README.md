# 🦟 Dengue Outbreak Prediction System

**CS-245 Machine Learning Course Project | NUST SEECS | Fall 2025**

A machine learning system that predicts dengue outbreaks 2 weeks ahead for all 17 regions of the Philippines. Uses historical case data, weather measurements from NASA, and satellite vegetation imagery.

---

## 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **XGBoost** | **31.67** | **69.70** | **0.788** |
| Random Forest | 32.48 | 76.33 | 0.746 |
| Baseline (Persistence) | 43.40 | 90.64 | 0.642 |
| Ridge Regression | 69.05 | 115.42 | 0.450 |

- **Training data**: 2016-2019 (3,298 samples)
- **Test data**: 2020 (850 samples)
- **Prediction horizon**: 2 weeks ahead

---

## �️ Setup Instructions

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib plotly streamlit requests
```

### Step 3: Verify Installation

```bash
python -c "import xgboost; import streamlit; print('All packages installed successfully!')"
```

---

## 🚀 Usage

### Run the Full Pipeline

This will process data, engineer features, train all models, and evaluate them:

```bash
python pipeline.py
```

### Skip Data Preparation (if already processed)

If you've already run the pipeline once and just want to retrain models:

```bash
python pipeline.py --skip-data
```

### Launch the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
dengue_project/
│
├── app.py              # Streamlit dashboard for predictions
├── config.py           # Configuration (paths, hyperparameters)
├── pipeline.py         # Main orchestration script
├── requirements.txt    # Python dependencies
│
├── src/                    # Core ML modules
│   ├── data_preparation.py     # Loads and merges raw data
│   ├── feature_engineering.py  # Creates 70+ features
│   ├── model_training.py       # Trains XGBoost, Random Forest, Ridge
│   ├── evaluation.py           # Calculates metrics, plots
│   └── predict.py              # Makes predictions with trained models
│
├── data/                   # Data files
│   ├── philippines_dengue.csv              # DOH dengue case reports
│   ├── weather/                            # Weather CSVs per region
│   ├── Philippines-Vegetation-*.csv        # NDVI satellite data
│   ├── philippines_dengue_dataset_FINAL.csv  # Merged raw data
│   └── dengue_dataset_engineered.csv       # Final feature set
│
├── models/                 # Saved trained models
│   ├── xgboost_best.joblib
│   ├── random_forest.joblib
│   ├── ridge.joblib
│   └── feature_list.txt
│
├── report/                 # LaTeX report
│   └── main.tex
│
└── Data Gathering/
    └── data_gathering.ipynb
```

---

## 🔧 Pipeline Overview

The pipeline runs in 4 phases:

### Phase 1: Data Preparation
- Loads dengue case data from DOH Philippines
- Fetches weather data from NASA POWER API
- Loads NDVI vegetation index from MODIS satellite
- Merges all sources by region and date

### Phase 2: Feature Engineering
Creates 70+ features including:
- **Lag features**: Weather and cases from 1-12 weeks ago
- **Rolling statistics**: 4/8/12-week means, std, max
- **Momentum**: Week-over-week changes, percentage changes
- **Interactions**: Rain×Humidity, Temp×Rain, Rain×NDVI
- **Cyclical encoding**: Sin/Cos for month and week

### Phase 3: Model Training
Trains three models with log-transformed target:
- **XGBoost**: 800 trees, learning rate 0.015
- **Random Forest**: 300 trees, max depth 15
- **Ridge Regression**: Cross-validated alpha, feature selection

### Phase 4: Evaluation
- Computes MAE, RMSE, R² on test set
- Generates feature importance analysis
- Creates prediction plots

---

## 🌐 Data Sources

| Source | Data | Coverage |
|--------|------|----------|
| DOH Philippines | Weekly dengue cases | 2016-2021, 17 regions |
| NASA POWER API | Temperature, rainfall, humidity | Daily → aggregated weekly |
| NASA MODIS | NDVI vegetation index | 16-day composites |

---

## 📈 Key Findings

- **Historical cases are the strongest predictor** (~70% of feature importance)
- **Weather features add ~27%** predictive power on top of case history
- **4-8 week lagged weather** works better than current weather (biological delay)
- **XGBoost outperforms** simpler models due to non-linear patterns in the data

---

## 👥 Authors

| Name | CMS ID |
|------|--------|
| Saqib Mehdi | 462682 |
| M. Shees ur Rehman | 470810 |

**Section**: BSCS-13-B  
**Instructor**: Mr. Usama Athar  
**Course**: CS-245 Machine Learning  
**Institution**: National University of Sciences and Technology (NUST), SEECS

---

## 📝 License

This project is for educational purposes as part of the CS-245 Machine Learning course at NUST.
