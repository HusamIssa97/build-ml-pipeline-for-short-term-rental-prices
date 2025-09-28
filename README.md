# ML Pipeline for NYC Airbnb Price Prediction v1.0.0

## Overview
Complete MLOps pipeline for predicting short-term rental prices using NYC Airbnb data. This pipeline processes raw data, trains a Random Forest model, and provides production-ready predictions.

## What's Included
- **Data Processing**: Automated cleaning and validation
- **Model Training**: Random Forest with hyperparameter optimization  
- **Testing**: Comprehensive data and model validation
- **Deployment**: Production-ready model artifacts

## Key Features
- End-to-end MLflow pipeline
- Weights & Biases experiment tracking
- Automated hyperparameter tuning
- Data quality validation tests
- Reproducible model training

## Model Performance
- **Best MAE**: ~35.78 (from hyperparameter optimization)
- **Features**: 60 experimental runs completed
- **Optimization**: TF-IDF features (10,15,30) x Random Forest max_features (0.1,0.33,0.5,0.75,1.0)

## Quick Start
```bash
# Run the complete pipeline
mlflow run https://github.com/HusamIssa97/build-ml-pipeline-for-short-term-rental-prices.git -v 1.0.0

# Run with new data
mlflow run https://github.com/HusamIssa97/build-ml-pipeline-for-short-term-rental-prices.git -v 1.0.0 -P hydra_options="etl.sample='sample2.csv'"
```

## Pipeline Steps
1. **Download** - Fetch raw data
2. **Clean** - Remove outliers and handle missing values  
3. **Validate** - Run data quality tests
4. **Split** - Create train/validation/test sets
5. **Train** - Random Forest with feature engineering
6. **Test** - Validate model performance

## Requirements
- Python 3.13
- MLflow 3.3.2
- Weights & Biases account
- Conda environment manager

## Components
All pipeline components are containerized with proper dependency management for reproducible execution across different environments.