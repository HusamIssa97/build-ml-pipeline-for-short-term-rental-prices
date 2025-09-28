# NYC Airbnb Price Prediction Pipeline

MLOps pipeline for predicting short-term rental prices using machine learning.

## Links
- **W&B Project**: https://wandb.ai/k12352655-johannes-kepler-universit-t-linz/nyc_airbnb
- **GitHub Repository**: https://github.com/HusamIssa97/build-ml-pipeline-for-short-term-rental-prices

## Quick Start

### Run Complete Pipeline
```bash
mlflow run https://github.com/HusamIssa97/build-ml-pipeline-for-short-term-rental-prices.git -v 1.0.0
```

### Run with New Data
```bash
mlflow run https://github.com/HusamIssa97/build-ml-pipeline-for-short-term-rental-prices.git -v 1.0.0 -P hydra_options="etl.sample='sample2.csv'"
```

## What It Does

1. **Downloads** NYC Airbnb data
2. **Cleans** data (removes outliers, handles missing values)
3. **Validates** data quality with automated tests
4. **Splits** data into train/validation/test sets
5. **Trains** Random Forest model with feature engineering
6. **Tests** final model performance

## Model Results

- **Algorithm**: Random Forest Regressor
- **Best MAE**: ~35.78 USD
- **Features**: Property details, location, reviews, and text analysis
- **Optimization**: 60 hyperparameter experiments completed

## Requirements

- Python 3.13
- Conda
- Weights & Biases account

## Setup

```bash
# Clone repository
git clone https://github.com/HusamIssa97/build-ml-pipeline-for-short-term-rental-prices.git
cd build-ml-pipeline-for-short-term-rental-prices

# Create environment
conda env create -f environment.yml
conda activate nyc_airbnb_dev

# Login to W&B
wandb login [your-api-key]

# Run pipeline
mlflow run .
```

## Project Structure

```
├── main.py                 # Pipeline orchestration
├── config.yaml            # Configuration settings
├── src/
│   ├── basic_cleaning/     # Data cleaning component
│   ├── data_check/         # Data validation tests
│   ├── train_val_test_split/ # Data splitting
│   └── train_random_forest/ # Model training
└── environment.yml        # Dependencies
```

## Pipeline Components

- **Data Cleaning**: Removes price outliers ($10-$350), handles missing values
- **Data Validation**: Tests data quality, distribution, and boundaries
- **Feature Engineering**: TF-IDF on property names, date processing, categorical encoding
- **Model Training**: Random Forest with automated hyperparameter tuning
- **Model Testing**: Performance validation on held-out test set

## Reproducibility

All pipeline runs are tracked in Weights & Biases with:
- Hyperparameters
- Metrics (MAE, R²)
- Model artifacts
- Feature importance plots
- Data lineage

Built for production deployment and retraining workflows.