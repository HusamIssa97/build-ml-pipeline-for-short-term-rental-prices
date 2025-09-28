#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from mlflow.models.signature import infer_signature

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() -d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def plot_feature_importance(pipe, feat_names):
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names)-1]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(args, rf_config, X_train, y_train):

    # Define feature groups
    zero_imputed_columns = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month", 
        "calculated_host_listings_count",
        "availability_365"
    ]
    ordinal_categorical = ["room_type"]
    nominal_categorical = ["neighbourhood_group"]
    
    # Create feature processing transformers
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
    ordinal_categorical_preproc = OrdinalEncoder()

    # Create a pipeline for date processing
    date_pipeline = Pipeline([
        ("date_imputer", SimpleImputer(strategy='constant', fill_value='2010-01-01')),
        ("date_transformer", FunctionTransformer(delta_date_feature, validate=False, check_inverse=False))
    ])

    # Create the tfidf_title step
    tfidf_title = TfidfVectorizer(
        binary=False, max_features=args.max_tfidf_features, stop_words='english'
    )

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("nominal_cat", OneHotEncoder(), nominal_categorical),
            ("zero_imputed", zero_imputer, zero_imputed_columns),
            ("date_processed", date_pipeline, ["last_review"]),
            ("tfidf_title", tfidf_title, "name")
        ],
        remainder="drop"  # This drops the columns that we do not transform
    )

    processed_features = ordinal_categorical + nominal_categorical + zero_imputed_columns + ["last_review", "name"]

    # Create random forest
    random_forest = RandomForestRegressor(**rf_config)

    # Create pipeline
    sk_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("random_forest", random_forest)
    ])

    return sk_pipe, processed_features


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config['random_state'] = args.random_seed

    # Use run.use_artifact(...).file() to get the train and validation artifact
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")  # this removes the column "price" from X and puts it into y

    # COMPREHENSIVE DATA CLEANING
    logger.info("Performing comprehensive data cleaning")
    
    # 1. Clean text data
    X['name'] = X['name'].fillna('Unknown')
    X['name'] = X['name'].astype(str)
    
    # 2. Clean all numeric columns
    numeric_cols = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count", 
        "availability_365"
    ]
    for col in numeric_cols:
        X[col] = X[col].fillna(0)
    
    # 3. Clean date column
    X['last_review'] = X['last_review'].fillna('2010-01-01')
    
    # 4. Clean categorical columns
    X['room_type'] = X['room_type'].fillna('Unknown')
    X['neighbourhood_group'] = X['neighbourhood_group'].fillna('Unknown')
    
    # 5. Remove any remaining rows with missing critical data
    mask = (
        (X['name'].str.strip() != '') & 
        (X['room_type'] != '') & 
        (X['neighbourhood_group'] != '')
    )
    X = X[mask].copy()
    y = y[mask].copy()
    
    logger.info(f"Dataset size after cleaning: {len(X)} rows")
    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    # Check for any remaining NaN values
    nan_counts = X.isnull().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Remaining NaN values: {nan_counts[nan_counts > 0]}")
        # Fill any remaining NaN with appropriate defaults
        X = X.fillna({
            'name': 'Unknown',
            'last_review': '2010-01-01',
            'room_type': 'Unknown',
            'neighbourhood_group': 'Unknown'
        })
        # Fill numeric columns with 0
        for col in numeric_cols:
            X[col] = X[col].fillna(0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(args, rf_config, X_train, y_train)

    # Fit the pipeline
    logger.info("Fitting")
    sk_pipe.fit(X_train, y_train)

    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info("Score: %s", r_squared)
    logger.info("MAE: %s", mae)

    # Save model
    signature = infer_signature(X_val, y_pred)
    
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    mlflow.sklearn.save_model(
        sk_pipe,
        "random_forest_dir",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=signature,
        input_example=X_val.iloc[:2],
    )

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    # Upload to W&B
    run.log({
        "r2": r_squared,
        "mae": mae
    })

    run.log({"feature_importance": wandb.Image(fig_feat_imp)})

    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Trained Random Forest model"
    )

    artifact.add_dir("random_forest_dir")
    run.log_artifact(artifact)

    # Save feature importance plot
    fig_feat_imp.savefig("feature_importance.png")

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words to consider for the TFIDF",
        default=10,
        type=int
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)