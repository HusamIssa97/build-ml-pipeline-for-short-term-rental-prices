#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args) if hasattr(args, "__dict__") else args)

    logger.info(f"Fetching input artifact: {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading raw dataset")
    df = pd.read_csv(artifact_local_path)

    # Drop price outliers using provided bounds
    logger.info(f"Filtering price between {args.min_price} and {args.max_price}")
    price_mask = df["price"].between(args.min_price, args.max_price)
    df = df.loc[price_mask].copy()

    # Convert last_review to datetime (do not impute here)
    logger.info("Converting 'last_review' to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    
    logger.info("Filtering geolocation boundaries")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    
    df = df[idx].copy()

    # Save cleaned data
    output_filename = "clean_sample.csv"
    logger.info(f"Saving cleaned data to {output_filename}")
    df.to_csv(output_filename, index=False)

    # Log cleaned artifact to W&B
    logger.info(
        f"Logging artifact '{args.output_artifact}' "
        f"(type='{args.output_type}') to Weights & Biases"
    )
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_filename)
    run.log_artifact(artifact)
    artifact.wait()

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact to download (e.g., sample.csv:latest)",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact to create (e.g., clean_sample.csv)",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact (e.g., clean_sample)",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="A brief description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to consider when filtering outliers",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to consider when filtering outliers",
        required=True
    )

    args = parser.parse_args()
    go(args)

