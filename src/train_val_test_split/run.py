#!/usr/bin/env python
import argparse
import logging
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    logger.info("Downloading input artifact")
    artifact_local_path = run.use_artifact(args.input).file()

    logger.info("Reading input artifact")
    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != "none" else None,
    )

    logger.info("Uploading test_data.csv dataset")
    test.to_csv("test_data.csv", index=False)
    artifact = wandb.Artifact("test_data.csv", type="test_data", description="Test dataset")
    artifact.add_file("test_data.csv")
    run.log_artifact(artifact)

    logger.info("Uploading trainval_data.csv dataset")
    trainval.to_csv("trainval_data.csv", index=False)
    artifact = wandb.Artifact("trainval_data.csv", type="trainval_data", description="Train and validation dataset")
    artifact.add_file("trainval_data.csv")
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")
    parser.add_argument("--input", type=str, help="Input artifact to split", required=True)
    parser.add_argument("--test_size", type=float, help="Size of the test split")
    parser.add_argument("--random_seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("--stratify_by", type=str, help="Column to use for stratification", default="none")
    
    args = parser.parse_args()
    go(args)