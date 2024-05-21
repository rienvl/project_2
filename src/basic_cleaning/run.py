#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info('OK - Fetched artifact %s', args.input_artifact)

    # read dataframe
    df = pd.read_csv(artifact_local_path)
    logger.info('OK - Read dataframe with %d rows', df.shape[0])

    # data cleaning derived from EDA step
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info('OK - basic cleaning of dataframe')

    # write dataframe to local csv file
    local_file = f"{args.output_artifact}"
    df.to_csv(local_file, index=False)  # remove index column, otherwise data checks will fail

    # create wandb artifact
    output_artifact = wandb.Artifact(name=args.output_artifact,
                                     type=args.output_type,
                                     description=args.output_description)

    # attach local csv file to artifact
    output_artifact.add_file(local_file)

    # upload output artifact to wandb
    run.log_artifact(output_artifact)

    run.finish()
    logger.info('OK - Logged artifact %s', args.output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--output_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price used to remove outliers",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price used to remove outliers",
        required=True,
    )

    args = parser.parse_args()

    go(args)
