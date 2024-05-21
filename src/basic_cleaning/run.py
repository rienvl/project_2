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

    # remove Nans
    df.fillna({'last_review': '2000-01-01'}, inplace=True)
    df.fillna({'reviews_per_month': 0}, inplace=True)
    logger.info('OK - basic cleaning of dataframe')

    # write dataframe to local csv file
    local_file = f"{args.artifact_name}"
    df.to_csv(local_file)

    # create wandb artifact
    output_artifact = wandb.Artifact(name=args.artifact_name, type=args.artifact_type, description=args.artifact_description)

    # attach local csv file to artifact
    output_artifact.add_file(local_file)

    # upload output artifact to wandb
    run.log_artifact(output_artifact)

    run.finish()
    logger.info('OK - Logged artifact %s', args.artifact_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
