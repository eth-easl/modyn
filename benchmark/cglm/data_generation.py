import argparse
import logging
import os
import json
import pathlib
import shutil
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description=f"CGLM Benchmark Data Generation")
    parser_.add_argument(
        "sourcedir", type=pathlib.Path, action="store", help="Path to source data directory"
    )
    parser_.add_argument(
        "output", type=pathlib.Path, action="store", help="Path where we will output the dataset"
    )

    parser_.add_argument(
        "--metadata", type=pathlib.Path, default="cglm_labels_timestamps_clean.csv", action="store", help="The path to the csv file containing the metadata"
    )

    parser_.add_argument(
        "--dummy",
        action="store_true",
        help="Add a final dummy item in the fair future to train also on the last trigger in Modyn",
    )

    parser_.add_argument(
        "--min_samples_per_class",
        type=int,
        default=25,
        action="store",
        help="Classes that have less than `min_samples_per_class` samples will be deleted. Defaults to 25.",
    )

    parser_.add_argument(
        "--clean",
        action="store_true",
        help="If given, the GLM cleaned subset is used. Check out the repo for more details on how this was generated.",
    )

    parser_.add_argument(
        "--labeltype",
        default="landmark",
        help="Which label type should be used. `landmark` are the default labels from the CGLM paper, and `supercategory` and `hierarchical` are higher order labels provided by Google.",
        choices=["landmark", "supercategory", "hierarchical"]
    )

    parser_.add_argument(
        "--eval_split",
        type=str,
        choices=["uniform", "yearly"],
        help="Generate an evaluation split. 'uniform' samples 10% uniformly at random, 'yearly' samples 10% from each year.",
    )

    return parser_


def main():
    parser = setup_argparser()
    args = parser.parse_args()
    identifier = f"cglm{'_clean' if args.clean else ''}_{args.labeltype}_min{args.min_samples_per_class}"
    logger.info(f"Final destination is {args.output / identifier}. Generating subset to use. Identifier is {identifier}.")

    label_type_to_column = {"landmark": 'landmark_id', "supercategory": "supercategory_label", "hierarchical": "hierarchical_label_label"}
    label_column = label_type_to_column[args.labeltype]

    df = pd.read_csv(args.metadata)
    df = df[df["clean"] == args.clean]
    df = df[df[label_column].notnull()]
    label_counts = df[label_column].value_counts()
    label_ids_to_drop = label_counts[label_counts < args.min_samples_per_class].index
    df = df[~df[label_column].isin(label_ids_to_drop)]

    label_to_new_label = {old_label: label for label, old_label in enumerate(df[label_column].unique())}
    df['label'] = df[label_column].map(label_to_new_label)
    df["year"] = df["upload_date"].apply(lambda x: datetime.fromtimestamp(x).year)
    print(f"We got {df.shape[0]} samples with {len(label_to_new_label)} classes for this configuration. Generating subset.")

    if args.eval_split:
        if args.eval_split == "uniform":
            train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
        elif args.eval_split == "yearly":
            train_dfs, eval_dfs = [], []
            for _year, group in df.groupby("year"):
                train_group, eval_group = train_test_split(group, test_size=0.1, random_state=42)
                train_dfs.append(train_group)
                eval_dfs.append(eval_group)
            train_df = pd.concat(train_dfs)
            eval_df = pd.concat(eval_dfs)

        loop_iterator = [("train", train_df, args.output / identifier / "train"), ("eval", eval_df, args.output / identifier / "eval")]
    else:
        loop_iterator = [("train", df, args.output / identifier / "train")]
    
    overall_stats = {}
    for split, split_df, output_dir in loop_iterator:
        split_stats = {"total_samples": 0, "total_classes": len(label_to_new_label), "per_year": {}, "per_class": {}, "per_year_and_class": {}}

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {split} split with {len(split_df)} samples...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            id = row["id"]
            label = row["label"]
            timestamp = row["upload_date"]
            file_path = args.sourcedir / f"{id[0]}/{id[1]}/{id[2]}/{id}.jpg"

            # Generate stats
            year = row["year"]
            split_stats["total_samples"] += 1
            if year not in split_stats["per_year"]:
                split_stats["per_year"][year] = 1
            else:
                split_stats["per_year"][year] += 1

            if label not in split_stats["per_class"]:
                split_stats["per_class"][label] = 1
            else:
                split_stats["per_class"][label] += 1

            if year not in split_stats["per_year_and_class"]:
                split_stats["per_year_and_class"][year] = {}
                split_stats["per_year_and_class"][year][label] = 1
            else:
                if label not in split_stats["per_year_and_class"][year]:
                    split_stats["per_year_and_class"][year][label] = 1
                else:
                    split_stats["per_year_and_class"][year][label] += 1

            # Create files
            if not file_path.exists():
                logger.error(f"File {file_path} is supposed to exist, but it does not. Skipping...")
                continue

            destination_path = output_dir / f"{id}.jpg"

            if destination_path.exists():
                logger.error(f"File {destination_path} already exists. Skipping...")
                continue

            shutil.copy(file_path, destination_path)
            os.utime(destination_path, (timestamp, timestamp))

            with open(output_dir / f"{id}.label", "w", encoding="utf-8") as file:
                file.write(str(int(label)))
            os.utime(output_dir / f"{id}.label", (timestamp, timestamp))

        overall_stats[split] = split_stats

    with open(args.output / identifier / "dataset_stats.json", "w") as f:
        json.dump(overall_stats, f, indent=4)


    if args.dummy:
        dummy_path =  args.output / identifier / "train" / "dummy.jpg"
        shutil.copy(file_path, dummy_path) # just use the last file_path
        dummy_ts = timestamp + 24 * 60 * 60 * 1000
        os.utime(dummy_path, (dummy_ts, dummy_ts))
        with open(args.output / identifier / "train" / "dummy.label", "w", encoding="utf-8") as file:
            file.write(str(int(label)))
        os.utime(args.output / identifier / "train" / "dummy.label", (dummy_ts, dummy_ts))


    logger.info("Done.")


if __name__ == "__main__":
    main()


