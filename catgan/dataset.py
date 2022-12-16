import argparse
import os
import shutil
from random import sample
from glob import glob
import tarfile


DATASET_URL = "https://github.com/fferlito/Cat-faces-dataset.git"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--download", action="store_true", help="download dataset"
    )
    parser.add_argument(
        "-s",
        "--split",
        action="store_true",
        help="split dataset into training and testing one",
    )
    parser.add_argument(
        "-t",
        "--test-percentage",
        action="store",
        type=int,
        default=20,
        help="percent of data splitted into testing data",
    )
    parser.add_argument(
        "-v",
        "--valid-percentage",
        action="store",
        type=int,
        default=25,
        help="percent of data splitted into validating data",
    )

    args = parser.parse_args()

    if args.download:
        from git import Repo

        Repo.clone_from(DATASET_URL, "data_repo")

        for tar_file_name in glob("data_repo/*.tar.gz"):
            with tarfile.open(tar_file_name) as file:
                file.extractall("data")

        shutil.rmtree("data_repo")

    if args.split:
        all_files = glob("data/dataset-part*/*.png")

        test_files = sample(
            all_files, k=int(args.test_percentage / 100 * len(all_files))
        )

        os.mkdir("data/train")
        os.mkdir("data/test")
        os.mkdir("data/valid")

        for fname in test_files:
            os.rename(
                fname,
                os.path.join("data", "test", os.path.basename(fname)),
            )

        all_files = glob("data/dataset-part*/*.png")
        valid_files = sample(
            all_files, k=int(args.valid_percentage / 100 * len(all_files))
        )

        for fname in valid_files:
            os.rename(
                fname,
                os.path.join("data", "valid", os.path.basename(fname)),
            )

        train_files = glob("data/dataset-part*/*.png")
        for fname in train_files:
            os.rename(
                fname,
                os.path.join("data", "train", os.path.basename(fname)),
            )

        for folder in glob("data/dataset-part*"):
            shutil.rmtree(folder)
