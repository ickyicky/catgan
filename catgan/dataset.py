import argparse
import os
from random import choice
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

    args = parser.parse_args()

    if args.download:
        from git import Repo

        Repo.clone_from(DATASET_URL, "data_repo")

        for tar_file_name in glob("data_repo/*.tar.gz"):
            with tarfile.open(tar_file_name) as file:
                file.extractall("data")

    if args.split:
        all_files = os.listdir("data")

        test_files = random.choice(
            all_files, k=int(args.test_percentage / 100 * len(all_files))
        )
        train_files = [fname for fname in all_files if fname not in test_files]

        os.mkdir("data/train")
        os.mkdir("data/test")

        for fname in test_files:
            os.rename(
                fname,
                os.path.join("data", "test", os.path.basename(fname)),
            )

        for fname in train_files:
            os.rename(
                fname,
                os.path.join("data", "train", os.path.basename(fname)),
            )
