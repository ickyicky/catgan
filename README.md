# CatGan

LSGan based neural network built for generating 64x64 cat images.

## Usage

Network is implemented as python module. It requires providing configuration file, which example can be found in `config.yaml` file. It shold be pretty self exmplanatory.

As a module catgan provides 3 main functions:

- `python3 -m catgan --train` - training the model on train data
- `python3 -m catgan --test` - test the model on test data
- `python3 -m catgan --generate` - generate batch of fake cat images

## Requirements

In order to run code you need to install all the required python librares. It is **highly** recommended that you install PyTorch as binary using your package manager in order to have CUDA support! Anyway, to install all required packages run:

```bash
pip3 install -r requirements.txt
```

## Dataset

For dataset catgan uses [github cat faces dataset](https://github.com/fferlito/Cat-faces-dataset.git). There are python scripts available for downloading and splitting the dataset into train and test sub datasets. To fetch and split dataset simply run:

```bash
python3 -m catgan.dataset --download
python3 -m catgan.dataset --split
```

This should result in creating `data/test` and `data/train` folders with cat images. By default, test dataset contains 20% of total cat images chosen by random. You can of course provide your own dataset, move/copy/delete files or decide how much data should be splitted into test folder when splitting by providing `--test-percentage` flag like follows:

```bash
python3 -m catgan.dataset --split --test-percentage=30
```

## References

Generator and discriminator are implementation (slightly changed to generate 64x64 images) of network proposed in [Least Squares Generative Adversarial Networks paper](https://arxiv.org/abs/1611.04076).

Dataset used during training: [Cat-faces-dataset](https://github.com/fferlito/Cat-faces-dataset.git), huge thanks to [fferlito](https://github.com/fferlito)

Dataset implemenetation was inspired by one found in [cat-gan](https://github.com/gmalivenko/cat-gan), transformation used in this repository saved me countless hours of fighting various training problems.
