"""
Run neural network modelling
"""

import argparse

import torch
from model import YOLOv3


def main(opts: argparse.Namespace) -> None:
    """main

    Parameters
    ----------
    opts : `argparse.Namespace`
        Command-line options
    """

    # load YOLOv3 model
    model = YOLOv3(in_channels=opts.channels, num_classes=opts.num_classes)

    # example
    IMAGE_SIZE = 416
    x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    print(type(out))
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

    # Asserting output shapes
    assert model(x)[0].shape == (
        1,
        3,
        IMAGE_SIZE // 32,
        IMAGE_SIZE // 32,
        opts.num_classes + 5,
    )
    assert model(x)[1].shape == (
        1,
        3,
        IMAGE_SIZE // 16,
        IMAGE_SIZE // 16,
        opts.num_classes + 5,
    )
    assert model(x)[2].shape == (
        1,
        3,
        IMAGE_SIZE // 8,
        IMAGE_SIZE // 8,
        opts.num_classes + 5,
    )
    print("Output shapes are correct!")

    return None


if __name__ == "__main__":
    # command line arguments
    args = argparse.ArgumentParser()
    # args.add_argument(
    #     "--config-file", "-C", default="", help="name of file with configuration options",
    # )
    # args.add_argument(
    #     "--dataset-directory",
    # )
    # args.add_argument(
    #     "--split_fraction", default=0.8, help="fraction of elements in each dataset",
    # )
    args.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Channels in input image",
    )
    args.add_argument(
        "--num-classes",
        type=int,
        default=20,
        help="Number of classes in dataset",
    )
    args.add_argument(
        "--mode",
        "-M",
        type=str,
        default="inference",
        help="Mode for the neural network",
    )
    args.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs for the training phase",
    )
    args.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the training phase",
    )
    opts = args.parse_args()

    # main entry point
    main(opts)
