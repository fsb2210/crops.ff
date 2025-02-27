"""
Configuration options for every layer of a YOLOv3 neural network
"""

from pathlib import Path
from typing import Dict, List, Union


class YOLOv3Cfg:
    """Config object for YOLO neural network

    Parameters
    ----------
    fname : `str / Path`
        Filename with config of YOLO network
    """

    def __init__(self, fname: Union[str, Path] = "") -> None:
        self.fname = fname
        self.opts: dict = dict()
        _blocks = self.parse_file()
        self._block_generator(blocks=_blocks)

    def parse_file(self) -> List:
        """Parses config file from YOLO darknet network into a list"""

        if self.fname == "" or self.fname is None:
            raise ValueError(f"unknown cfg filename: {self.fname}")

        # load cfg file
        file = open(str(self.fname), "r")
        lines = file.read().split("\n")
        lines = [x for x in lines if len(x) > 0]
        lines = [x for x in lines if x[0] != "#"]
        lines = [x.rstrip().lstrip() for x in lines]

        # return a list with block configs
        blocks: list = []
        block = dict()
        for line in lines:
            if line[0] == "[":
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()
            else:
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)

        return blocks

    def _block_generator(self, blocks: List[Dict] = []) -> None:
        """Generate blocks in YOLO architecture

        Parameters
        ----------
        blocks : `list`
            List with every single block of a YOLOv3 model
        """

        if len(blocks) <= 0:
            raise ValueError(
                "got an unknown block for generating YOLO architecture configuration"
            )

        # controls in_channels of Conv2d layers
        prev_filters = 3
        output_filters = []
        filters = -1

        # save config in a dictionary
        nn_info = dict()
        nn_info["global"] = blocks[0]

        # loop over cfg list
        for k, blk in enumerate(blocks[1:]):
            nn_info[f"layer_{k}"] = dict()
            # different blocks have different options
            if blk["type"] == "convolutional":
                activation_fn = blk["activation"]
                # check if conv layer has batchnorm
                try:
                    batchnorm = int(blk["batch_normalize"])
                    bias = False
                except Exception:
                    batchnorm = 0
                    bias = True

                # this are options for a conv layer
                filters = int(blk["filters"])
                padding = int(blk["pad"])
                kernel_size = int(blk["size"])
                stride = int(blk["stride"])
                pad = 0
                if padding:
                    pad = (kernel_size - 1) // 2

                conv_dict = {
                    "in_channels": prev_filters,
                    "out_channels": filters,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "padding": pad,
                    "bias": bias,
                }

                batchnorm_dict = {
                    "num_features": filters,
                }

                activation_dict = dict()
                if len(activation_fn) > 0:
                    activation_dict["fn"] = activation_fn
                    activation_dict["negative_slope"] = 0.1
                    activation_dict["inplace"] = True

                nn_info[f"layer_{k}"]["Conv2d"] = conv_dict
                if batchnorm:
                    nn_info[f"layer_{k}"]["BatchNorm2d"] = batchnorm_dict
                if len(activation_dict) > 0:
                    nn_info[f"layer_{k}"]["Activation"] = activation_dict

            elif blk["type"] == "upsample":
                stride = blk["stride"]
                nn_info[f"layer_{k}"]["Upsample"] = {
                    "scale_factor": 2,
                    "mode": "bilinear",
                }

            elif blk["type"] == "route":
                blk["layers"] = blk["layers"].split(",")
                # start  of a route
                start = int(blk["layers"][0])
                # end, if there exists one.
                try:
                    end = int(blk["layers"][1])
                except Exception:
                    end = 0
                # positive anotation
                if start > 0:
                    start = start - k
                if end > 0:
                    end = end - k
                if end < 0:
                    filters = output_filters[k + start] + output_filters[k + end]
                else:
                    filters = output_filters[k + start]

                nn_info[f"layer_{k}"]["Route"] = {
                    "empty_layer": True,
                }

            # shortcut corresponds to skip connection
            elif blk["type"] == "shortcut":
                nn_info[f"layer_{k}"]["Shortcut"] = {
                    "empty_layer": True,
                }

            elif blk["type"] == "yolo":
                mask = blk["mask"].split(",")
                mask = [int(x) for x in mask]

                anchors = blk["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [
                    (anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)
                ]
                anchors = [anchors[i] for i in mask]

                nn_info[f"layer_{k}"]["YOLO"] = {
                    "anchors": anchors,
                }

            # update filters for next step in iteration
            prev_filters = filters
            output_filters.append(filters)

        # opts contains everything required to make a YOLOv3 neural network
        self.opts = nn_info
