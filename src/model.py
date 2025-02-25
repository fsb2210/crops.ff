"""
Neural network model
"""

from typing import Dict, List, Tuple

from torch import nn


class EmptyLayer(nn.Module):
    """Placeholder layer, no operations yet"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    """Placeholder layer, no operations yet"""
    def __init__(self, anchors: List[Tuple]) -> None:
        super(DetectionLayer, self).__init__()
        if not isinstance(anchors, list):
            raise TypeError(f"anchors option must be a list, got {type(anchors)}")
        self.anchors = anchors

    def extra_repr(self):
        """Print extra parameters of the custom module"""
        return f"anchors = {self.anchors}"

class YOLOv3(nn.Module):
    """YOLOv3 neural network

    Parameters
    ----------
    configs : `dict`
        Dictionary with configs for each layer of the YOLOv3 model
    """

    def __init__(self, configs: Dict = dict()) -> None:
        super(YOLOv3, self).__init__()
        self.configs = configs

        # save global opts in this dict
        self.net_info = None

        # create layers in YOLO as a nn.Sequential
        _layers = nn.Sequential()
        for k, lname in enumerate(self.configs.keys()):
            layer_opts = self.configs[lname]
            if lname == "global":
                self.net_info = self.configs[lname]
                continue
            for ltype in layer_opts.keys():
                lyr = None
                name = ""
                if ltype == "Conv2d":
                    name = f"conv_{k}"
                    lyr = nn.Conv2d(**layer_opts[ltype])
                    # module.add_module("conv_{0}".format(index), conv)
                elif ltype == "BatchNorm2d":
                    name = f"bn_{k}"
                    lyr = nn.BatchNorm2d(**layer_opts[ltype])
                elif ltype == "Activation":
                    name = f"leaky_{k}"
                    fn_name = layer_opts[ltype]["fn"]
                    if fn_name == "leaky":
                        lyr = nn.LeakyReLU(
                            negative_slope=layer_opts[ltype]["negative_slope"],
                            inplace=layer_opts[ltype]["inplace"],
                        )
                elif ltype == "Upsample":
                    name = f"upsample_{k}"
                    lyr = nn.Upsample(**layer_opts[ltype])
                elif ltype == "Route":
                    name = f"route_{k}"
                    lyr = EmptyLayer()
                elif ltype == "Shortcut":
                    name = f"shortcut_{k}"
                    lyr = EmptyLayer()
                elif ltype == "YOLO":
                    name = f"detection_{k}"
                    lyr = DetectionLayer(**layer_opts[ltype])
                else:
                    raise ValueError(f"unknown layer for YOLO structure: {ltype}")

                if lyr is not None: 
                    _layers.add_module(name, lyr)

        self.layers = _layers

    def forward(self, x):
        pass
