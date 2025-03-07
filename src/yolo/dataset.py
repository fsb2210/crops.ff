"""
Dataset
"""

import math
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import cv2

class PlantdocDataset(Dataset):
    def __init__(
            self, df: pd.DataFrame, image_dir: Union[str, Path], image_size: int, labels_dict: dict, transform=None,
    ) -> None:
        # dataframe with all the information:
        # filename|width|height|class|xmin|ymin|xmax|ymax
        self.df = df
        self.image_dir = image_dir
        self.image_size = image_size
        self.labels_dict = labels_dict
        self.transform = transform

    def __getitem__(self, index):
        img_full_name = Path(self.image_dir) / self.df.iloc[index, 0]

        # load image
        img = self.load_image(name=img_full_name)

        # mask everything in dataframe that is not `img_name`
        df_m = self.df[self.df["filename"] == self.df.iloc[index, 0]]

        # get all bounding boxes in an image
        bboxes = []
        for row in df_m.iterrows():
            bbox = self.convert_to_yolo(row[1]["xmin"], row[1]["ymin"], row[1]["xmax"], row[1]["ymax"], row[1]["width"], row[1]["height"])
            bboxes.append(bbox)

        # bboxes = np.array(bboxes, dtype=float).reshape(len(bboxes), 4)
        img = img.transpose(2, 0, 1)[::-1]
        img = np.ascontiguousarray(img)

        # handle labels
        label = self.labels_dict[self.df.iloc[index,3]]


        return torch.from_numpy(img), bboxes, label

    def __len__(self) -> int:
        return self.df.shape[0]

    def load_image(self, name):
        im = cv2.imread(name) # BGR
        h0, w0 = im.shape[:2]
        r = self.image_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
        return im

    def convert_to_yolo(
            self,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float, float, float]:
        """Convert format to YOLO"""
        # calculate box width and height
        width = xmax - xmin
        height = ymax - ymin
        # calculate center coordinates
        x_center = xmin + (width / 2)
        y_center = ymin + (height / 2)
        # normalize coordinates to image dimensions
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height = image_height
        return x_center, y_center, width, height
