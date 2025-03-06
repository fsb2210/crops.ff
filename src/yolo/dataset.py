"""
Dataset
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PlantdocDataset(Dataset):
    def __init__(
        self, fname: Union[str, Path], image_dir: Union[str, Path], transform=None
    ) -> None:
        self.fname = fname
        self.image_dir = image_dir
        self.transform = transform

        # load dataframe with all the information:
        # filename|width|height|class|xmin|ymin|xmax|ymax
        self.df = pd.read_csv(fname)

    def __getitem__(self, index):
        img_name = Path(self.image_dir) / self.df.iloc[index, 0]

        # mask everything in dataframe that is not `img_name`
        df_m = self.df[self.df["filename"] == self.df.iloc[index, 0]]

        # get all bounding boxes in an image
        bboxes = []
        for row in df_m.iterrows():
            bbox = [row[1]["xmin"], row[1]["ymin"], row[1]["xmax"], row[1]["ymax"]]
            bboxes.append(bbox)

        bboxes = np.array(bboxes, dtype=float).reshape(len(bboxes), 4)
        return img_name, bboxes

    def __len__(self) -> int:
        return self.df.shape[0]

    def _bounding_box_to_yolo_format_(*bbox) -> Tuple[float, float, float, float]:
        return None
