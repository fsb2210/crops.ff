"""
Create annotations for each image and store them in a proper directory
as required by YOLO & detectron2 models
"""

import argparse
import glob
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import yaml


def convert_to_yolo(
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
    height /= image_height
    return x_center, y_center, width, height


def annotate_dataset(
    opt: argparse.Namespace, df: pd.DataFrame, ds_id: str, classes_dict: dict
) -> None:
    """Annotate dataset and save annotations to new database"""
    # loop over elements in the datase.txtt
    for k, row in df.iterrows():
        if not opt.verbose:
            print(f"\r  - Annotating {k+1}/{df.shape[0]}", end=" ")

        rclass = row["class"]
        rclassid = next((k for k, v in classes_dict.items() if v == rclass), None)
        rw, rh = int(row["width"]), int(row["height"])
        rxmin, rxmax = int(row["xmin"]), int(row["xmax"])
        rymin, rymax = int(row["ymin"]), int(row["ymax"])

        bbox = convert_to_yolo(rxmin, rymin, rxmax, rymax, rw, rh)

        ann_str = (
            f"{rclassid:2d} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n"
        )
        if opt.verbose:
            print(f"  - Annotation for {row['filename']}: >>> {ann_str}")
        with open(
            f"{opt.root_dataset_dir}/labels/{ds_id}/{row['filename'][:-4]}.txt", mode="a"
        ) as f:
            f.write(ann_str)
    print()
    return


def copy_images(
    src_dir: Union[str, Path], dest_dir: Union[str, Path], fnames: List[str],
    verbose: bool = False,
) -> None:
    """Copy images to new database location

    Parameters
    ----------
    src_dir : `str / Path`
        Source directory

    dest_dir : `str / Path`
        Destionation directory

    fnames : `list`
        List of files to copy from src_dir to dest_dir
    """
    for k, fname in enumerate(fnames):
        print(f"\r  - Copying images to dataset {k+1}/{len(fnames)}", end=" ")
        shutil.copyfile(f"{src_dir}/{fname}", f"{dest_dir}/{fname}")
    print()
    return


def clean_dataframe(
    df: pd.DataFrame, image_directory: Union[str, Path],
    verbose: bool = False,
) -> pd.DataFrame:
    """Clean dataframe from images not present in the dataset

    Parameters
    ----------
    df : `pd.DataFrame`
        Dataframe of the dataset

    image_directory : `str / Path`
        Directory with images of the dataset

    verbose : `bool`
        Add verbosity to standard output
    """
    if verbose: print("  - Cleaning dataframe from images not present in dataset and images with zero width or height")
    images_in_ds = glob.glob("*.*g", root_dir=image_directory)
    keys_to_drop, images_to_drop = [], []
    for k, row in df.iterrows():
        # aux vars
        fname = row["filename"]
        width, height = row["width"], row["height"]

        if fname not in images_in_ds:
            if fname not in images_to_drop:
                print(
                    f"WARNING: {fname} not found in {image_directory}, will skip it"
                )
            keys_to_drop.append(k)
            images_to_drop.append(fname)

        if width == 0 or height == 0:
            if fname not in images_to_drop:
                print(f"WARNING: found 0 dimension(s) in {fname}: w={width:.2f} h={height:.2f}. will skip it")
            keys_to_drop.append(k)
            images_to_drop.append(fname)
    new_df = df.drop(keys_to_drop)
    new_df = new_df.reset_index(drop=True)
    return new_df

def _get_dict_classes(fname: Union[str, Path]) -> Dict[int,str]:
    _df = pd.read_csv(filepath_or_buffer=fname)
    classes = list(set(_df["class"]))
    i2c = {k: name for k, name in enumerate(classes)}
    return i2c


def main(opt: argparse.Namespace) -> None:
    """Make YOLO-supported annotations

    Parameters
    ----------
    opt : `argparse.Namespace`
        Options from the command line
    """

    if opt.verbose: print(f"Command-line arguments: {opt}")

    # first check: is zip file present?
    if not Path(opt.zip_filename).is_file():
        raise FileNotFoundError(f"zip file does not exist: {opt.zip_filename}")

    # create temp directory using context manager & extract zipfile there
    with tempfile.TemporaryDirectory() as __zipdir:
        with zipfile.ZipFile(f"{opt.zip_filename}", "r") as zf:
            if opt.verbose: print(f"- Extracting dataset into {__zipdir}...", end=" ")
            zf.extractall(f"{__zipdir}")
            if opt.verbose: print("done")

        # create root directories
        _root_dir = Path(f"{opt.root_dataset_dir}")
        _images = _root_dir / "images"
        _labels = _root_dir / "labels"
        _root_dir.mkdir(parents=True, exist_ok=False)
        _images.mkdir(parents=True, exist_ok=False)
        _labels.mkdir(parents=True, exist_ok=False)
        for name in ("train", "test"):
            dirname = _images / name
            dirname.mkdir()
            dirname = _labels / name
            dirname.mkdir()

        # from here on, assume __zipdir has entire dataset

        # create dictionary map between indices and classes
        i2c = _get_dict_classes(fname=f"{__zipdir}/train_labels.csv")

        # YAML file with dataset config
        yml_dict = {
            "path": str(_root_dir),
            "train": "images/train",
            "val": "images/val",
            "names": i2c,
        }
        with open(f"{_root_dir.parent}/plantdoc_dataset.yaml", "w") as f:
            yaml.dump(yml_dict, f)

        # repeat annotations for each dataset
        for ds in ("train", "test"):
            if opt.verbose: print(f"- Processing {ds} dataset")

            # load train/test dataset
            ds_name = f"{__zipdir}/{ds}_labels.csv"
            df = pd.read_csv(filepath_or_buffer=ds_name)

            # clean dataset to remove entries with no matching image in the train/test directory
            clean_df = clean_dataframe(
                df=df, image_directory=Path(__zipdir) / f"{ds}".upper(),
                verbose=opt.verbose
            )

            # make annotations
            annotate_dataset(opt=opt, df=clean_df, ds_id=ds, classes_dict=i2c)

            # copy files
            copy_images(
                src_dir=Path(__zipdir) / f"{ds}".upper(),
                dest_dir=_images / f"{ds}",
                fnames=list(clean_df["filename"]),
            )


if __name__ == "__main__":
    print("Find bounding boxes and make annotations for each skin image")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip-filename",
        "-Z",
        type=str,
        help="Name of zip file with dataset",
    )
    parser.add_argument(
        "--root-dataset-dir",
        type=str,
        help="Path to root directory where train & valid datasets with annotations are saved",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose mode",
    )
    opt = parser.parse_args()

    st = time.monotonic()
    try:
        main(opt)
    except FileNotFoundError as e:
        print(f"ERROR: could not annotate images due to error `{e}`")
    et = time.monotonic()
    print(f"[-- CPU runtime: {et - st:.2f} sec(s) --]")
