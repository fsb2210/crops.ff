"""
Create annotations for each image and store them in a proper directory
as required by YOLO & detectron2 models
"""

import glob
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import shutil
import zipfile
import time

import pandas as pd


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
    height = image_height
    return x_center, y_center, width, height


def _get_categories(df: pd.DataFrame) -> Dict[str, int]:
    """Create dictionary for class -> index equivalence"""
    classes = df["class"]
    keys = list(set(classes))
    class2id = dict(zip((keys), range(len(keys))))
    return class2id


def annotate_dataset(
    opt: argparse.Namespace, df: pd.DataFrame, ds_id: str, classes_dict: dict
) -> None:
    """Annotate dataset and save annotations to new database"""
    # loop over elements in the datase.txtt
    for k, fname in enumerate(df["filename"]):
        if not opt.verbose:
            print(f"\r- annotating {k}/{df.shape[0]}", end=" ")

        row = df.iloc[k]
        rclass = classes_dict[row["class"]]

        rw, rh = int(row["width"]), int(row["height"])
        if rw == 0 or rh == 0:
            print(f"WARNING: found 0 dimension(s) in {fname}: w={rw:.2f} h={rh:.2f}")
            continue
        rxmin, rxmax = int(row["xmin"]), int(row["xmax"])
        rymin, rymax = int(row["ymin"]), int(row["ymax"])
        bbox = convert_to_yolo(rxmin, rymin, rxmax, rymax, rw, rh)

        ann_str = (
            f"{rclass:2d} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n"
        )
        if opt.verbose:
            print(f"- annotation for {fname}: >>> {ann_str}")
        with open(
            f"{opt.root_dataset_dir}/labels/{ds_id}/{fname[:-4]}.txt", mode="a"
        ) as f:
            f.write(ann_str)
    print()
    return


def copy_images(
    src_dir: Union[str, Path], dest_dir: Union[str, Path], fnames: List[str]
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
    for fname in fnames:
        shutil.copyfile(f"{src_dir}/{fname}", f"{dest_dir}/{fname}")
    return


def clean_dataframe(
    df: pd.DataFrame, image_directory: Union[str, Path]
) -> pd.DataFrame:
    """Clean dataframe from images not present in the dataset

    Parameters
    ----------
    df : `pd.DataFrame`
        Dataframe of the dataset

    image_directory: `str / Path`
        Directory with images of the dataset
    """
    images_in_ds = glob.glob("*.*g", root_dir=image_directory)
    keys_to_drop = []
    for k, fname in enumerate(df["filename"]):
        if fname not in images_in_ds:
            print(
                f"WARNING: {fname} not found in {image_directory}, will not be added to dataset"
            )
            keys_to_drop.append(k)
    new_df = df.drop(keys_to_drop)
    return new_df


def main(opt: argparse.Namespace) -> None:
    """Make YOLO-supported annotations

    Parameters
    ----------
    opt : `argparse.Namespace`
        Options from the command line
    """

    # first check: is zip file present?
    if not Path(opt.zip_filename).is_file():
        raise FileNotFoundError(f"zip file does not exist: {opt.zip_filename}")

    # use hardcode path, not gonna reach end product
    __zipdir = Path(__file__).parent / "temp"
    if not Path(f"{__zipdir}").is_dir():
        __zipdir.mkdir(parents=True)
        with zipfile.ZipFile(f"{opt.zip_filename}", "r") as zf:
            if opt.verbose:
                print(f"- extracting dataset into {__zipdir}...", end=" ")
            zf.extractall(f"{__zipdir}")
            if opt.verbose:
                print("done")
    else:
        if opt.verbose:
            print(f"- dataset already present in {__zipdir}")

    # create root directories
    _root_dir = Path(f"{opt.root_dataset_dir}")
    _root_dir.mkdir(parents=True, exist_ok=True)
    _images = _root_dir / "images"
    _labels = _root_dir / "labels"

    # from here on, assume __zipdir has entire dataset
    # dictionary matching class names to int ids
    __classes_ids = Path(__file__).parent / "temp" / "classes_ids.json"
    if __classes_ids.is_file():
        with open(__classes_ids, "r") as f:
            c2i = json.load(f)
    else:
        _df = pd.read_csv(filepath_or_buffer=f"{__zipdir}/train_labels.csv")
        c2i = _get_categories(_df)
        del _df
        with open(__classes_ids, "w") as f:
            json.dump(c2i, f, indent=2)

    # copy classes -> id dict to root dataset directory
    with open(f"{_root_dir}/classes_ids.json", "w") as f:
        json.dump(c2i, f, indent=2)

    # repeat annotations for each dataset
    for ds in ("train", "test"):
        # create directory structure
        for _dir in (_images, _labels):
            _ds = _dir / ds
            _ds.mkdir(parents=True, exist_ok=False)

        # load train/test dataset
        ds_name = f"{__zipdir}/{ds}_labels.csv"
        df = pd.read_csv(filepath_or_buffer=ds_name)

        # clean dataset to remove entries with no matching image in the train/test directory
        clean_df = clean_dataframe(
            df=df, image_directory=Path(__file__).parent / "temp" / f"{ds}".upper()
        )

        # make annotations
        annotate_dataset(opt=opt, df=clean_df, ds_id=ds, classes_dict=c2i)

        # copy files
        copy_images(
            src_dir=Path(__file__).parent / "temp" / f"{ds}".upper(),
            dest_dir=_images / f"{ds}",
            fnames=clean_df["filename"],
        )

    # remove zipdir created
    shutil.rmtree(__zipdir)


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
