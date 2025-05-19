import os
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

DeviceStr = Literal["cuda", "cpu"]
Number = int | float


def load_landscape(fp: str) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Loads a loss landscape from an .npz file.

    Args:
        fp (str): Path to the file.

    Returns:
        Tuple(npt.ArrayLike, npt.ArrayLike): loss, coordinates.

    Raises:
        ValueError: Thrown if file is invalid.
    """
    ext = os.path.splitext(fp)[1]
    if ext != ".npz":
        raise ValueError(f"File is not a .npz; got: {ext}")

    required = ["loss", "coordinates"]
    d = np.load(fp)

    if any([r not in d for r in required]):
        raise ValueError("File is not a loss landscape.")

    loss = d.get("loss")
    coords = d.get("coordinates")

    dims = coords.shape[0]
    if dims < 2:
        raise ValueError("Coordinates must at least be 2 dimensional.")

    return loss, coords


def validate_dataframe(df: pd.DataFrame, required: list[str], name: str = "df"):
    """Checks if a dataframe contains the correct data.
    TODO: replace with https://github.com/unionai-oss/pandera

    Args:
        df (pd.DataFrame): The dataframe to validate.
        required (list[str]): List of columns that must be present in the dataframe.
        name (str): Name of the dataframe in the error message.

    Raises:
        ValueError: Thrown if the columns in `required` are missing in `df`.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}.")
