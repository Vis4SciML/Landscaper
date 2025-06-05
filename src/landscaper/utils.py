"""Utility functions for the Landscaper package."""

import os
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch

DeviceStr = Literal["cuda", "cpu"]
Number = int | float


def add_random_orthogonal_direction(start_point, directions):
    random_dir = [torch.randn_like(p) for p in start_point]
    for prev_dir in directions:
        dot_product = sum(
            (d1 * d2).sum() for d1, d2 in zip(random_dir, prev_dir, strict=False)
        )

        for j, (d1, d2) in enumerate(zip(random_dir, prev_dir, strict=False)):
            random_dir[j] = d1 - dot_product * d2
    directions.append(random_dir)


def group_product(xs: list[torch.Tensor], ys: list[torch.Tensor]) -> torch.Tensor:
    """Computes the dot product of two lists of tensors.

    Args:
        xs (list[torch.Tensor]): List of tensors.
        ys (list[torch.Tensor]): List of tensors.

    Returns:
        torch.Tensor: The sum of the element-wise products of the tensors in xs and ys.
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys, strict=False)])


def group_add(
    params: list[torch.Tensor], update: list[torch.Tensor], alpha: float = 1
) -> list[torch.Tensor]:
    """Adds the update to the parameters with a scaling factor alpha.

    Params = params + update*alpha

    Args:
        params (list[torch.Tensor]): List of parameters.
        update (list[torch.Tensor]): List of updates.
        alpha (float, optional): Scaling factor. Defaults to 1.

    Returns:
        list[torch.Tensor]: Updated list of parameters.
    """
    for i, _ in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v: list[torch.Tensor]) -> list[torch.Tensor]:
    """Normalization of a list of vectors v.

    Args:
        v (list[torch.Tensor]): List of tensors to normalize.

    Returns:
        list[torch.Tensor]: Normalized list of tensors.
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def orthnormal(w: list[torch.Tensor], v_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """Make vector w orthogonal to each vector in v_list and normalize the output w.

    Args:
        w (list[torch.Tensor]): The vector to be made orthogonal.
        v_list (list[torch.Tensor]): List of vectors to which w should be made orthogonal.

    Returns:
        list[torch.Tensor]: The orthogonalized and normalized vector w.
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)


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
