import os
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

DeviceStr = Literal["cuda", "cpu"]
Number = int | float


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0.0 if param.grad is None else param.grad + 0.0)
    return params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(
        gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True
    )
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
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
