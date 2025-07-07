"""Module for computing loss landscapes for PyTorch models."""

# Landscaper Copyright (c) 2025, The Regents of the University of California, 
# through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the 
# U.S. Dept. of Energy), University of California, Berkeley, and Arizona State University. All rights reserved.

# If you have questions about your rights to use or distribute this software, 
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

# NOTICE. This Software was developed under funding from the U.S. Department of Energy and 
# the U.S. Government consequently retains certain rights. As such, the U.S. Government has been
# granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide 
# license in the Software to reproduce, distribute copies to the public, prepare derivative works, 
# and perform publicly and display publicly, and to permit others to do so.

import copy
from collections.abc import Callable
from itertools import product

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from .utils import DeviceStr


# Helper functions for loss landscape computation
def get_model_parameters(
    model: torch.nn.Module, as_complex: bool
) -> list[torch.Tensor]:
    """Get model parameters as a list of tensors.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be retrieved.
        as_complex (bool): If True, convert parameters to complex tensors. If False, keep them as real tensors.

    Returns:
        list[torch.Tensor]: List of model parameters.
    """
    params = [p.data for p in model.parameters()]

    if as_complex:
        params = [
            torch.complex(p, torch.zeros_like(p)) if not torch.is_complex(p) else p
            for p in params
        ]
    return params


def clone_parameters(
    parameters: list[torch.Tensor], as_complex: bool
) -> list[torch.Tensor]:
    """Clone model parameters to avoid modifying the original tensors.

    Args:
        parameters (list[torch.Tensor]): List of model parameters to clone.
        as_complex (bool): If True, convert cloned parameters to complex tensors. If False, keep them as real tensors.

    Returns:
        list[torch.Tensor]: List of cloned parameters.
    """
    params = [p.clone() for p in parameters]

    if as_complex:
        params = [
            torch.complex(p, torch.zeros_like(p)) if not torch.is_complex(p) else p
            for p in params
        ]
    return params


def add_direction(
    parameters: list[torch.Tensor], direction: list[torch.Tensor]
) -> None:
    """Add a direction to parameters in-place.

    Args:
        parameters (list[torch.Tensor]): List of model parameters to modify.
        direction (list[torch.Tensor]): List of direction tensors to add to the parameters.
    """
    for p, d in zip(parameters, direction, strict=False):
        p.add_(d)


def sub_direction(
    parameters: list[torch.Tensor], direction: list[torch.Tensor]
) -> None:
    """Subtract a direction from parameters in-place.

    Args:
        parameters (list[torch.Tensor]): List of model parameters to modify.
        direction (list[torch.Tensor]): List of direction tensors to subtract from the parameters.
    """
    for p, d in zip(parameters, direction, strict=False):
        p.sub_(d)


def scale_direction(direction: list[torch.Tensor], scale: float) -> list[torch.Tensor]:
    """Scale a direction by a given factor.

    Args:
        direction (list[torch.Tensor]): List of direction tensors to scale.
        scale (float): Scaling factor.

    Returns:
        list[torch.Tensor]: Scaled direction tensors.
    """
    for d in direction:
        d.mul_(scale)


def set_parameters(model: torch.nn.Module, parameters: list[torch.Tensor]) -> None:
    """Set model parameters from a list of tensors.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be set.
        parameters (list[torch.Tensor]): List of tensors to set as model parameters.
    """
    for p, new_p in zip(model.parameters(), parameters, strict=False):
        if not torch.is_complex(p) and torch.is_complex(new_p):
            new_p = new_p.real
        p.data.copy_(new_p)


def get_model_norm(parameters: list[torch.Tensor]) -> float:
    """Get L2 norm of parameters.

    Args:
        parameters (list[torch.Tensor]): List of model parameters.

    Returns:
        float: L2 norm of the model parameters.
    """
    return torch.sqrt(sum((p**2).sum() for p in parameters))


def normalize_direction(
    direction: list[torch.Tensor], parameters: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Normalize a direction based on the number of parameters.

    Args:
        direction (list[torch.Tensor]): List of direction tensors to normalize.
        parameters (list[torch.Tensor]): List of model parameters to use for normalization.

    Returns:
        list[torch.Tensor]: Normalized direction tensors.
    """
    for d, p in zip(direction, parameters, strict=False):
        d.mul_(
            torch.sqrt(torch.tensor(p.numel(), dtype=torch.float32, device=d.device))
            / (d.norm() + 1e-10)
        )
    return direction


def compute_loss_landscape(
    model: torch.nn.Module,
    data: npt.ArrayLike,
    dirs: npt.ArrayLike,
    scalar_fn: Callable[[torch.nn.Module, npt.ArrayLike], float],
    steps: int = 41,
    distance: float = 0.01,
    dim: int = 3,
    device: DeviceStr = "cuda",
    use_complex: bool = False,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Computes the loss landscape along the top-N eigenvector directions.

    Args:
        model (torch.nn.Module): The model to analyze.
        data (npt.ArrayLike): Data that will be used to evaluate the loss function for each point on the landscape.
        dirs (npt.ArrayLike): 2D array of directions to generate the landscape with.
        scalar_fn (Callable[[torch.nn.Module, npt.ArrayLike], float]): This function should take a model
            and your data and return a scalar value; it gets called repeatedly with perturbed versions of the model.
        steps (int): Number of steps in each dimension.
        distance (float): Total distance to travel in parameter space. Setting this value too high
            may lead to unreliable results.
        dim (int): Number of dimensions for the loss landscape (default: 3)
        device (Literal["cuda", "cpu"]): Device used to compute landscape.
        use_complex (bool): Computes Landscape using complex numbers if this is set to true;
            use if your directions are complex.

    Returns:
        The loss values and coordinates for the landscape as numpy arrays.
    """
    # Initialize loss hypercube - For dim dimensions, we need a dim-dimensional array
    loss_shape = tuple([steps] * dim)
    loss_hypercube = np.zeros(loss_shape)

    coordinates = [np.linspace(-distance, distance, steps) for _ in range(dim)]

    # Compute loss landscape - this is the core logic that needs to be efficient for N dimensions
    if dim > 5:
        print(
            f"Warning: {dim} dimensions may require significant memory and computation time."
        )
        print(
            f"Consider reducing the 'steps' parameter (currently {steps}) or using a lower dimension."
        )

    with torch.no_grad():
        # Get starting parameters and save original weights
        start_point = get_model_parameters(model, use_complex)
        original_weights = clone_parameters(start_point, use_complex)

        # Get top-N eigenvectors as directions
        directions = copy.deepcopy(dirs)
        if dim > len(directions):
            raise ValueError(
                f"Requested dimension {dim} exceeds available directions ({len(directions)})."
            )

        # Normalize all directions
        for i in range(dim):
            directions[i] = normalize_direction(directions[i], start_point)

        # Scale directions to match steps and total distance
        model_norm = get_model_norm(start_point)
        for i in range(dim):
            dir_norm = get_model_norm(directions[i])
            scale_direction(
                directions[i], ((model_norm * distance) / (steps / 2)) / dir_norm
            )

        # Generate grid coordinates
        grid_points = list(product(range(steps), repeat=dim))
        print(f"Computing {len(grid_points)} points in {dim}D space...")

        center_idx = steps // 2
        try:
            for gp in tqdm(grid_points, desc=f"Computing {dim}D landscape"):
                # Create a new parameter set for this grid point
                point_params = clone_parameters(original_weights, use_complex)

                # Move to the specified grid point by adding appropriate steps in each direction
                for dim_idx, point_idx in enumerate(gp):
                    steps_from_center = point_idx - center_idx

                    if steps_from_center > 0:
                        for _ in range(steps_from_center):
                            add_direction(point_params, directions[dim_idx])
                    elif steps_from_center < 0:
                        for _ in range(-steps_from_center):
                            sub_direction(point_params, directions[dim_idx])

                # Set model parameters
                set_parameters(model, point_params)
                loss = scalar_fn(model, data)
                loss_hypercube[gp] = loss

                # Clear GPU memory
                if gp[0] % 5 == 0 and device == "cuda" and all(x == 0 for x in gp[1:]):
                    torch.cuda.empty_cache()
        finally:
            # Restore original weights
            set_parameters(model, original_weights)

        # Handle extreme values in loss surface
        loss_hypercube = np.nan_to_num(
            loss_hypercube,
            nan=np.nanmean(loss_hypercube),
            posinf=np.nanmax(loss_hypercube[~np.isinf(loss_hypercube)]),
            neginf=np.nanmin(loss_hypercube[~np.isinf(loss_hypercube)]),
        )

        # Print statistics about the loss hypercube
        print(
            f"Loss hypercube stats - min: {np.min(loss_hypercube)}, max: {np.max(loss_hypercube)}, "
            f"mean: {np.mean(loss_hypercube)}"
        )

    return loss_hypercube, coordinates
