"""Module for computing loss landscapes for PyTorch models."""

import copy
from collections.abc import Callable
from itertools import product

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from .utils import DeviceStr


# Helper functions for loss landscape computation
def get_model_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    """Get model parameters as a list of tensors.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be retrieved.

    Returns:
        list[torch.Tensor]: List of model parameters.
    """
    return [p.data for p in model.parameters()]


def clone_parameters(parameters: list[torch.Tensor]) -> list[torch.Tensor]:
    """Clone model parameters to avoid modifying the original tensors.

    Args:
        parameters (list[torch.Tensor]): List of model parameters to clone.

    Returns:
        list[torch.Tensor]: List of cloned parameters.
    """
    return [p.clone() for p in parameters]


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
    return direction


def set_parameters(model: torch.nn.Module, parameters: list[torch.Tensor]) -> None:
    """Set model parameters from a list of tensors.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be set.
        parameters (list[torch.Tensor]): List of tensors to set as model parameters.
    """
    for p, new_p in zip(model.parameters(), parameters, strict=False):
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
    loss_function: Callable[[torch.nn.Module, npt.ArrayLike], float],
    steps=41,
    distance: float = 0.01,
    dim: int = 3,
    batch_size: int = 10,
    device: DeviceStr = "cuda",
) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """Computes the loss landscape along the top-N eigenvector directions.

    Args:
        model (torch.nn.Module): The model to analyze.
        data (npt.ArrayLike): Data that will be used to evaluate the loss function for each point on the landscape.
        dirs (npt.ArrayLike): 2D array of directions to generate the landscape with.
        loss_function (Callable[[torch.nn.Module, npt.ArrayLike], float]): Loss function for the model.
            Should take the model and data as input and return a float loss value.
        steps (int): Number of steps in each dimension.
        distance (float): Total distance to travel in parameter space. Setting this value too high may lead to unreliable results.
        dim (int): Number of dimensions for the loss landscape (default: 3)
        batch_size (int): Batch size used to compute each point on landscape.
        device (Literal["cuda", "cpu"]): Device used to compute landscape.
    """

    # Initialize loss hypercube - For dim dimensions, we need a dim-dimensional array
    loss_shape = tuple([steps] * dim)
    loss_hypercube = np.zeros(loss_shape)

    coordinates = [np.linspace(-distance, distance, steps) for _ in range(dim)]

    with torch.no_grad():
        # Get starting parameters and save original weights
        start_point = get_model_parameters(model)
        original_weights = clone_parameters(start_point)

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
            scale_direction(directions[i], ((model_norm * distance) / steps) / dir_norm)

        # Move start point to corner (lowest point in all dimensions)
        current_point = clone_parameters(original_weights)
        for i in range(dim):
            scaled_dir = clone_parameters(directions[i])
            scale_direction(scaled_dir, steps / 2)
            sub_direction(current_point, scaled_dir)
            # Rescale direction vectors for stepping
            scale_direction(directions[i], 2.0 / steps)

        # Compute loss landscape - this is the core logic that needs to be efficient for N dimensions
        if dim > 5:
            print(
                f"Warning: High dimensionality ({dim}) may require significant memory and computation time."
            )
            print(
                f"Consider reducing the 'steps' parameter (currently {steps}) or using a lower dimension."
            )

        # Generate grid coordinates
        grid_points = list(product(range(steps), repeat=dim))
        print(f"Computing {len(grid_points)} points in {dim}D space...")

        center_idx = steps // 2
        try:
            for gp in tqdm(grid_points, desc=f"Computing {dim}D landscape"):
                # Create a new parameter set for this grid point
                point_params = clone_parameters(current_point)

                # Move to the specified grid point by adding appropriate steps in each direction
                for dim_idx, point_idx in enumerate(gp):
                    steps_from_center = point_idx - center_idx

                    if steps_from_center > 0:
                        for _ in range(steps_from_center):
                            add_direction(point_params, directions[dim_idx])
                    elif steps_from_center < 0:
                        for _ in range(steps_from_center):
                            sub_direction(point_params, directions[dim_idx])

                # Set model parameters
                set_parameters(model, point_params)
                loss = loss_function(model, data)

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
