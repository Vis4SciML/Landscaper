"""Module for computing loss landscapes for PyTorch models."""

from collections.abc import Callable
from itertools import product

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from .hessian import PyHessian
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


def add_direction(parameters: list[torch.Tensor], direction: list[torch.Tensor]) -> None:
    """Add a direction to parameters in-place.

    Args:
        parameters (list[torch.Tensor]): List of model parameters to modify.
        direction (list[torch.Tensor]): List of direction tensors to add to the parameters.
    """
    for p, d in zip(parameters, direction, strict=False):
        p.add_(d)


def sub_direction(parameters: list[torch.Tensor], direction: list[torch.Tensor]) -> None:
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


def normalize_direction(direction: list[torch.Tensor], parameters: list[torch.Tensor]) -> list[torch.Tensor]:
    """Normalize a direction based on the number of parameters.

    Args:
        direction (list[torch.Tensor]): List of direction tensors to normalize.
        parameters (list[torch.Tensor]): List of model parameters to use for normalization.

    Returns:
        list[torch.Tensor]: Normalized direction tensors.
    """
    for d, p in zip(direction, parameters, strict=False):
        d.mul_(torch.sqrt(torch.tensor(p.numel(), dtype=torch.float32, device=d.device)) / (d.norm() + 1e-10))
    return direction


def compute_loss_landscape(
    model: torch.nn.Module,
    data: npt.ArrayLike,
    hessian_comp: PyHessian,
    loss_function: Callable[[torch.nn.Module, npt.ArrayLike], float],
    top_n: int = 10,
    steps=41,
    distance: float = 1.0,
    dim: int = 3,
    batch_size: int = 10,
    device: DeviceStr = "cuda",
) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """Computes the loss landscape along the top-N eigenvector directions.

    Args:
        model (torch.nn.Module): The model to analyze.
        data (npt.ArrayLike): Data that will be used to evaluate the loss function for each point on the landscape.
        hessian_comp (PyHessian): PyHessian instance used for hessian computation.
        loss_function (Callable[[torch.nn.Module, npt.ArrayLike], float]): Loss function for the model.
            Should take the model and data as input and return a float loss value.
        top_n (int): Number of hessian eigenvalues to compute.
        steps (int): Number of steps in each dimension.
        distance (float): Total distance to travel in parameter space.
        dim (int): Number of dimensions for the loss landscape (default: 3)
        batch_size (int): Batch size used to compute each point on landscape.
        device (Literal["cuda", "cpu"]): Device used to compute landscape.
    """
    top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=top_n)
    print(f"Top {top_n} eigenvalues: {top_eigenvalues}")

    # Get starting parameters and save original weights
    with torch.no_grad():
        start_point = get_model_parameters(model)
        original_weights = clone_parameters(start_point)

    try:
        coordinates = [np.linspace(-distance, distance, steps) for _ in range(dim)]

        # Get top-N eigenvectors as directions
        directions = []
        for i in range(dim):
            if i < len(top_eigenvectors):
                directions.append([v.clone() for v in top_eigenvectors[i]])
            else:
                print(
                    f"Warning: Requested dimension {dim} exceeds available eigenvectors ({len(top_eigenvectors)}). "
                    f"Using random direction for dimension {i + 1}"
                )
                # Create a random direction if we don't have enough eigenvectors
                random_dir = [torch.randn_like(p) for p in start_point]
                # Make it orthogonal to previous directions (simplified Gram-Schmidt)
                for prev_dir in directions:
                    dot_product = sum((d1 * d2).sum() for d1, d2 in zip(random_dir, prev_dir, strict=False))

                    for j, (d1, d2) in enumerate(zip(random_dir, prev_dir, strict=False)):
                        random_dir[j] = d1 - dot_product * d2
                directions.append(random_dir)

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

        # Initialize loss hypercube - For dim dimensions, we need a dim-dimensional array
        loss_shape = tuple([steps] * dim)
        loss_hypercube = np.zeros(loss_shape)

        # Compute loss landscape - this is the core logic that needs to be efficient for N dimensions
        with torch.no_grad():
            # For very high dimensions, we'll need to be smarter about traversal
            # Rather than recursive approach, use an iterative approach with mesh grid

            # For dimensions > 5, we'll likely run out of memory with a full grid
            if dim > 5:
                print(f"Warning: High dimensionality ({dim}) may require significant memory and computation time.")
                print(f"Consider reducing the 'steps' parameter (currently {steps}) or using a lower dimension.")
                # For very high dimensions, we might want to do random sampling instead

            # Generate grid coordinates

            grid_points = list(product(range(steps), repeat=dim))
            print(f"Computing {len(grid_points)} points in {dim}D space...")

            # Batch processing for efficiency - compute multiple grid points at once
            num_batches = (len(grid_points) + batch_size - 1) // batch_size

            # Initialize counters for averaging
            loss_counts = np.zeros(loss_shape, dtype=int)

            for batch_idx in tqdm(range(num_batches), desc=f"Computing {dim}D landscape"):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(grid_points))
                current_batch = grid_points[batch_start:batch_end]

                for grid_point in current_batch:
                    # Create a new parameter set for this grid point
                    point_params = clone_parameters(current_point)

                    # Move to the specified grid point by adding appropriate steps in each direction
                    for dim_idx, steps_in_dim in enumerate(grid_point):
                        for _ in range(steps_in_dim):
                            add_direction(point_params, directions[dim_idx])

                    # Set model parameters and compute loss
                    set_parameters(model, point_params)
                    with torch.no_grad():
                        loss = loss_function(model, data)

                    # Accumulate loss and increment counter for averaging
                    loss_hypercube[grid_point] += loss.item()
                    loss_counts[grid_point] += 1

                # Clear GPU memory
                if batch_idx % 5 == 0 and device == "cuda":
                    torch.cuda.empty_cache()

            # Compute averages
            with np.errstate(divide="ignore", invalid="ignore"):
                loss_hypercube = np.divide(
                    loss_hypercube,
                    loss_counts,
                    where=loss_counts != 0,
                    out=np.zeros_like(loss_hypercube),
                )

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

    except Exception as e:
        print(f"Error during loss landscape computation: {e}.")
        import traceback

        traceback.print_exc()
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

    return top_eigenvalues, top_eigenvectors, loss_hypercube, coordinates
