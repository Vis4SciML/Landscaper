"""Plots for visualizing topological data analysis results.

TODO: get rid of plt.show calls so that users can pick between saving the figure & displaying it interactively
"""

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

from collections.abc import Callable
from typing import TypedDict

import drawsvg as dw
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import topopy as tp
from coloraide import Color
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

from .tda import get_persistence_dict
from .utils import Number


def persistence_barcode(
    msc: tp.MorseSmaleComplex, show: bool = True, figsize: tuple[int, int] = (12, 6)
) -> None | Figure:
    """Plots the [persistence barcode](https://en.wikipedia.org/wiki/Persistence_barcode)  for a Morse-Smale complex.

    Args:
        msc (tp.MorseSmaleComplex): A Morse-Smale complex.
        show (bool): Shows the plot if true, otherwise returns the figure.
        figsize (tuple[int,int]): Size of the figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    node_list = [str(node) for node in list(get_persistence_dict(msc).keys())]
    persistence_list = list(get_persistence_dict(msc).values())
    ax.barh(node_list, persistence_list)
    ax.set_xlabel("Persistence")
    ax.set_ylabel("Node")
    ax.set_title("Node vs Persistence")

    if not show:
        return fig
    plt.show()


def linearScale(min_val: Number, max_val: Number, new_min: Number, new_max: Number) -> Callable[[Number], Number]:
    """Creates a linear scale that maps [min_val, max_val] -> [new_min, new_max]; similar to d3's `linearScale`.

    Args:
        min_val (int | float): Current min value.
        max_val (int | float): Current max value.
        new_min (int | float): Desired min value.
        new_max (int | float): Desired max value.

    Returns:
        A function to convert values from the old range to the new one.
    """
    return lambda x: (new_max - new_min) / (max_val - min_val) * (x - max_val) + new_max


class AxisOptions(TypedDict):
    """Options for the Y axis of the topology profile.

    Attributes:
        tick_format (Callable[[float], str]): Function to format the tick labels.
        font_size (int): Font size for the tick labels.
    """

    tick_format: Callable[[float], str]
    font_size: int


default_axis = AxisOptions(tick_format=lambda x: str(round(x, 3)), font_size=14)


def topology_profile(
    data,
    y_min: float | None = None,
    y_max: float | None = None,
    size: int = 800,
    margin: int = 15,
    color: str = "red",
    background_color: str = "white",
    gradient: bool = True,
    y_axis: AxisOptions | None = default_axis,
) -> dw.Drawing:
    """Renders a topological profile.

    Renders a topological profile for the given merge tree data
    extracted with `extract_merge_tree` from `landscaper.tda`.

    Args:
        data (List[List[float]]): The merge tree data.
        y_min (Optional[float]): Optional minimum y value for the drawing.
        y_max (Optional[float]): Optional maximum y value for the drawing.
        size (int): Size in pixels of the resulting drawing.
        margin (int): Size of the margins in pixels.
        color (str): Color used to draw the profile.
        background_color (str): Color used to draw the background.
        gradient (bool): If true, fills the profile using a gradient from `background_color` to `color`.
            If false, only uses `color` to fill the path. Set this to false if you are
            exporting the drawing into a different format.
        y_axis (AxisOptions): Sets options for the Y axis. Set to None to disable.
    """
    # TODO: validate profile data
    width = size
    height = size
    marginTop = margin
    marginRight = margin
    marginBottom = margin
    marginLeft = margin

    loss_max = float("-inf")
    loss_min = float("inf")
    x_max = float("-inf")
    x_min = float("inf")

    # data should be a list of lists
    for d in data:
        xVals = [pt[0] for pt in d]
        yVals = [pt[1] for pt in d]

        x_max = max(x_max, max(xVals))
        x_min = min(x_min, min(xVals))
        loss_max = max(loss_max, max(yVals))
        loss_min = min(loss_min, min(yVals))

    # keep colors consistent regardless of y min and max chosen
    basinColors = Color.interpolate(
        [color, background_color],
        domain=[max(loss_min, 1e-10), loss_max],
    )

    if y_max is not None:
        loss_max = y_max

    if y_min is not None:
        loss_min = y_min

    xScale = linearScale(x_min, x_max, marginLeft, width - marginRight)
    yScale = linearScale(loss_min, loss_max, height - marginBottom, marginTop)

    svg = dw.Drawing(width, height)
    svg.append(dw.Rectangle(0, 0, width, height, fill="white", stroke="#777"))  # background color

    for d in data:
        yVals = [pt[1] for pt in d]
        minY = min(yVals)
        maxY = max(yVals)

        if gradient:
            grad = dw.LinearGradient("0%", "100%", "0%", "0%", gradientUnits="objectBoundingBox")

            for t in np.linspace(0.0, 1.0, 100):
                yValue = minY + t * (maxY - minY)
                grad.add_stop(f"{t * 100}%", basinColors(yValue).to_string(hex=True, upper=True))
        else:
            grad = color

        path = dw.Path(stroke=grad, fill=grad)
        start, *pts = d
        sx, sy = start
        path.M(xScale(sx), yScale(sy))
        for pt in pts:
            x, y = pt
            path.L(xScale(x), yScale(y))
        svg.append(path)

    if y_axis is not None:
        ax = dw.Line(
            marginLeft / 2,
            height - marginBottom,
            marginLeft / 2,
            marginTop,
            stroke="black",
        )

        svg.append(ax)
        for t in np.linspace(0.0, 1.0, 10):
            v = loss_min + t * (loss_max - loss_min)
            tv = yScale(v)
            tick = dw.Line(marginLeft / 2, tv, marginLeft, tv, stroke="black")
            lbl = dw.Text(
                y_axis["tick_format"](v),
                font_size=y_axis["font_size"],
                dominant_baseline="middle",
                x=marginLeft,
                y=tv,
            )
            svg.append(lbl)
            svg.append(tick)

    return svg


def contour(
    coordinates: npt.ArrayLike,
    loss: npt.ArrayLike,
    show: bool = True,
    figsize: tuple[int, int] = (12, 8),
) -> None | Figure:
    """Draws a contour plot from the provided coordinates and values.

    Args:
        coordinates (npt.ArrayLike): n-dimensional coordinates.
        loss (npt.ArrayLike): Value for each coordinate.
        figsize (tuple[int, int]): Size of the figure.
        show (bool): If true, shows the plot; otherwise returns the figure.

    Raises:
        ValueError: Raised if rendering fails.
    """
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    X, Y = np.meshgrid(coordinates[0], coordinates[1])

    # Ensure all values are positive for log scale
    min_loss = np.min(loss)
    if min_loss <= 0:
        shift = -min_loss + 1e-6
        loss = loss + shift
        print(f"Shifted loss surface by {shift} to ensure positive values")

    # Create logarithmically spaced levels
    min_val = np.min(loss[loss > 0])
    max_val = np.max(loss)

    if min_val >= max_val:
        raise ValueError("Invalid level range")

    try:
        levels = np.logspace(np.log10(min_val), np.log10(max_val), 30)
        # Create contour plot with log scale
        contour_filled = ax1.contourf(
            X,
            Y,
            loss,
            levels=levels,
            norm=LogNorm(vmin=min_val, vmax=max_val),
            cmap="RdYlBu_r",
        )

        contour_lines = ax1.contour(
            X,
            Y,
            loss,
            levels=levels[::3],
            colors="black",
            linewidths=0.5,
            alpha=0.5,
        )
        ax1.clabel(contour_lines, inline=True, fontsize=8, fmt="%.3f")

    except Exception as e:
        print(f"Warning: Log-scale contour plot failed ({e}). Using linear scale...")
        try:
            # Try linear scale with fewer levels
            levels = np.linspace(np.min(loss), np.max(loss), 20)
            contour_filled = ax1.contourf(X, Y, loss, levels=levels, cmap="RdYlBu_r")
            contour_lines = ax1.contour(
                X,
                Y,
                loss,
                levels=levels[::2],
                colors="black",
                linewidths=0.5,
                alpha=0.5,
            )
            ax1.clabel(contour_lines, inline=True, fontsize=8, fmt="%.3f")
        except Exception as e:
            print(f"Warning: Linear scale plotting failed ({e}). Using pcolormesh...")
            contour_filled = ax1.pcolormesh(X, Y, loss, cmap="RdYlBu_r", shading="auto")

    try:
        plt.colorbar(contour_filled, ax=ax1, label="Loss")
    except Exception as e:
        print(f"Warning: Could not create colorbar: {e}")

    ax1.set_xlabel("Direction of First Eigenvector", fontsize=12)
    ax1.set_ylabel("Direction of Second Eigenvector", fontsize=12)
    ax1.set_title("Loss Landscape Contour", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.axis("equal")

    if not show:
        return fig
    plt.show()


def surface_3d(
    coords: npt.ArrayLike,
    loss: npt.ArrayLike,
    show: bool = True,
    figsize: tuple[int, int] = (12, 8),
) -> None | Figure:
    """Generates a 3d surface plot for the given coordinates and values. Fails if dimensions are greater than 2.

    Args:
        coords (npt.ArrayLike): 2-D coordinates.
        loss (npt.ArrayLike): Values for the coordinates.
        show (bool): Shows the plot if true, otherwise returns the figure.
        figsize (tuple[int,int]): Size of the figure.
    """
    # Create 3D surface plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(coords[0], coords[1])

    min_val = np.min(loss[loss > 0])
    max_val = np.max(loss)

    try:
        # Try log-scale surface plot
        print("Attempting log-scale surface plot...")
        norm = LogNorm(vmin=min_val, vmax=max_val)
        surf = ax.plot_surface(
            X,
            Y,
            loss,
            cmap="RdYlBu_r",
            norm=norm,
            linewidth=0,
            antialiased=True,
        )
        plt.colorbar(surf, label="Loss (log scale)")
    except Exception as e:
        print(f"Warning: Log-scale 3D plotting failed ({e}). Using linear scale...")
        surf = ax.plot_surface(X, Y, loss, cmap="RdYlBu_r", linewidth=0, antialiased=True)
        plt.colorbar(surf, label="Loss")

    ax.set_xlabel("Direction of First Eigenvector")
    ax.set_ylabel("Direction of Second Eigenvector")
    ax.set_zlabel("Loss")
    ax.set_title("3D Loss Landscape")

    # Adjust the viewing angle for better visualization
    ax.view_init(elev=30, azim=45)

    if not show:
        return fig
    plt.show()


def hessian_density(eigen: npt.ArrayLike, weight: npt.ArrayLike, show: bool = True, figsize=(12, 6)) -> None | Figure:
    """Plots the density distribution of Hessian eigenvalues.

    Args:
        eigen (npt.ArrayLike): Array of Hessian eigenvalues.
        weight (npt.ArrayLike): Corresponding weights for the eigenvalues.
        show (bool): Shows the plot if true, otherwise returns the figure.
        figsize (tuple[int, int]): Size of the figure.
    """
    density_eigen = np.array(eigen)
    density_weight = np.array(weight)

    # Ensure both arrays are 1D
    if density_eigen.ndim > 1:
        density_eigen = density_eigen.ravel()
    if density_weight.ndim > 1:
        density_weight = density_weight.ravel()

    # Ensure arrays have matching dimensions
    if len(density_eigen) != len(density_weight):
        # Create new x points for interpolation
        x_old = np.linspace(min(density_eigen), max(density_eigen), len(density_weight))
        x_new = np.linspace(min(density_eigen), max(density_eigen), len(density_eigen))

        f = interp1d(x_old, density_weight, kind="linear", fill_value="extrapolate")
        density_weight = f(x_new)

    # Ensure we're only plotting real components
    if np.iscomplexobj(density_eigen):
        density_eigen = density_eigen.real
    if np.iscomplexobj(density_weight):
        density_weight = density_weight.real

    # Sort values for better visualization
    sort_idx = np.argsort(density_eigen)
    density_eigen = density_eigen[sort_idx]
    density_weight = density_weight[sort_idx]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Create smooth curves using kernel density estimation
    if len(density_eigen) > 1:  # Only if we have enough points
        # Separate positive and negative regions
        pos_mask = density_eigen >= 0
        neg_mask = density_eigen < 0

        # Create histogram data with more bins for better resolution
        num_bins = 200  # Increased from 100 to 200 for more detail

        # Find the global min and max for consistent binning
        global_min = min(density_eigen)
        global_max = max(density_eigen)

        # Create consistent bins across the entire range
        bins = np.linspace(global_min, global_max, num_bins + 1)

        # Create separate histograms but using the same bin definitions
        pos_hist, _ = np.histogram(density_eigen[pos_mask], bins=bins, density=True)
        neg_hist, _ = np.histogram(density_eigen[neg_mask], bins=bins, density=True)

        # Plot histograms with consistent bins
        ax.hist(
            density_eigen[pos_mask],
            bins=bins,
            alpha=0.4,
            color="#90CAF9",
            label="Positive Histogram",
            density=True,
            edgecolor="#2E86C1",
            linewidth=0.5,
        )
        ax.hist(
            density_eigen[neg_mask],
            bins=bins,
            alpha=0.4,
            color="#FFAB91",
            label="Negative Histogram",
            density=True,
            edgecolor="#E74C3C",
            linewidth=0.5,
        )

    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_xlabel("Eigenvalue", fontsize=12, fontweight="bold")
    ax.set_title(
        "Hessian Eigenvalue Density Distribution",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Add vertical line at x=0
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

    # Add legend if we have both positive and negative values
    if np.any(density_eigen < 0) and np.any(density_eigen >= 0):
        ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if not show:
        return fig
    plt.show()


def hessian_eigenvalues(top_eigenvalues: npt.ArrayLike, show: bool = True, figsize=(12, 6)) -> None | Figure:
    """Plots the top-10 Hessian eigenvalues as an enhanced bar chart.

    Args:
        top_eigenvalues (npt.ArrayLike): Array of top-10 Hessian eigenvalues.
        show (bool): Shows the plot if true, otherwise returns the figure.
        figsize (tuple[int, int]): Size of the figure.
    """
    # Plot the top-10 eigenvalues as an enhanced bar chart
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    indices = np.arange(len(top_eigenvalues))

    # Create bars with different colors for positive and negative values
    colors = ["#2E86C1" if val >= 0 else "#E74C3C" for val in top_eigenvalues]
    bars = ax.bar(indices, top_eigenvalues, color=colors, width=0.7)

    # Add value labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01 * abs(yval),
            f"{yval:.3f}",
            ha="center",
            va="bottom" if yval >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    plt.xlabel("Index", fontsize=12, fontweight="bold")
    plt.ylabel("Eigenvalue", fontsize=12, fontweight="bold")
    plt.title(
        f"Top-{len(top_eigenvalues)} Hessian Eigenvalues",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    # Improve x-axis ticks
    ax.set_xticks(indices, [f"{i + 1}" for i in indices], fontsize=10)

    # Add horizontal line at y=0 with better styling
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.2, zorder=0)

    # Add grid with better styling
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)

    legend_elements = [
        Patch(facecolor="#2E86C1", label="Positive Eigenvalues", alpha=0.9),
        Patch(facecolor="#E74C3C", label="Negative Eigenvalues", alpha=0.9),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Adjust layout
    plt.tight_layout()
    if not show:
        return fig
    plt.show()
