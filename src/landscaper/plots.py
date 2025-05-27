import drawsvg as dw
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import topopy as tp
from coloraide import Color
from matplotlib.colors import LogNorm

from .tda import get_persistence_dict
from .utils import Number


def persistence_barcode(msc: tp.MorseSmaleComplex):
    """Plots the [persistence barcode](https://en.wikipedia.org/wiki/Persistence_barcode)  for a Morse-Smale complex.

    Args:
        msc (tp.MorseSmaleComplex): A Morse-Smale complex.
    """

    node_list = [str(node) for node in list(get_persistence_dict(msc).keys())]
    persistence_list = list(get_persistence_dict(msc).values())
    plt.barh(node_list, persistence_list)
    plt.xlabel("Persistence")
    plt.ylabel("Node")
    plt.title("Node vs Persistence")
    plt.show()


def linearScale(min_val: Number, max_val: Number, new_min: Number, new_max: Number):
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


def topology_profile(
    data,
    y_min: float | None = None,
    y_max: float | None = None,
    size: int = 800,
    margin: int = 15,
    color: str = "red",
    background_color: str = "white",
    gradient: bool = True,
):
    """Renders a topological profile for the given merge tree data extracted with `extract_merge_tree` from `landscaper.tda`.

    Args:
        data (List[List[float]]): The merge tree data.
        y_min (Optional[float]): Optional minimum y value for the drawing.
        y_max (Optional[float]): Optional maximum y value for the drawing.
        size (int): Size in pixels of the resulting drawing.
        margin (int): Size of the margins in pixels.
        color (str): Color used to draw the profile.
        background_color (str): Color used to draw the background.
        gradient (bool): If true, fills the profile using a gradient from `background_color` to `color`. If false, only uses `color` to fill the path. Set this to false if you are exporting the drawing into a different format.
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
    svg.append(
        dw.Rectangle(0, 0, width, height, fill="white", stroke="#777")
    )  # background color

    for d in data:
        yVals = [pt[1] for pt in d]
        minY = min(yVals)
        maxY = max(yVals)

        if gradient:
            grad = dw.LinearGradient(
                "0%", "100%", "0%", "0%", gradientUnits="objectBoundingBox"
            )

            for t in np.linspace(0.0, 1.0, 100):
                yValue = minY + t * (maxY - minY)
                grad.add_stop(
                    f"{t * 100}%", basinColors(yValue).to_string(hex=True, upper=True)
                )
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
    return svg

    """
        // Add the y-axis (removed grid lines and changed tick format from .1e to .1f)
        svg.append("g")
            .attr("transform", `translate(${marginLeft},0)`)
            .call(d3.axisLeft(yScale).ticks(10, ".1f"))
            .style("font-size", "14px");

        // Update y-axis label with larger font and better positioning
        svg.append("text")
            .attr("x", -height / 2) // For vertical text, start from middle of height
            .attr("y", 15) // Move further left from axis (smaller number moves it left)
            .attr("transform", "rotate(-90)") // Rotate text to be vertical
            .attr("text-anchor", "middle") // Center the text
            .style("font-size", "18px") // Larger font size
            .text("Function Value");

        // Add title with truncated filename
        svg.append("text")
            .attr("x", width / 2)
            .attr("y", marginTop - 15)
            .attr("text-anchor", "middle")
            .style("font-size", "18px")
            .style("font-weight", "bold")
            .text(
                file
                    .replace("_TP.json", "")
                    .split("UnstructuredGrid_rknn")[0]
            );

 
    """


def contour(
    coordinates: npt.ArrayLike, loss: npt.ArrayLike, figsize: tuple[int, int] = (12, 8)
):
    """Draws a contour plot from the provided coordinates and values.

    Args:
        coordinates (npt.ArrayLike): n-dimensional coordinates.
        values (npt.ArrayLike): Value for each coordinate.
        figsize (tuple[int,int]): Size of the figure.

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


def surface_3d(
    coords: npt.ArrayLike, loss: npt.ArrayLike, figsize: tuple[int, int] = (12, 8)
):
    """Generates a 3d surface plot for the given coordinates and values. Fails if dimensions are greater than 2.

    Args:
        coords (npt.ArrayLike): 2-D coordinates.
        loss (npt.ArrayLike): Values for the coordinates.
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
        surf = ax.plot_surface(
            X, Y, loss, cmap="RdYlBu_r", linewidth=0, antialiased=True
        )
        plt.colorbar(surf, label="Loss")

    ax.set_xlabel("Direction of First Eigenvector")
    ax.set_ylabel("Direction of Second Eigenvector")
    ax.set_zlabel("Loss")
    ax.set_title("3D Loss Landscape")

    # Adjust the viewing angle for better visualization
    ax.view_init(elev=30, azim=45)

    plt.show()
