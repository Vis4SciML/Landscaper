import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import drawsvg as dw

from .math import get_persistence_dict
import sys

from coloraide import Color
import numpy as np


def persistence_barcode(msc):
    """Plots the persistence barcode for a Morse-Smale complex."""
    node_list = [str(node) for node in list(get_persistence_dict(msc).keys())]
    persistence_list = list(get_persistence_dict(msc).values())
    plt.barh(node_list, persistence_list)
    plt.xlabel("Persistence")
    plt.ylabel("Node")
    plt.title("Node vs Persistence")
    plt.show()


def linearScale(min_val, max_val, new_min, new_max):
    return lambda x: (new_max - new_min) / (max_val - min_val) * (x - max_val) + new_max


def plot_topology_profile(
    data, y_min=None, y_max=None, output_path=None, size=800, margin=15
):
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

    # points itself is a dictionary
    # each point is a dictionary

    for d in data:
        # area is a list of points
        area = d["area"]
        xVals = [pt[0] for pt in area]
        yVals = [pt[1] for pt in area]

        x_max = max(x_max, max(xVals))
        x_min = min(x_min, min(xVals))
        loss_max = max(loss_max, max(yVals))
        loss_min = min(loss_min, min(yVals))

    # keep colors consistent regardless of y min and max chosen
    basinColors = Color.interpolate(
        ["red", "orange", "yellow", "green", "blue", "purple"],
        domain=[max(loss_min, 1e-10), loss_max],
    )

    if y_max is not None:
        loss_max = y_max

    if y_min is not None:
        loss_min = y_min

    print(f"X-axis range: {x_min}, {x_max}")
    print(f"Y-axis range: {loss_min}, {loss_max}")

    xScale = linearScale(x_min, x_max, marginLeft, width - marginRight)
    yScale = linearScale(loss_min, loss_max, height - marginBottom, marginTop)

    svg = dw.Drawing(width, height)
    svg.append(
        dw.Rectangle(0, 0, width, height, fill="white", stroke="#777")
    )  # background color

    for d in data:
        area = d["area"]

        yVals = [pt[1] for pt in area]
        minY = min(yVals)
        maxY = max(yVals)

        grad = dw.LinearGradient(
            "0%", "100%", "0%", "0%", gradientUnits="objectBoundingBox"
        )

        for t in np.linspace(0.0, 1.0, 100):
            yValue = minY + t * (maxY - minY)
            grad.add_stop(
                f"{t * 100}%", basinColors(yValue).to_string(hex=True, upper=True)
            )

        path = dw.Path(stroke=grad, fill=grad)
        start, *pts = area
        sx, sy = start
        path.M(xScale(sx), yScale(sy))
        for pt in pts:
            x, y = pt
            path.L(xScale(x), yScale(y))
        svg.append(path)
    return svg

    """
        // Add the x-axis.
        svg.append("g")
            .attr("transform", `translate(0,${height - marginBottom})`)
            .call(d3.axisBottom(xScale))
            .style("font-size", "14px"); // Larger font size

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


def plot_contour(coordinates, loss):
    fig = plt.figure(figsize=(10, 8))
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


def plot_3d_surface(coords, loss, figsize=(12, 8)):
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
