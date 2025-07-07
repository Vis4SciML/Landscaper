"""This module provides functions to compute the loss landscape of a model and visualize it in various ways.

It includes methods for computing the loss landscape, loading it from a file, and
visualizing it as a 3D surface, contour plot, or persistence barcode.
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

import nglpy as ngl
import numpy as np
import numpy.typing as npt
import topopy as tp

from .compute import compute_loss_landscape
from .plots import contour, persistence_barcode, surface_3d, topology_profile
from .tda import get_persistence_dict, merge_tree, topological_index
from .topology_profile import generate_profile
from .utils import load_landscape


class LossLandscape:
    """A class representing a loss landscape of a model.

    It contains methods to compute the landscape, visualize it, and analyze its topological properties.
    """

    @staticmethod
    def compute(*args, **kwargs) -> "LossLandscape":
        """Computes a loss landscape and directly creates a LossLandscape object.

        See `landscaper.compute` for more information.

        Returns:
            (LossLandscape) A LossLandscape object.
        """
        loss, coords = compute_loss_landscape(*args, **kwargs)
        return LossLandscape(loss, coords)

    @staticmethod
    def load_from_npz(fp: str) -> "LossLandscape":
        """Creates a LossLandscape object directly from an `.npz` file.

        Args:
            fp (str): path to the file.

        Returns:
            (LossLandscape) A LossLandscape object created from the file.
        """
        loss, coords = load_landscape(fp)
        return LossLandscape(loss, coords)

    def __init__(self, loss: npt.ArrayLike, ranges: npt.ArrayLike) -> None:
        """Initializes a LossLandscape object.

        Args:
            loss (npt.ArrayLike): A numpy array representing the loss values of the landscape.
            ranges (npt.ArrayLike): A list of numpy arrays representing the ranges of each dimension of the landscape.

        Raises:
            ValueError: If the dimensions of the loss array do not match the number of coordinates.
        """
        self.loss = loss
        # converts meshgrid output of arbitrary dimensions into list of coordinates
        grid = np.meshgrid(*ranges)
        self.coords = np.array([list(z) for z in zip(*(x.flat for x in grid), strict=False)])

        if self.coords.shape[0] != np.multiply.reduce(self.loss.shape):
            raise ValueError(
                f"Loss dimensions do not match coordinate dimensions: Loss - {self.loss.shape}; "
                f"Coordinates - {self.coords.shape}"
            )

        self.ranges = ranges
        self.dims = self.coords.shape[1]
        self.graph = ngl.EmptyRegionGraph(beta=1.0, relaxed=False, p=2.0)
        self.ms_complex = None
        self.sub_tree = None
        self.super_tree = None
        self.topological_indices = None

    def save(self, filename: str) -> None:
        """Saves the loss and coordinates of the landscape to the specified path for later use.

        Args:
            filename (str): path to save the landscape to.
        """
        np.savez(filename, loss=self.loss, coordinates=self.ranges)

    def get_sublevel_tree(self) -> tp.MergeTree:
        """Gets the merge tree corresponding to the minima of the loss landscape.

        Returns:
            A tp.MergeTree object corresponding to the minima of the loss landscape.
        """
        if self.sub_tree is None:
            self.sub_tree = merge_tree(self.loss, self.coords, self.graph)
        return self.sub_tree

    def get_super_tree(self) -> tp.MergeTree:
        """Gets the merge tree corresponding to the maxima of the loss landscape.

        Returns:
            A tp.MergeTree object corresponding to the maxima of the loss landscape.
        """
        if self.super_tree is None:
            self.super_tree = merge_tree(self.loss, self.coords, self.graph, direction=-1)
        return self.super_tree

    def get_ms_complex(self) -> tp.MorseSmaleComplex:
        """Gets the MorseSmaleComplex corresponding to the loss landscape.

        Returns:
            A tp.MorseSmaleComplex.
        """
        if self.ms_complex is None:
            ms_complex = tp.MorseSmaleComplex(graph=self.graph, gradient="steepest", normalization="feature")
            ms_complex.build(np.array(self.coords), self.loss.flatten())
            self.ms_complex = ms_complex
        return self.ms_complex

    def get_topological_indices(self) -> dict[int, int]:
        """Returns a dictionary that maps point indices to their topological indices.

        Returns:
            (dict[int, int]): A dictionary mapping point indices to their topological indices.
        """
        msc = self.get_ms_complex()
        mt = self.get_sublevel_tree()
        if self.topological_indices is None:
            ti = {}
            for n in mt.nodes:
                ti[n] = topological_index(msc, n)
            self.topological_indices = ti
        return self.topological_indices

    def get_persistence(self):
        """Returns the persistence of the landscape as a dictionary."""
        return get_persistence_dict(self.get_ms_complex())

    def show(self, **kwargs):
        """Renders a 3D representation of the loss landscape.

        See :obj:`landscaper.plots.surface_3d` for keyword arguments.

        Raises:
            ValueError: Thrown if the landscape has too many dimensions.
        """
        if self.dims == 2:
            return surface_3d(self.ranges, self.loss, **kwargs)
        else:
            raise ValueError(f"Cannot visualize a landscape with {self.dims} dimensions.")

    def show_profile(self, **kwargs):
        """Renders the topological profile of the landscape.

        See :obj:`landscaper.plots.topological_profile` for more details.
        """
        mt = self.get_sublevel_tree()
        profile = generate_profile(mt)
        return topology_profile(profile, **kwargs)

    def show_contour(self, **kwargs):
        """Renders a contour plot of the landscape.

        See :obj:`landscaper.plots.contour` for more details.
        """
        return contour(self.ranges, self.loss, **kwargs)

    def show_persistence_barcode(self, **kwargs):
        """Renders the persistence barcode of the landscape.

        See :obj:`landscaper.plots.persistence_barcode` for more details.
        """
        msc = self.get_ms_complex()
        return persistence_barcode(msc, **kwargs)

    def smad(self) -> float:
        """Calculates the Saddle-Minimum Average Distance (SMAD) for the landscape.

        See our publication for more details.

        Returns:
            (float) A descriptor of the smoothness of the landscape.
        """
        mt = self.get_sublevel_tree()
        ti = self.get_topological_indices()

        if len(mt.branches) == 0:
            return 0.0

        # branch persistence
        bp = []
        for b in mt.branches:
            for edge in list(mt.edges):
                n1, n2 = edge
                if (b == n1 and ti[n2] == 0) or (b == n2 and ti[n1] == 0):
                    bp.append(abs(mt.nodes[n1] - mt.nodes[n2]))
        m = len(bp)

        return sum(bp) / m
