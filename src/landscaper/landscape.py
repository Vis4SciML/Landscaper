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
from typing import Literal

from .compute import compute_loss_landscape
from .plots import contour, persistence_barcode, surface_3d, topology_profile, draw_tree
from .tda import get_persistence_dict, merge_tree, topological_index, PContourTree, saddle_minima_pairs
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
        self.ms_complex = None
        self.sub_tree = None
        self.super_tree = None
        self.contour_tree = None
        self.topological_indices = None
        self.loss_range = np.max(loss) - np.min(loss)

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
            self.sub_tree = merge_tree(self.loss, self.coords)
        return self.sub_tree

    def get_super_tree(self) -> tp.MergeTree:
        """Gets the merge tree corresponding to the maxima of the loss landscape.

        Returns:
            A tp.MergeTree object corresponding to the maxima of the loss landscape.
        """
        if self.super_tree is None:
            self.super_tree = merge_tree(self.loss, self.coords, direction=-1)
        return self.super_tree

    def get_contour_tree(self, **kwargs) -> PContourTree:
        """Returns the contour tree corresponding to the Landscape.

            **kwargs: Kwargs get forwarded to PContourTree.

        Returns:
            Contour tree representation of the landscape.
        """
        if self.contour_tree is None:
            ct = PContourTree(graph=ngl.EmptyRegionGraph(beta=1.0, relaxed=False, p=2.0), **kwargs)
            ct.build(np.array(self.coords), self.loss.flatten())
            ct.vals = [x for x in ct.sortedNodes if x[0] in ct.superNodes]
            ct.nodes = {n: dict(ct.vals)[n] for n in ct.superNodes}
            # TODO: figure out how to draw a contour tree profile
            self.contour_tree = ct
        return self.contour_tree

    def get_ms_complex(self) -> tp.MorseSmaleComplex:
        """Gets the MorseSmaleComplex corresponding to the loss landscape.

        Returns:
            A tp.MorseSmaleComplex.
        """
        if self.ms_complex is None:
            ms_complex = tp.MorseSmaleComplex(
                graph=ngl.EmptyRegionGraph(beta=1.0, relaxed=False, p=2.0),
                gradient="steepest",
            )
            ms_complex.build(np.array(self.coords), self.loss.flatten())
            self.ms_complex = ms_complex
        return self.ms_complex

    def get_topological_indices(self, mt) -> dict[int, int]:
        """Returns a dictionary that maps point indices to their topological indices.

        Returns:
            (dict[int, int]): A dictionary mapping point indices to their topological indices.
        """
        msc = self.get_ms_complex()
        ti = {}
        for n in mt.nodes:
            ti[n] = topological_index(msc, n)
        return ti

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

    def show_tree(self, tree_type: Literal["sublevel", "super"], **kwargs):
        """Draws the selected type of merge tree for the landscape.
        Can either be the sublevel (minima) or super (maxima) tree.

        See :obj:`landscaper.plots.draw_tree` for more details.
        """
        if tree_type == "sublevel":
            mt = self.get_sublevel_tree()
        else:
            mt = self.get_super_tree()
        return draw_tree(mt, **kwargs)

    def show_sublevel_tree(self, **kwargs):
        """Draws the sublevel merge tree of the landscape.
        See :obj:`landscaper.plots.draw_tree` for more details.
        """
        return self.show_tree("sublevel", **kwargs)

    def show_super_tree(self, **kwargs):
        """Draws the super merge tree of the landscape.
        See :obj:`landscaper.plots.draw_tree` for more details.
        """
        return self.show_tree("super", **kwargs)

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

    def index_with_basins(self, values, ordered=True):
        """Meant to be used with basin_metric to derive per-basin values. Gets the unstable manifolds from
        the landscape (i.e. points in a basin) and indexes into values
        .
        Args:
            values: Values to index into using the indices of points that belong to each basin.
            ordered: If true, basins are returned as a dictionary of minima index to a list of values; else its a 2D array

        Returns:
            Either a dict or a np.array depending on the settings for ordered.
        """
        assert values.shape == self.loss.shape
        um = self.get_ms_complex().get_unstable_manifolds()
        if ordered:
            return {k: values.flatten()[np.array(v)] for (k, v) in um.items()}
        return [values.flatten()[np.array(v)] for v in um.values()]

    def basin_metric(self, values, per_basin_fn=lambda x: np.mean(x), final_op=lambda x: np.mean(x)):
        """Use this function to build custom metrics for basins. You can pass any np.array to values as long
        as it matches the shape of the loss landscape. This function indexes into the values array correctly
        to get the corresponding values. See gnn.ipynb in the documentation for more details and ideas on what
        can be done with this function.

        Args:
            values: Function values to use for the metric.
            per_basin_fn: What form of aggregation (if any) is being used for each basin's values. Defaults to np.mean.
            final_op: How the final result is calculated; defaults to np.mean.

        Returns:
            A metric over all of the basins in the landscape.
        """
        msc = self.get_ms_complex()
        basins = self.index_with_basins(values)

        bv = np.zeros(len(msc.min_indices))
        for i, x in enumerate(msc.min_indices):
            bv[i] = per_basin_fn(basins[x])

        return final_op(bv)

    def smad(self, normalize: bool = False, weighted: bool = False) -> float:
        """Calculates the Saddle-Minimum Average Distance (SMAD) for the landscape.
        See our publication for more details.

        Args:
            normalize (boolean): If true, divides each saddle-minimum gap by the total range of the loss values.
            weighted (boolean): If true, weights each saddle-minimum gap by the volume of the basin created by the saddle-minimum pair.

        Returns:
            (float) A descriptor of the smoothness of the landscape.
        """
        mt = self.get_sublevel_tree()
        ti = self.get_topological_indices(mt)
        msc = self.get_ms_complex()

        if len(mt.branches) == 0:
            return 0.0

        tot = len(self.loss.flatten())
        # branch persistence
        um = msc.get_unstable_manifolds()
        sp = saddle_minima_pairs(mt, ti)
        bp = np.empty(len(sp))
        vol = np.empty(len(sp))

        for i, (n1, n2) in enumerate(sp):
            minima = n1 if (ti[n1] == 0) else n2
            gap = abs(mt.nodes[n1] - mt.nodes[n2])
            if normalize:
                gap = gap / self.loss_range
            vol[i] = len(um[minima]) / tot
            bp[i] = gap

        if not weighted:
            return np.mean(bp)

        bp = np.log(bp)
        vol = np.log(vol)

        x = bp + vol
        x = np.sum(np.exp(x)) / len(x)

        return x

    def persistence_range(self) -> float:
        """
        Calculates the difference in persistence between the root
        of the merge tree and the global minimum.
        """
        msc = self.get_ms_complex()
        p = msc.persistences
        return abs(p[-1] - p[0])
