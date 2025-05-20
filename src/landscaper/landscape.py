import nglpy as ngl
import numpy as np
import numpy.typing as npt
import topopy as tp

from .compute import compute_loss_landscape
from .plots import contour, persistence_barcode, surface_3d, topology_profile
from .tda import (
    extract_mergetree,
    get_persistence_dict,
    merge_tree,
)
from .topology_profile import generate_profile
from .utils import load_landscape


class LossLandscape:
    @staticmethod
    def compute(*args, **kwargs):
        """Computes a loss landscape and directly creates a LossLandscape object. See `landscaper.compute` for more information.

        Returns:
            (LossLandscape) A LossLandscape object.
        """
        top_eigenvalues, top_eigenvectors, loss, coords = compute_loss_landscape(
            *args, **kwargs
        )
        return LossLandscape(loss, coords)

    @staticmethod
    def load_from_npz(fp: str):
        """Creates a LossLandscape object directly from an `.npz` file.

        Args:
            fp (str): path to the file.

        Returns:
            A LossLandscape object.
        """
        loss, coords = load_landscape(fp)
        return LossLandscape(loss, coords)

    def __init__(self, loss: npt.ArrayLike, ranges: npt.ArrayLike):
        self.loss = loss
        # converts meshgrid output of arbitrary dimensions into list of coordinates
        grid = np.meshgrid(*ranges)
        self.coords = np.array(
            [list(z) for z in zip(*(x.flat for x in grid), strict=False)]
        )

        if self.coords.shape[0] != np.multiply.reduce(self.loss.shape):
            raise ValueError(
                f"Loss dimensions do not match coordinate dimensions: Loss - {self.loss.shape}; Coordinates - {self.coords.shape}"
            )

        self.ranges = ranges
        self.dims = self.coords.shape[1]
        self.graph = ngl.EmptyRegionGraph(beta=1.0, relaxed=False, p=2.0)
        self.ms_complex = None
        self.merge_tree = None
        self.sub_tree = None
        self.super_tree = None

    def save(self, filename: str):
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
            self.super_tree = merge_tree(
                self.loss, self.coords, self.graph, direction=-1
            )
        return self.super_tree

    def get_ms_complex(self) -> tp.MorseSmaleComplex:
        """Gets the MorseSmaleComplex corresponding to the loss landscape.

        Returns:
            A tp.MorseSmaleComplex.
        """
        if self.ms_complex is None:
            ms_complex = tp.MorseSmaleComplex(
                graph=self.graph, gradient="steepest", normalization="feature"
            )
            ms_complex.build(np.array(self.coords), self.loss.flatten())
            self.ms_complex = ms_complex
        return self.ms_complex

    def get_persistence(self):
        """Returns the persistence of the landscape as a dictionary."""
        return get_persistence_dict(self.get_ms_complex())

    def show(self, **kwargs):
        """Renders a 3D representation of the loss landscape. See :obj:`landscaper.plots.surface_3d` for keyword arguments.

        Raises:
            ValueError: Thrown if the landscape has too many dimensions.
        """
        if self.dims == 2:
            return surface_3d(self.ranges, self.loss, **kwargs)
        else:
            raise ValueError(
                f"Cannot visualize a landscape with {self.dims} dimensions."
            )

    def show_profile(self, **kwargs):
        """Renders the topological profile of the landscape. See :obj:`landscaper.plots.topological_profile` for more details."""
        msc = self.get_ms_complex()
        mt = self.get_sublevel_tree()

        segInfo, mergeInfo, edgeInfo = extract_mergetree(msc, mt, self.loss)
        profile = generate_profile(segInfo, mergeInfo, edgeInfo)
        return topology_profile(profile, **kwargs)

    def show_contour(self, **kwargs):
        """Renders a contour plot of the landscape. See :obj:`landscaper.plots.contour` for more details."""
        return contour(self.ranges, self.loss)

    def show_persistence_barcode(self):
        """Renders the persistence barcode of the landscape. See :obj:`landscaper.plots.persistence_barcode` for more details."""
        msc = self.get_ms_complex()
        return persistence_barcode(msc)
