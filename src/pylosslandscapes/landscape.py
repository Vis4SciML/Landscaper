import nglpy as ngl
import numpy as np
import topopy as tp
import pandas as pd

from .math import (
    JointTree,
    get_persistence_dict,
    merge_tree,
    minima_and_nodes,
    extract_mergetree,
)
from .utils import load_landscape
from .topology_profile import generate_profile
from .plots import plot_topology_profile, plot_3d_surface
from .compute import compute_loss_landscape


class LossLandscape:
    @staticmethod
    def compute(*args, **kwargs):
        top_eigenvalues, top_eigenvectors, loss, coords = compute_loss_landscape(
            *args, **kwargs
        )
        return LossLandscape(loss, coords)

    @staticmethod
    def load_from_npz(fp):
        loss, coords = load_landscape(fp)
        return LossLandscape(loss, coords)

    def __init__(self, loss, ranges):
        self.loss = loss
        # converts meshgrid output of arbitrary dimensions into list of coordinates
        grid = np.meshgrid(*ranges)
        self.coords = np.array([list(z) for z in zip(*(x.flat for x in grid))])

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
        self.minima_and_nodes = None

    def show(self, **kwargs):
        if self.dims == 2:
            plot_3d_surface(self.ranges, self.loss, **kwargs)
        else:
            raise ValueError(
                f"Cannot visualize a landscape with {self.dims} dimensions."
            )

    def get_sublevel_tree(self):
        if self.sub_tree is None:
            self.sub_tree = merge_tree(self.loss, self.coords, self.graph)
        return self.sub_tree

    def get_super_tree(self):
        if self.super_tree is None:
            self.super_tree = merge_tree(
                self.loss, self.coords, self.graph, direction=-1
            )
        return self.super_tree

    def get_merge_tree(self):
        """Computes merge trees and manually constructs the joint tree."""
        if self.merge_tree is None:
            sub_tree = self.get_sublevel_tree()
            super_tree = self.get_super_tree()

            self.merge_tree = JointTree(sub_tree, super_tree, self.coords, self.loss)
            self.minima_and_nodes = minima_and_nodes(self.merge_tree.get_tree())

        return self.merge_tree  # Store as a dictionary

    def get_minima(self):
        if self.minima_and_nodes is None:
            self.get_merge_tree()  # will set minima
        return self.minima_and_nodes

    def get_ms_complex(self):
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

    def show_profile(self, **kwargs):
        msc = self.get_ms_complex()
        mt = self.get_sublevel_tree()

        segInfo, mergeInfo, edgeInfo = extract_mergetree(msc, mt, self.loss)
        profile = generate_profile(segInfo, mergeInfo, edgeInfo)
        return plot_topology_profile(profile, **kwargs)
