# can be renamed to tda.py later, keeping it as math for now for compatibility purposes

from typing import Literal

import gudhi as gd
import nglpy as ngl
import numpy as np
import numpy.typing as npt
import pandas as pd
import topopy as tp

from .landscape import LossLandscape


def compute_bottleneck_distance(l1: LossLandscape, l2: LossLandscape) -> float:
    """Computes the [bottleneck distance](https://mtsch.github.io/PersistenceDiagrams.jl/v0.3/generated/distances/) between two loss landscapes by first computing their persistence diagrams.

    Args:
        l1 (LossLandscape): First loss landscape.
        l2 (LossLandscape): Second loss landscape.

    Returns:
        A float representing the bottleneck distance between the two landscapes.
    """
    """"""
    p1 = l1.get_persistence()
    p2 = l2.get_persistence()

    return bottleneck_distance(list(p1.values()), list(p2.values))


def bottleneck_distance(p1: npt.ArrayLike, p2: npt.ArrayLike) -> float:
    """
    Calculates the bottleneck distance between two persistence diagrams.
        Args:
            persistence1 (list[float]): Persistence diagram values.
            persistence2 (list[float]): Persistence diagram values.

        Returns:
            A float representing the bottleneck distance between the two diagrams.
    """
    ppd1 = [(0, val) for val in p1]
    ppd2 = [(0, val) for val in p2]
    return gd.bottleneck_distance(ppd1, ppd2)


def get_persistence_dict(msc: tp.MorseSmaleComplex):
    """Returns the persistence of the given Morse-Smale complex as a dictionary.

    Args:
        msc (tp.MorseSmaleComplex): A Morse-Smale complex from [Topopy](https://github.com/maljovec/topopy).

    Returns:
        The values of the Morse-Smale Complex as a dictionary of nodes to persistence values.
    """
    """"""
    return {key: msc.get_merge_sequence()[key][0] for key in msc.get_merge_sequence()}


def merge_tree(
    loss: npt.ArrayLike,
    coords: npt.ArrayLike,
    graph: ngl.nglGraph,
    direction: Literal[-1, 1] = 1,
) -> tp.MergeTree:
    """Helper function used to generate a merge tree for a loss landscape.

    Args:
        loss (np.ArrayLike): Function values for each point in the space.
        coords (np.ArrayLike): N-dimensional array of ranges for each dimension in the space.
        graph (ngl.nglGraph): nglpy graph of the space (usually filled out by topopy).
        direction (Literal[-1,1]): The direction to generate a merge tree for. -1 generates a merge tree for maxima, while the default value (1) is for minima.

    Returns:
        Merge tree for the space.
    """
    loss_flat = loss.flatten()
    t = tp.MergeTree(graph=graph)
    t.build(np.array(coords), direction * loss_flat)
    return t


def topological_index(msc: tp.MorseSmaleComplex, idx: int) -> Literal[0, 1, 2]:
    """Gets the topological index of a given point.

    Args:
        msc (tp.MorseSmaleComplex): The Morse-Smale complex that represents the space being analyzed.
        idx (int): The index of the point to get a topological index for.

    Returns:
        Either 0, 1, or 2. This indicates that the point is either a minima (0), saddle point (1) or a maxima (2).
    """
    c = msc.get_classification(idx)
    if c == "minimum":
        return 0
    elif c == "regular":
        return 1
    else:
        return 2


def extract_mergetree(
    msc: tp.MorseSmaleComplex, mt: tp.MergeTree, vals: npt.ArrayLike
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Converts a merge tree into a representation that can be used to generate a topological profile.

    Args:
        msc (tp.MorseSmaleComplex): Morse-Smale representation of the space.
        mt (tp.MergeTree): Merge tree of the space.
        vals (npt.ArrayLike): Function values of the space.

    Returns:
        Tuple of dataframes for profile generation.
    """
    fv = vals.flatten()
    seg = mt.augmentedEdges
    segmentation = np.zeros((len(fv), 2))

    e_to_p = {e: i for i, e in enumerate(list(mt.augmentedEdges.keys()))}

    for e, idxes in seg.items():
        for idx in idxes:
            segmentation[idx] = [fv[idx], e_to_p[e]]
    s_df = pd.DataFrame(np.array(segmentation), columns=["Loss", "SegmentationId"])

    s_df = s_df.astype({"SegmentationId": np.int16})

    mergeInfo = []
    for nID, val in mt.nodes.items():
        mergeInfo.append([nID, val, topological_index(msc, nID)])

    midf = pd.DataFrame(
        np.array(mergeInfo), columns=["NodeId", "Scalar", "CriticalType"]
    )
    midf = midf.astype({"NodeId": np.int16, "CriticalType": np.uint8})
    midf = midf.sort_values(by="Scalar")

    edgeInfo = []
    for e in mt.augmentedEdges:
        n1, n2 = e
        segID = e_to_p[e]
        edgeInfo.append([segID, n2, n1])

    eidf = pd.DataFrame(
        np.array(edgeInfo), columns=["SegmentationId", "upNodeId", "downNodeId"]
    )

    return s_df, midf, eidf
