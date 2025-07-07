"""This module contains for topological data analysis (TDA) of loss landscapes."""

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

from typing import Literal

import gudhi as gd
import networkx as nx
import nglpy as ngl
import numpy as np
import numpy.typing as npt
import topopy as tp


def bottleneck_distance(p1: npt.ArrayLike, p2: npt.ArrayLike) -> float:
    """Calculates the bottleneck distance between two persistence diagrams.

    Args:
        p1 (npt.ArrayLike): The first persistence diagram, represented as a 1D array of persistence values.
        p2 (npt.ArrayLike): The second persistence diagram, represented as a 1D array of persistence values.

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
    graph: ngl.EmptyRegionGraph,
    direction: Literal[-1, 1] = 1,
) -> tp.MergeTree:
    """Helper function used to generate a merge tree for a loss landscape.

    Args:
        loss (np.ArrayLike): Function values for each point in the space.
        coords (np.ArrayLike): N-dimensional array of ranges for each dimension in the space.
        graph (ngl.nglGraph): nglpy graph of the space (usually filled out by topopy).
        direction (Literal[-1,1]): The direction to generate a merge tree for.
            -1 generates a merge tree for maxima, while the default value (1) is for minima.

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


def merge_tree_to_nx(mt: tp.MergeTree) -> nx.Graph:
    """Converts a Topopy MergeTree to a networkx representation.

    Args:
        mt (tp.MergeTree): The merge tree to convert.

    Returns:
       A networkx Graph representation of the merge tree. Can be used for visualization and analysis.
    """
    g = nx.Graph()
    for n, v in mt.nodes.items():
        g.add_node(n, value=v)
    g.add_edges_from(list(mt.edges))
    return g


def digraph_mt(mt: tp.MergeTree) -> nx.DiGraph:
    """Converts a merge tree to a directed graph representation that makes it easy to navigate the hierarchy.

    The root is the maximum which points down the tree towards saddles and minima.
    The 'partition' edge attributes list the members of the integral line from node a->b,
    while 'counts' contains the number of members along that line.

    Args:
        mt (tp.MergeTree): The merge tree to convert.

    Returns:
        A networkx DiGraph representation of the merge tree hierarchy.
    """
    g = nx.DiGraph()
    for n, v in mt.nodes.items():
        g.add_node(n, value=v)

    g.add_edges_from([(e[1], e[0]) for e in list(mt.edges)])
    e_info = {(e[1], e[0]): [e[0]] for e in list(mt.edges)}
    aug_info = {(e[1], e[0]): [e[0], *v] for e, v in mt.augmentedEdges.items()}
    e_info.update(aug_info)

    nx.set_edge_attributes(g, e_info, "partitions")
    len_part_dict = {e: len(v) for e, v in e_info.items()}
    nx.set_edge_attributes(g, len_part_dict, "counts")

    return g
