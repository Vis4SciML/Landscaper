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
from topopy import ContourTree


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
    direction: Literal[-1, 1] = 1,
    beta: float = 1.0,
    relaxed: bool = False,
    p: float = 2.0,
    **kwargs,
) -> tp.MergeTree:
    """Helper function used to generate a merge tree for a loss landscape.

    Args:
        loss (np.ArrayLike): Function values for each point in the space.
        coords (np.ArrayLike): N-dimensional array of ranges for each dimension in the space.
        graph (ngl.nglGraph): nglpy graph of the space (usually filled out by topopy).
        direction (Literal[-1,1]): The direction to generate a merge tree for.
            -1 generates a merge tree for maxima, while the default value (1) is for minima.
        beta: Used for building the region graph for the merge tree computation. From the nglpy docs:
            A positive value working in conjunction with the p value to specify
            the size and shape of the empty region required around a valid
            edge. Roughly, beta is a size parameter and p is a shape parameter.
            For p values less than one, beta is inversely proportional to
            the size of the empty region, and for p values greater than one,
            beta is directly proportional to the size of the empty region.
            When p is one, the beta parameter is irrelevant.

            For an interactive example of how these values interact, see:
            http://www.cs.utah.edu/~maljovec/bpSkeleton.html
        relaxed: Used for building the region graph for the merge tree computation. From the nglpy docs:
            Determines whether the relaxed graph (as determined by Correa and
            Lindstrom's algorithm) should be computed. A relaxed edge
            satisifies the empty region criteria from the perspective of one
            endpoint, whereas a strict edge satisfies this criteria from the
            perspective of both endpoints.
        p: Used for building the region graph for the merge tree computation. From the nglpy docs:
            A positive value working in conjunction with the beta value to
            specify the shape and size of the empty region required around a
            valid edge. Thsi value specifies the Lp-norm to use for generating
            the empty regions. Roughly, beta is a size parameter and p is a
            shape parameter. For p values less than one, beta is inversely
            proportional to the size of the empty region, and for p values
            greater than one, beta is directly proportional to the size of the
            empty region. When p is one, the beta parameter is irrelevant.

            For an interactive example of how these values interact, see:
            http://www.cs.utah.edu/~maljovec/bpSkeleton.html
        The rest of the kwargs are passed to nglpy's EmptyRegionGraph. Look [here](https://github.com/maljovec/nglpy/blob/main/nglpy/EmptyRegionGraph.py) for more details.
    Returns:
        Merge tree for the space.
    """
    loss_flat = loss.flatten()
    t = tp.MergeTree(graph=ngl.EmptyRegionGraph(beta=beta, relaxed=relaxed, p=p, **kwargs))
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


def _mt_subtree_length(n, mtg):
    st = nx.dfs_tree(mtg, source=n)
    return len(st)


def _mt_get_children(n, mtg):
    # sort them such that n is always first
    return [n2 if n == n1 else n1 for (n1, n2) in mtg.out_edges(n)]


def tree_layout(
    t: tp.MergeTree, log_scale: bool = False, node_size: int = 300
) -> tuple[nx.DiGraph, dict[int, list[float]]]:
    """Lays out a mergetree according to the algorithm outlined in the paper
    ["Sketching Merge Trees for Scientific Data Visualization"](https://arxiv.org/abs/2101.03196).

    Args:
        t: The merge tree to lay out.
        log_scale: Whether or not the node values should be log scaled.
        node_size: The size of the nodes in pixels. Used during layout to prevent overlap.

    Returns:
        A networkx.DiGraph representation of the tree and a dict of node ids to lists of x,y positions.
    """
    G = tree_to_nx(t)

    def get_node_value(x):
        val = t.Y[x]
        if log_scale:
            return np.log(max(val, 1e-12))
        return val

    roots = [x for x in G if not G.in_edges(x)]
    roots.sort(key=lambda x: get_node_value(x))

    n = roots.pop()
    pos = {n: [0.0, get_node_value(n)]}
    visited = {n}
    s = [n]

    branch = 0
    while len(visited) != len(G):
        if not s:
            branch += 1
            n = roots.pop()
            s.append(n)
            visited.add(n)
            pos[n] = [branch, get_node_value(n)]

        n = s.pop()
        parent_x, parent_y = pos[n]
        children = _mt_get_children(n, G)
        info = []
        xs = []
        for n2 in children:
            x = _mt_subtree_length(n2, G)
            y = get_node_value(n2)
            c = G[n][n2]["counts"]
            xs.append(x)
            info.append((n2, x, y, c))

        ignore1 = [x for x in xs if x != 1]
        # check for duplicates - use 3a.) of algorithm
        if len(ignore1) == len(set(ignore1)):
            info.sort(key=lambda x: (x[1], x[3], x[0]))
        else:
            info.sort(key=lambda x: (x[2], x[1], x[0]))
            evens = [x for i, x in enumerate(info) if i % 2 == 0]  # evens
            odds = [x for i, x in enumerate(info) if i % 2 != 0]

            if len(info) % 2 == 0:
                sinfo = evens[::-1] + odds
            else:
                sinfo = odds[::-1] + evens
            info = sinfo

        for i, (n_id, x, y, c) in enumerate(info):
            pos[n_id] = [parent_x + i, y]

            if n_id not in visited:
                s.append(n_id)
        visited.add(n)

    return G, pos


def tree_to_nx(t: tp.MergeTree) -> nx.DiGraph:
    """Converts a merge tree to a directed graph representation that makes it easy to navigate the hierarchy.

    The root is the maximum which points down the tree towards saddles and minima.
    The 'partition' edge attributes list the members of the integral line from node a->b,
    while 'counts' contains the number of members along that line.

    Args:
        t (tp.MergeTree): The merge tree to convert.

    Returns:
        A networkx DiGraph representation of the merge tree hierarchy.
    """
    g = nx.DiGraph()
    for n, v in t.nodes.items():
        g.add_node(n, value=v)

    g.add_edges_from([(e[1], e[0]) for e in list(t.edges)])
    e_info = {(e[1], e[0]): [e[0]] for e in list(t.edges)}
    aug_info = {(e[1], e[0]): [e[0], *v] for e, v in t.augmentedEdges.items()}
    e_info.update(aug_info)

    nx.set_edge_attributes(g, e_info, "partitions")
    len_part_dict = {e: len(v) for e, v in e_info.items()}
    nx.set_edge_attributes(g, len_part_dict, "counts")

    # Remove isolated nodes
    # isolated_nodes = list(nx.isolates(g))
    # g.remove_nodes_from(isolated_nodes)
    return g


def saddle_minima_pairs(ct: ContourTree | tp.MergeTree, ti: dict[int, int]) -> list[tuple[int, int]]:
    """Iterates through the edges in a topological tree and returns all edges that are saddle-minima pairs.

    Args:
        ct: The contour tree to get saddle minima pairs for.
        ti: Dictionary of topological indices for each critical point; use LossLandscape.get_topological indices() to
        get this from the landscape.

    Returns:
        A list of saddle-minima pairs.
    """
    pairs = []
    for e in list(ct.edges):
        n1, n2 = e
        for b in ct.branches:
            if (b == n1 and ti[n2] == 0) or (b == n2 and ti[n1] == 0):
                pairs.append((n1, n2))
                break
    return pairs
