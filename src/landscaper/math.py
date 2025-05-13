# can be renamed to tda.py later, keeping it as math for now for compatibility purposes

import gudhi as gd
import numpy as np
from treelib import Tree
import topopy as tp
import pandas as pd


def minima_and_nodes(tree_dict):
    """Returns a dictionary with the leaf nodes as keys and the minima as values."""
    # Find leaf nodes (nodes that only appear as the second node in edges)
    edges = tree_dict["sub_edges"] + tree_dict["super_edges"]
    leaf_nodes = set()
    for edge in edges:
        if edge[1] not in leaf_nodes:
            leaf_nodes.add(edge[1])
        elif edge[0] in leaf_nodes:
            leaf_nodes.remove(edge[0])

    # Create dictionary of leaf nodes and their function values
    sorted_tree = {leaf: tree_dict["nodes"][leaf][1] for leaf in leaf_nodes}
    return dict(sorted(sorted_tree.items(), key=lambda item: item[1]))


def compute_bottleneck_distance(l1, l2):
    """Computes the bottleneck distance between two loss landscapes."""
    p1 = l1.get_persistence()
    p2 = l2.get_persistence()

    d = bottleneck_distance(p1, p2)

    return d


def bottleneck_distance(persistence1, persistence2):
    """Calculates the bottleneck distance between two persistence diagrams."""
    ppd1 = [(0, val) for val in persistence1]
    ppd2 = [(0, val) for val in persistence2]
    return gd.bottleneck_distance(ppd1, ppd2)


def minima_variance(landscapes):
    """Calculates the variance of minima across a list of trees."""
    minima_list = []
    tree_list = [x.get_minima() for x in landscapes]
    for tree in tree_list:
        if len(tree) == 1:
            minima_list.append(list(tree.values())[0])
        else:
            minima_list.append(np.mean(list(tree.values())))
    return np.var(minima_list)


def get_persistence_dict(msc):
    """Returns the persistence of the tree as a dictionary."""
    return {key: msc.get_merge_sequence()[key][0] for key in msc.get_merge_sequence()}


# not really necessary
def get_persistence(msc):
    """Returns the persistence of the tree as a list."""
    return list(get_persistence_dict(msc).values())


def build_tree_structure(merge_tree_dict):
    """Builds a tree structure from a merge tree dictionary."""
    tree = Tree()

    # Get nodes and their values
    nodes = merge_tree_dict["nodes"]

    # Find root node (node with highest function value)
    root_node = max(nodes.items(), key=lambda x: x[1])[0]
    tree.create_node(tag=f"Root: {root_node}", identifier=root_node)

    # Add branch nodes (nodes that appear in edges but aren't leaves)
    edges = merge_tree_dict["sub_edges"] + merge_tree_dict["super_edges"]
    branch_nodes = set()
    leaf_nodes = set()

    # First pass: identify branch and leaf nodes
    for edge in edges:
        for node in edge:
            if node not in branch_nodes and node not in leaf_nodes:
                branch_nodes.add(node)
            elif node in leaf_nodes:
                branch_nodes.add(node)
                leaf_nodes.remove(node)

    # Add branch nodes to tree
    for branch in branch_nodes:
        if branch != root_node:
            tree.create_node(
                tag=f"Branch: {branch}", identifier=branch, parent=root_node
            )

    # Add leaf nodes to tree
    for leaf in leaf_nodes:
        # Find parent (node connected by edge)
        for edge in edges:
            if leaf in edge:
                parent = edge[0] if edge[1] == leaf else edge[1]
                break
        tree.create_node(tag=f"Leaf: {leaf}", identifier=leaf, parent=parent)

    return tree


def merge_tree(loss, coords, graph, direction=1):
    loss_flat = loss.flatten()
    t = tp.MergeTree(graph=graph)
    t.build(np.array(coords), direction * loss_flat)
    return t


def topological_index(msc, idx):
    c = msc.get_classification(idx)
    if c == "minimum":
        return 0
    elif c == "regular":
        return 1
    else:
        return 2


def extract_mergetree(msc, mt, vals):
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


class JointTree:
    """Class to store and represent the joint tree."""

    def __init__(self, sub_tree, super_tree, coords, vals):
        self.nodes = {}  # {id: (x, y, function value)}
        self.sub_edges = []  # Sublevel tree edges (normal function)
        self.super_edges = []  # Superlevel tree edges (negated function)

        fvals = vals.flatten()

        all_nodes = set()
        for node in sub_tree.nodes:
            all_nodes.add(node)
        for node in super_tree.nodes:
            all_nodes.add(node)

        # Add nodes with their coordinates and function values
        for node_id in all_nodes:
            self.add_node(node_id, coords[node_id], fvals[node_id])

        # Add edges from both trees
        for edge in sub_tree.edges:
            self.add_sub_edge(edge[0], edge[1])

        for edge in super_tree.edges:
            self.add_super_edge(edge[0], edge[1])

    def add_node(self, node_id, coords, f_value):
        self.nodes[node_id] = (coords, f_value)

    def add_sub_edge(self, node1, node2):
        self.sub_edges.append((node1, node2))

    def add_super_edge(self, node1, node2):
        self.super_edges.append((node1, node2))

    def get_tree(self):
        """Returns a dictionary representation of the joint tree."""
        return {
            "nodes": self.nodes,
            "sub_edges": self.sub_edges,
            "super_edges": self.super_edges,
        }
