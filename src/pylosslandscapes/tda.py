# Standard library imports
import random

import gudhi as gd
import matplotlib.pyplot as plt
import nglpy as ngl

# Third-party imports
import numpy as np
import pandas as pd
import topopy as tp
import torch
from treelib import Tree


class JointTree:
    """Class to store and represent the joint tree."""

    def __init__(self):
        self.nodes = {}  # {id: (x, y, function value)}
        self.sub_edges = []  # Sublevel tree edges (normal function)
        self.super_edges = []  # Superlevel tree edges (negated function)

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


class TDAUtils:
    """A utility class for Topological Data Analysis operations."""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize TDAUtils with device configuration."""
        self.device = device
        self.has_cuda = torch.cuda.is_available()
        self._set_random_seeds()

    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        np.random.seed(0)
        random.seed(0)

    @staticmethod
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

    @staticmethod
    def minima_list(tree):
        """Returns a list of the minima of the tree."""
        return [tree.get_y(leaf)[0] for leaf in tree.leaves]

    @staticmethod
    def leaf_nodes(tree):
        """Returns a list of the leaf nodes of the tree."""
        return [leaf for leaf in tree.leaves]

    @staticmethod
    def get_value(tree_dict, index):
        """Returns the value of the leaf node at the specified index."""
        list_vals = list(tree_dict.values())
        return list_vals[index]

    @staticmethod
    def minima_variance(tree_list):
        """Calculates the variance of minima across a list of trees."""
        minima_list = []
        for tree in tree_list:
            if len(tree) == 1:
                minima_list.append(list(tree.values())[0])
            else:
                minima_list.append(np.mean(list(tree.values())))
        return np.var(minima_list)

    @staticmethod
    def get_persistence(msc):
        """Returns the persistence of the tree as a list."""
        return [msc.get_merge_sequence()[key][0] for key in msc.get_merge_sequence()]

    @staticmethod
    def get_persistence_dict(msc):
        """Returns the persistence of the tree as a dictionary."""
        return {
            key: msc.get_merge_sequence()[key][0] for key in msc.get_merge_sequence()
        }

    @staticmethod
    def bottleneck_distance(persistence1, persistence2):
        """Calculates the bottleneck distance between two persistence diagrams."""
        ppd1 = [(0, val) for val in persistence1]
        ppd2 = [(0, val) for val in persistence2]
        return gd.bottleneck_distance(ppd1, ppd2)

    @staticmethod
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

    @staticmethod
    def plot_persistence_barcode(msc):
        """Plots the persistence barcode for a Morse-Smale complex."""
        node_list = [
            str(node) for node in list(TDAUtils.get_persistence_dict(msc).keys())
        ]
        persistence_list = list(TDAUtils.get_persistence_dict(msc).values())
        barh = plt.barh(node_list, persistence_list)
        plt.xlabel("Persistence")
        plt.ylabel("Node")
        plt.title("Node vs Persistence")
        # plt.bar_label(barh)
        plt.show()

    def compute_merge_trees(self, loss_matrices):
        """Computes merge trees and manually constructs the joint tree."""
        trees = {}
        x, y = loss_matrices[0].shape
        coords = [[float(xx), float(yy)] for yy in range(y) for xx in range(x)]

        for i in range(len(loss_matrices)):
            loss_flat = loss_matrices[i].flatten()
            graph = ngl.EmptyRegionGraph(beta=1.0, relaxed=False, p=2.0)

            # Compute Sublevel Set Merge Tree
            sub_tree = tp.MergeTree(graph=graph)
            sub_tree.build(np.array(coords), loss_flat)

            # Compute Superlevel Set Merge Tree (negating function values)
            super_tree = tp.MergeTree(graph=graph)
            super_tree.build(np.array(coords), -loss_flat)

            # Construct the Joint Tree
            joint_tree = JointTree()

            # Add nodes from both trees
            # Get all unique nodes from both trees
            all_nodes = set()
            for node in sub_tree.nodes:
                all_nodes.add(node)
            for node in super_tree.nodes:
                all_nodes.add(node)

            # Add nodes with their coordinates and function values
            for node_id in all_nodes:
                joint_tree.add_node(node_id, coords[node_id], loss_flat[node_id])

            # Add edges from both trees
            for edge in sub_tree.edges:
                joint_tree.add_sub_edge(edge[0], edge[1])

            for edge in super_tree.edges:
                joint_tree.add_super_edge(edge[0], edge[1])

            trees[i] = joint_tree.get_tree()  # Store as a dictionary

        return trees

    def compute_ms_complexes(self, loss_matrices):
        """Computes Morse-Smale complexes for a list of loss matrices."""
        ms_complexes = {}
        x, y = loss_matrices[0].shape
        coords = [[float(xx), float(yy)] for yy in range(y) for xx in range(x)]

        graph = ngl.EmptyRegionGraph(beta=1.0, relaxed=False, p=2.0)
        for i in range(len(loss_matrices)):
            ms_complex = tp.MorseSmaleComplex(
                graph=graph, gradient="steepest", normalization="feature"
            )
            ms_complex.build(np.array(coords), loss_matrices[i].flatten())
            ms_complexes[i] = ms_complex

        return ms_complexes

    def compute_bottleneck_distance(self, loss_matrices, ms_complexes, trees):
        """Computes the bottleneck distance between two loss matrices."""
        persistence = {}
        # for i in range(10):
        for i in range(len(loss_matrices)):
            persistence[i] = self.get_persistence(ms_complexes[i])
        print("persistence: ", persistence)

        # compute the variance between the minima of the top 2 eigenvectors with all other eigenvectors
        variances = {}
        # for i in range(10):
        for i in range(0, len(loss_matrices)):
            variances[i] = self.minima_variance(
                [self.minima_and_nodes(trees[0]), self.minima_and_nodes(trees[i])]
            )
        print("variances: ", variances)

        # compute the bottleneck distance between the persistence diagrams of the top 2 eigenvectors with all other eigenvectors
        bottlenecks = {}

        for i in range(len(loss_matrices)):
            bottlenecks[i] = self.bottleneck_distance(persistence[0], persistence[i])
        print("bottlenecks: ", bottlenecks)
        df = pd.DataFrame({"variances": variances, "bottlenecks": bottlenecks})

        return df
