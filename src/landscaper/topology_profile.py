from collections import deque
from landscaper.tda import topological_index, digraph_mt

import networkx as nx
import itertools


class TreeNode:
    def __init__(self, id, loss, parent=None):
        self.node_id = id
        self.loss = loss
        self.children = {}
        self.parent = parent
        self.off_set_points_left = []
        self.off_set_points_right = []


def dfs(root):
    if not root:
        return

    for ck in root.children:
        child = root.children[ck]
        child.parent = root
        dfs(child)


def construct_nodes(mt, msc):
    nodes = {}
    for nID, val in mt.nodes.items():
        nodes[nID] = [val, topological_index(msc, nID)]
    return nodes, mt.root


def construct_tree(mt, nodes):
    # node_id -> TreeNode
    node_dict = {}
    # target_node_id -> segmentation_id
    edge_dict = {}

    for e in list(mt.edges):
        target, source = e
        edge_dict[target] = e

        # Create source node if it doesn't exist
        if source not in node_dict:
            node_dict[source] = TreeNode(source, nodes[source][0])

        # Create target node if it doesn't exist
        if target not in node_dict:
            node_dict[target] = TreeNode(target, nodes[target][0], node_dict[source])
        else:
            # Update parent if node exists
            node_dict[target].parent = node_dict[source]

        node_dict[source].children[target] = node_dict[target]

    dfs(node_dict[mt.root])

    return node_dict, edge_dict


def build_basin(node: TreeNode, g, mt, edge_dict):
    node.child_width = 0
    for child in node.children.values():
        node.child_width += build_basin(child, g, mt, edge_dict)

    off_set_points_left = deque([])
    off_set_points_right = deque([])

    curr_node = node.node_id
    reachable = list(nx.bfs_edges(g, curr_node))
    parts = nx.get_edge_attributes(g, "partitions")
    vals = list(itertools.chain.from_iterable([parts[e] for e in reachable]))
    # all nodes underneath this one
    vals.append(curr_node)
    segmentations = [mt.Y[v] for v in vals]
    segmentations.sort()

    # point with smallest loss gets placed at center, moving ascending
    off_set_points_right.append({"x": 0, "y": node.loss})
    off_set_points_left.appendleft({"x": 0, "y": node.loss})
    for i, s in enumerate(segmentations):
        off_set_points_right.append({"x": i, "y": s})
        off_set_points_left.appendleft(
            {
                "x": -i,
                "y": s,
            }
        )

    i += 1
    if node.parent:
        off_set_points_right.append(
            {
                "x": i,
                "y": node.parent.loss,
            }
        )
        off_set_points_left.appendleft(
            {
                "x": -i,
                "y": node.parent.loss,
            }
        )

    node.off_set_points_left = off_set_points_left
    node.off_set_points_right = off_set_points_right
    node.total_width = len(segmentations)

    return node.total_width


def assign_center(node: TreeNode, start: int, end: int):
    if not node:
        return

    node.center = (start + end) / 2
    if len(node.children.values()) == 0:
        return

    left = start + (end - start) / 2 - node.child_width / 2
    childrens = node.children.values()
    childrens = sorted(childrens, key=lambda item: item.total_width, reverse=True)
    for child in childrens:
        proportion = child.total_width / node.child_width
        partial_length = node.child_width * proportion
        sub_start = left
        sub_end = left + partial_length
        assign_center(child, sub_start, sub_end)
        left += partial_length


def generate_profile(mt, msc):
    nodes, root_id = construct_nodes(mt, msc)
    node_dict, edge_dict = construct_tree(mt, nodes)

    root = node_dict[root_id]

    g = digraph_mt(mt)
    build_basin(root, g, mt, edge_dict)
    assign_center(root, 0, root.total_width)

    # Initialize result arrays
    res = []

    def collect_individual_basins(node: TreeNode):
        for child in node.children.values():
            collect_individual_basins(child)

        left = [[ori["x"] + node.center, ori["y"]] for ori in node.off_set_points_left]
        right = [
            [ori["x"] + node.center, ori["y"]] for ori in node.off_set_points_right
        ]

        pts = left + right
        if len(pts) > 0:
            res.append(
                {
                    "area": pts,
                }
            )

    collect_individual_basins(root)

    return res
