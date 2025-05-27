from landscaper.tda import topological_index, digraph_mt

import networkx as nx
import itertools


class TreeNode:
    def __init__(self, id, loss, parent=None):
        self.node_id = id
        self.loss = loss
        self.children = {}
        self.parent = parent
        self.points = []


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

    curr_node = node.node_id
    reachable = list(nx.bfs_edges(g, curr_node))
    parts = nx.get_edge_attributes(g, "partitions")

    vals = list(itertools.chain.from_iterable([parts[e] for e in reachable]))
    segmentations = [mt.Y[v] for v in vals]
    segmentations.sort()

    if node.parent:
        segmentations.append(node.parent.loss)

    node.points = segmentations
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

        right = [[i + node.center, y] for i, y in enumerate(node.points)]

        node.points.reverse()
        left = [
            [-1 * (len(node.points) - i) + node.center, y]
            for i, y in enumerate(node.points)
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
