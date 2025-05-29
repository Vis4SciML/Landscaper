import itertools

import networkx as nx
import topopy as tp

from landscaper.tda import digraph_mt


def build_basin(node_id: int, g: nx.DiGraph, mt: tp.MergeTree) -> int:
    """Recursively calculates the points needed for each segmentation in the merge tree.

    Args:
        node_id (int): Node ID to calculate a basin for.
        g (nx.DiGraph): Directed graph representation of the merge tree.
        mt (tp.MergeTree): The merge tree.

    Returns:
        Total width of a given basin.
    """
    child_width = 0
    for child_id in g.successors(node_id):
        child_width += build_basin(child_id, g, mt)

    reachable = list(nx.bfs_edges(g, node_id))
    parts = nx.get_edge_attributes(g, "partitions")

    vals = list(itertools.chain.from_iterable([parts[e] for e in reachable]))
    segmentations = [mt.Y[v] for v in vals]
    segmentations.sort()

    parent = get_parent(g, node_id)
    if parent is not None:
        segmentations.append(g.nodes[parent]["value"])

    nx.set_node_attributes(
        g,
        {
            node_id: {
                "points": segmentations,
                "total_width": len(segmentations),
                "child_width": child_width,
            }
        },
    )
    return len(segmentations)


def get_parent(g: nx.DiGraph, n: int):
    """
    Gets the parent of a node in the directed graph. Errors if there is more than one parent (no longer a tree).
    Args:
        g (nx.DiGraph): Directed graph representation of a merge tree.
        n (int): Node ID to get parent for.

    Returns:
        The node id of the parent.

    Raises:
        ValueError: Thrown when there is more than one parent for a node.
    """
    pred = list(g.predecessors(n))
    if len(pred) == 0:
        return None
    elif len(pred) > 1:
        raise ValueError("Graph has nodes with more than 1 parent!")
    else:
        return pred[0]


def assign_center(node_id, g, start: int, end: int):
    """Assigns the center of the basin when constructing the profile. Needed for visualization.

    Args:
        node_id (int): Node ID.
        g (nx.DiGraph): Directed graph representation of a merge tree.
        start (int): Leftmost point.
        end (int): Rightmost point.
    """
    node = g.nodes[node_id]
    center = (start + end) / 2
    node["center"] = center

    children = list(g.successors(node_id))
    if len(children) == 0:
        return

    cw = node["child_width"]
    left = start + (end - start) / 2 - cw / 2
    children = sorted(children, key=lambda x: g.nodes[x]["total_width"], reverse=True)
    for child in children:
        proportion = g.nodes[child]["total_width"] / cw
        partial_length = cw * proportion
        assign_center(child, g, left, left + partial_length)
        left += partial_length


def generate_profile(mt: tp.MergeTree):
    """Generates a topological profile based on a merge tree. Used along with :obj:`landscaper.plots.topoplogy_profile`. Can be visualized directly with a LossLandscape object using :obj:`landscaper.landscape.LossLandscape.show_profile`.

    Args:
        mt (tp.MergeTree):

    Returns:
        Profile data to visualize.
    """
    root = mt.root

    g = digraph_mt(mt)
    build_basin(root, g, mt)
    assign_center(root, g, 0, g.nodes[root]["total_width"])

    # Initialize result arrays
    res = []

    def collect_individual_basins(node_id):
        node = g.nodes[node_id]
        for child in g.successors(node_id):
            collect_individual_basins(child)

        right = [[i + node["center"], y] for i, y in enumerate(node["points"])]

        node["points"].reverse()
        left = [[-1 * (len(node["points"]) - i) + node["center"], y] for i, y in enumerate(node["points"])]

        pts = left + right
        if len(pts) > 0:
            res.append(
                pts,
            )

    collect_individual_basins(root)

    return res
