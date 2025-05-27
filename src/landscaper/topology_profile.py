from landscaper.tda import digraph_mt

import networkx as nx
import itertools


def build_basin(node_id, g, mt):
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


def get_parent(g, n):
    pred = list(g.predecessors(n))
    if len(pred) == 0:
        return None
    elif len(pred) > 1:
        raise ValueError("Graph has nodes with more than 1 parent!")
    else:
        return pred[0]


def assign_center(node_id, g, start: int, end: int):
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


def generate_profile(mt, msc):
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
        left = [
            [-1 * (len(node["points"]) - i) + node["center"], y]
            for i, y in enumerate(node["points"])
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
