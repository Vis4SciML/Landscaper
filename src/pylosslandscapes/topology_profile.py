import pandas as pd
from collections import deque, defaultdict
from .utils import validate_dataframe


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


def construct_nodes(df):
    validate_dataframe(df, ["NodeId", "Scalar", "CriticalType"], "mergedf")

    nodes = {}
    root_id = None

    for row in df.itertuples():
        n_id = int(row.NodeId)
        s = row.Scalar
        ct = int(row.CriticalType)
        nodes[n_id] = [s, ct]
        if ct > 1:
            root_id = n_id

    if root_id is None:
        raise ValueError("No root node found!")

    return nodes, root_id


def group_segmentations(df):
    """segmentation_id -> list of loss values

    Args:
        df: Segmentation dataframe, [Loss: np.float,
                                     SegmentationId: np.int16]

    Returns:
        Dict{int,list[list[loss, idx]]}
    """
    validate_dataframe(df, ["Loss", "SegmentationId"], "segdf")
    grouped_segmentation = defaultdict(list)

    # Process segmentations first
    for i, s in enumerate(df.itertuples()):
        loss = s.Loss
        segId = int(s.SegmentationId)
        grouped_segmentation[segId].append([loss, i])

    return grouped_segmentation


def construct_tree(df, nodes, gs, root_id):
    validate_dataframe(df, ["upNodeId", "downNodeId", "SegmentationId"], "edgedf")

    # node_id -> TreeNode
    node_dict = {}
    # target_node_id -> segmentation_id
    edge_dict = {}

    missing_segmentations = []
    missing_data = []
    self_loops = []
    for e in df.itertuples():
        source = e.upNodeId
        target = e.downNodeId
        segId = e.SegmentationId

        if segId not in gs:
            missing_segmentations.append((target, segId))

        if source not in nodes or target not in nodes:
            missing_data.append((source, target, segId))

        if source == target:
            self_loops.append((source, target))

        edge_dict[target] = segId

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

    if missing_segmentations:
        for target_id, seg_id in missing_segmentations:
            print(f"Node {target_id} has segmentation ID {seg_id} but no data found.")
        raise ValueError("Missing segmentations.")

    if missing_data:
        for s, t, seg in missing_data:
            print(f"Missing data: {s}, {t}, {seg}")
        raise ValueError("Missing data.")

    if self_loops:
        for s, t in self_loops:
            print(f"Self loop detected: {s}->{t}")
        raise ValueError("Self loop detected.")

    root = node_dict[root_id]

    if not validate_tree(root):
        raise ValueError("Invalid tree structure detected.")

    dfs(root)

    return node_dict, edge_dict, root


def validate_tree(node):
    if not node:
        return True
    for child in node.children.values():
        if child.parent != node:
            print(f"Error: Invalid parent-child relationship for node {child.node_id}")
            return False
        if not validate_tree(child):
            return False
    return True


def build_basin(node: TreeNode, gs, edge_dict):
    if not node:
        return
    acc_number = 0
    for child in node.children.values():
        acc_number += build_basin(child, gs, edge_dict)
    node.child_width = acc_number
    if not node.parent:
        return

    # get a list of segmentations of loss values
    segmentations = gs[edge_dict[node.node_id]]
    segmentations.sort(key=lambda x: x[0])
    off_set_points_left = deque([])
    off_set_points_right = deque([])

    off_set_points_right.append({"x": 0, "y": node.loss, "node_id": node.node_id})
    off_set_points_left.appendleft({"x": 0, "y": node.loss, "node_id": node.node_id})
    for s in segmentations:
        acc_number += 1
        off_set_points_right.append({"x": acc_number / 2, "y": s[0], "node_id": s[1]})
        off_set_points_left.appendleft(
            {"x": -acc_number / 2, "y": s[0], "node_id": s[1]}
        )

    if node.parent:
        off_set_points_right.append(
            {
                "x": acc_number / 2,
                "y": node.parent.loss,
                "node_id": node.parent.node_id,
            }
        )
        off_set_points_left.appendleft(
            {
                "x": -acc_number / 2,
                "y": node.parent.loss,
                "node_id": node.parent.node_id,
            }
        )

    node.off_set_points_left = off_set_points_left
    node.off_set_points_right = off_set_points_right
    node.total_width = acc_number

    return acc_number


def collect_seg_point_ids(node: TreeNode) -> set:
    curr = set()
    for child in node.children.values():
        curr = curr.union(collect_seg_point_ids(child))
    node.acc_seg_point_ids = curr.union(node.acc_seg_point_ids)
    return node.acc_seg_point_ids


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


def generate_profile(segdf, mergedf, edgedf):
    nodes, root_id = construct_nodes(mergedf)
    grouped_segmentation = group_segmentations(segdf)
    node_dict, edge_dict, root = construct_tree(
        edgedf, nodes, grouped_segmentation, root_id
    )

    root = node_dict[root_id]
    root.total_width = len(segdf)

    build_basin(root, grouped_segmentation, edge_dict)
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
                    "isBasin": len(node.children.values()) == 0,
                    "segID": edge_dict.get(node.node_id, -1),
                }
            )

    collect_individual_basins(root)

    return res
