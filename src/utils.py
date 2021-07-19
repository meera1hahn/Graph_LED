import torch
import numpy as np
import json
import networkx as nx
import numpy as np
import math
import torch.nn as nn


def evaluate(args, splitFile, run_name):
    split_name = splitFile.split("_")[0]
    distance_scores = []
    splitData = json.load(open(args.data_dir + splitFile))
    fileName = args.predictions_dir + run_name + "_" + split_name + "_submission.json"
    submission = json.load(open(fileName))
    for gt in splitData:
        gt_graph = args.scan_graphs[gt["scanName"]]
        gt_vp = gt["finalLocation"]["viewPoint"]
        pred_vp = submission[gt["episodeId"]]["viewpoint"]
        distance_scores.append(get_geo_dist(gt_graph, gt_vp, pred_vp))

    distance_scores = np.asarray(distance_scores)
    print(
        f"Result {split_name} -- \n LE: {np.mean(distance_scores):.4f}",
        f"Acc@0m: {sum(distance_scores <= 0) * 1.0 / len(distance_scores):.4f}",
        f"Acc@3m: {sum(distance_scores <= 3) * 1.0 / len(distance_scores):.4f}",
        f"Acc@5m: {sum(distance_scores <= 5) * 1.0 / len(distance_scores):.4f}",
        f"Acc@10m: {sum(distance_scores <= 10) * 1.0 / len(distance_scores):.4f}",
    )


def accuracy(dists, threshold=3):
    """Calculating accuracy at 3 meters by default"""
    return np.mean((torch.tensor(dists) <= threshold).int().numpy())


def accuracy_batch(dists, threshold):
    return (dists <= threshold).int().numpy().tolist()


def distance(pose1, pose2):
    """Euclidean distance between two graph poses"""
    return (
        (pose1["pose"][3] - pose2["pose"][3]) ** 2
        + (pose1["pose"][7] - pose2["pose"][7]) ** 2
        + (pose1["pose"][11] - pose2["pose"][11]) ** 2
    ) ** 0.5


def open_graph(connectDir, scan_id):
    """Build a graph from a connectivity json file"""
    infile = "%s%s_connectivity.json" % (connectDir, scan_id)
    G = nx.Graph()
    with open(infile) as f:
        data = json.load(f)
        for i, item in enumerate(data):
            if item["included"]:
                for j, conn in enumerate(item["unobstructed"]):
                    if conn and data[j]["included"]:
                        assert data[j]["unobstructed"][i], "Graph should be undirected"
                        G.add_edge(
                            item["image_id"],
                            data[j]["image_id"],
                            weight=distance(item, data[j]),
                        )
    return G


def get_geo_dist(D, n1, n2):
    return nx.dijkstra_path_length(D, n1, n2)


def snap_to_grid(G, node2pix, sn, pred_coord, conversion, level, true_viewpoint=None):
    min_dist = math.inf
    best_nodes = []
    best_node = ""
    for node in node2pix[sn].keys():
        if node2pix[sn][node][2] != int(level) or node not in G:
            continue
        target_coord = [node2pix[sn][node][0][1], node2pix[sn][node][0][0]]
        dist = np.sqrt(
            (target_coord[0] - pred_coord[0]) ** 2
            + (target_coord[1] - pred_coord[1]) ** 2
        ) / (conversion)
        if dist.item() < min_dist:
            best_node = node
            min_dist = dist.item()
        if dist < 1:
            best_nodes.append(node)

    if true_viewpoint != None:
        min_dist = math.inf
        for b in best_nodes:
            dist = get_geo_dist(G, b, true_viewpoint)
            if dist < min_dist:
                best_node = b
                min_dist = dist
    return best_node


def distance_from_pixels(node2pix, scan_graphs, preds, mesh_conversions, info_elem):
    """Calculate distances between model predictions and targets within a batch.
    Takes the propablity map over the pixels and returns the geodesic distance"""
    # calculate location error and accuracy
    distances = []
    _, _, scan_names, _, true_viewpoints = info_elem
    b, max_floors, h, w = preds.size()
    for pred, conversion, sn, tv in zip(
        preds,
        mesh_conversions,
        scan_names,
        true_viewpoints,
    ):
        total_floors = len(set([v[2] for k, v in node2pix[sn].items()]))
        pred = nn.functional.interpolate(
            pred.unsqueeze(1), (700, 1200), mode="bilinear"
        ).squeeze(1)[:total_floors]
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        G = scan_graphs[sn]
        convers = conversion.view(max_floors, 1, 1)[pred_coord[0].item()]
        pred_viewpoint = snap_to_grid(
            G,
            node2pix,
            sn,
            [pred_coord[1].item(), pred_coord[2].item()],
            convers,
            pred_coord[0].item(),
            tv,
        )
        distances.append(get_geo_dist(G, pred_viewpoint, tv))
    return distances
