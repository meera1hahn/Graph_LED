import json
import random
import numpy as np
from src.utils import get_geo_dist
from src.cfg import *


def random_node_selection(args, split):
    splitData = json.load(open(args.data_dir + split))
    submission = {}
    for episode in splitData:
        nodes = [n for n in args.scan_graphs[episode["scanName"]].nodes()]
        vp = random.choice(nodes)
        submission[episode["episodeId"]] = {"viewpoint": vp}
    fileName = (
        args.predictions_dir
        + "randomBaseline_"
        + split.split("_")[0]
        + "_submission.json"
    )
    json.dump(
        submission,
        open(fileName, "w"),
        indent=3,
    )


def evaluate(args, split):
    split_name = split.split("_")[0]
    distance_scores = []
    splitData = json.load(open(args.data_dir + split))
    fileName = (
        args.predictions_dir + "randomBaseline_" + split_name + "_submission.json"
    )
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


if __name__ == "__main__":
    args = parse_args()

    data_splits = [
        "train_data.json",
        "valSeen_data.json",
        "valUnseen_data.json",
        "test_data_full.json",
    ]

    for split in data_splits:
        random_node_selection(
            args,
            split,
        )
        evaluate(args, split)
