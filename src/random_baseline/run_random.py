import json
import random
import numpy as np
from src.utils import evaluate
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


if __name__ == "__main__":
    args = parse_args()

    data_splits = [
        "train_data.json",
        "valSeen_data.json",
        "valUnseen_data.json",
        "test_data_full.json",
    ]

    for splitFile in data_splits:
        random_node_selection(
            args,
            splitFile,
        )
        evaluate(args, splitFile, run_name="randomBaseline")
