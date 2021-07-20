import csv
import sys
import numpy as np
from tqdm import tqdm
import base64
from src.utils import open_graph
import torch

csv.field_size_limit(sys.maxsize)

_TSV_FIELDNAMES = ["scanId", "viewpointId", "image_w", "image_h", "vfov", "features"]


def _convert_item(item):
    item["image_w"] = int(item["image_w"])  # pixels
    item["image_h"] = int(item["image_h"])  # pixels
    item["vfov"] = int(item["vfov"])  # degrees
    item["features"] = np.frombuffer(
        base64.b64decode(item["features"]), dtype=np.float32
    ).reshape(
        (-1, 2048)
    )  # 36 x 2048 region features
    return item


def load_feats():
    keys = []
    pano_feats = {}
    path = "data/ResNet-152-places365.tsv"
    with open(path, "rt") as fid:
        reader = csv.DictReader(fid, delimiter="\t", fieldnames=_TSV_FIELDNAMES)
        for item in tqdm(reader):
            item = _convert_item(item)
            key = item["scanId"] + "-" + item["viewpointId"]
            keys.append(key.encode())
            pano_feats[key.encode()] = item["features"]

    scanfile = "data/connectivity/scans.txt"
    scans = [d.strip() for d in open(scanfile).readlines()]
    max_nodes = 230

    for s in scans:
        if s == "B6ByNegPMKs":
            continue
        nodes = sorted(
            [
                n
                for n in open_graph(
                    "data/connectivity/",
                    s,
                ).nodes()
            ]
        )
        node_feats = torch.zeros(max_nodes, 36, 2048)
        for i, n in enumerate(nodes):
            key = s + "-" + n
            node_feats[i, :, :] = torch.tensor(pano_feats[key.encode()])
        torch.save(node_feats, "data/node_feats/" + s + ".pt")
        break


load_feats()
