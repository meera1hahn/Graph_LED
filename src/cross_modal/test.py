import csv
import sys
import numpy as np
from tqdm import tqdm
import base64
from utils import open_graph
import json
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
    path = "/srv/flash1/mhahn30/LED/ResNet-152-places365.tsv"
    with open(path, "rt") as fid:
        reader = csv.DictReader(fid, delimiter="\t", fieldnames=_TSV_FIELDNAMES)
        for item in tqdm(reader):
            item = _convert_item(item)
            key = item["scanId"] + "-" + item["viewpointId"]
            keys.append(key.encode())
            pano_feats[key.encode()] = item["features"]

    geodistance_nodes = json.load(open("geodistance_nodes.json"))

    split_dir = "/nethome/mhahn30/Repositories/fair_internship/src/data_splits/mp3d/"
    x = ["scenes_valUnseen.txt", "scenes_test.txt"]

    for split in x:
        scans = [d.strip() for d in open(split_dir + split).readlines()]
        max_nodes = 230

        for s in scans:
            if s == "B6ByNegPMKs":
                continue
            nodes = sorted(
                [
                    n
                    for n in open_graph(
                        "/srv/share/mhahn30/Projects/LED/data/public_data/connectivity/",
                        s,
                    ).nodes()
                ]
            )
            node_feats = torch.zeros(max_nodes, 36, 2048)
            for i, n in enumerate(nodes):
                key = s + "-" + n
                node_feats[i, :, :] = torch.tensor(pano_feats[key.encode()])
            torch.save(node_feats, "/srv/flash1/mhahn30/LED/node_feats/" + s + ".pt")


data = json.load(
    open(
        "/srv/share/mhahn30/Projects/LED/data/public_data/way_splits/new_trainData.json"
    )
)
new_data = []
for x in data:
    # if len(d["dialogArray"]) % 2 != 0:
    #     d["dialogArray"] = d["dialogArray"][:-1]

    if x["detailedNavPath"][-1][-1][0] != x["finalLocation"]["viewPoint"]:
        print(x["dialogArray"])
        print(x["detailedNavPath"])
        print(x["navPath"])
        print(x["finalLocation"]["viewPoint"])
        import ipdb

        ipdb.set_trace()
        print(x["detailedNavPath"][-1][-1][0])
        print()

    # print(d["detailedNavPath"])
    # print(len(d["detailedNavPath"]))
    # for enum, i in enumerate(range(2, len(d["dialogArray"]) + 2, 2)):
    #     print(d["dialogArray"][:i])
    #     print(enum)
    #     print(d["detailedNavPath"][enum])

    # print(d["dialogArray"])
    # print()
    # print()
