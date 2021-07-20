import argparse
import json
import os
import shutil
from urllib.request import urlopen

import networkx as nx
import numpy as np
from tqdm import tqdm
import gdown
import zipfile

WAY_LINKS = [
    (
        "data/way_splits/way_splits.zip",
        "https://drive.google.com/uc?id=19env7HjYpgimenS8CJA_1iqoCi_kVDux",
    )
]

WORD_EMBEDDING_LINKS = [
    (
        "data/word_embeddings.zip",
        "https://drive.google.com/uc?id=1gC6Y4jqFOFkKFLSiqkt_ZGU4MM0vYIW7",
    )
]

FLOORPLAN_LINKS = [
    (
        "data/floorplans.zip",
        "https://drive.google.com/uc?id=1_JHaTxty1cnZHnBKUWcNIgAPyCFx0nR7",
    )
]

MODEL_LINKS = [
    (
        "data/models/crossmodal_att.pt",
        "https://drive.google.com/uc?id=1qB-r1sybtJNH3siIoGQ4J3CToyc4BRTK",
    ),
    (
        "data/models/crossmodal_simple.pt",
        "https://drive.google.com/uc?id=1kvUofiaMCz6g6f1BWfvSnO32aU278HUi",
    ),
]

CONNECTIVITY_ROOT_URL = "https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/master/connectivity"
CONNECTIVITY_FILES = [
    "17DRP5sb8fy_connectivity.json",
    "1LXtFkjw3qL_connectivity.json",
    "1pXnuDYAj8r_connectivity.json",
    "29hnd4uzFmX_connectivity.json",
    "2azQ1b91cZZ_connectivity.json",
    "2n8kARJN3HM_connectivity.json",
    "2t7WUuJeko7_connectivity.json",
    "5LpN3gDmAk7_connectivity.json",
    "5q7pvUzZiYa_connectivity.json",
    "5ZKStnWn8Zo_connectivity.json",
    "759xd9YjKW5_connectivity.json",
    "7y3sRwLe3Va_connectivity.json",
    "8194nk5LbLH_connectivity.json",
    "82sE5b5pLXE_connectivity.json",
    "8WUmhLawc2A_connectivity.json",
    "aayBHfsNo7d_connectivity.json",
    "ac26ZMwG7aT_connectivity.json",
    "ARNzJeq3xxb_connectivity.json",
    "B6ByNegPMKs_connectivity.json",
    "b8cTxDM8gDG_connectivity.json",
    "cV4RVeZvu5T_connectivity.json",
    "D7G3Y4RVNrH_connectivity.json",
    "D7N2EKCX4Sj_connectivity.json",
    "dhjEzFoUFzH_connectivity.json",
    "E9uDoFAP3SH_connectivity.json",
    "e9zR4mvMWw7_connectivity.json",
    "EDJbREhghzL_connectivity.json",
    "EU6Fwq7SyZv_connectivity.json",
    "fzynW3qQPVF_connectivity.json",
    "GdvgFV5R1Z5_connectivity.json",
    "gTV8FGcVJC9_connectivity.json",
    "gxdoqLR6rwA_connectivity.json",
    "gYvKGZ5eRqb_connectivity.json",
    "gZ6f7yhEvPG_connectivity.json",
    "HxpKQynjfin_connectivity.json",
    "i5noydFURQK_connectivity.json",
    "JeFG25nYj2p_connectivity.json",
    "JF19kD82Mey_connectivity.json",
    "jh4fc5c5qoQ_connectivity.json",
    "JmbYfDe2QKZ_connectivity.json",
    "jtcxE69GiFV_connectivity.json",
    "kEZ7cmS4wCh_connectivity.json",
    "mJXqzFtmKg4_connectivity.json",
    "oLBMNvg9in8_connectivity.json",
    "p5wJjkQkbXX_connectivity.json",
    "pa4otMbVnkk_connectivity.json",
    "pLe4wQe7qrG_connectivity.json",
    "Pm6F8kyY3z2_connectivity.json",
    "pRbA3pwrgk9_connectivity.json",
    "PuKPg4mmafe_connectivity.json",
    "PX4nDJXEHrG_connectivity.json",
    "q9vSo1VnCiC_connectivity.json",
    "qoiz87JEwZ2_connectivity.json",
    "QUCTc6BB5sX_connectivity.json",
    "r1Q1Z4BcV1o_connectivity.json",
    "r47D5H71a5s_connectivity.json",
    "rPc6DW4iMge_connectivity.json",
    "RPmz2sHmrrY_connectivity.json",
    "rqfALeAoiTq_connectivity.json",
    "s8pcmisQ38h_connectivity.json",
    "S9hNv5qa7GM_connectivity.json",
    "sKLMLpTHeUy_connectivity.json",
    "SN83YJsR3w2_connectivity.json",
    "sT4fr6TAbpF_connectivity.json",
    "TbHJrupSAjP_connectivity.json",
    "ULsKaCPVFJR_connectivity.json",
    "uNb9QFRL6hY_connectivity.json",
    "ur6pFq6Qu1A_connectivity.json",
    "UwV83HsGsw3_connectivity.json",
    "Uxmj2M2itWa_connectivity.json",
    "V2XKFyX4ASd_connectivity.json",
    "VFuaQ6m2Qom_connectivity.json",
    "VLzqgDo317F_connectivity.json",
    "Vt2qJdWjCF2_connectivity.json",
    "VVfe2KiqLaN_connectivity.json",
    "Vvot9Ly1tCj_connectivity.json",
    "vyrNrziPKCB_connectivity.json",
    "VzqfbhrpDEA_connectivity.json",
    "wc2JMjhGNzB_connectivity.json",
    "WYY7iVyf5p8_connectivity.json",
    "X7HyMhZNoso_connectivity.json",
    "x8F5xyUWy9e_connectivity.json",
    "XcA2TqTSSAj_connectivity.json",
    "YFuZgdQ5vWj_connectivity.json",
    "YmJkqBEsHnH_connectivity.json",
    "yqstnuAEVhm_connectivity.json",
    "YVUC4YcDtcY_connectivity.json",
    "Z6MFQCViBuw_connectivity.json",
    "ZMojNkEp431_connectivity.json",
    "zsNo4HB9uLZ_connectivity.json",
    "README.md",
    "scans.txt",
]


def _download_url_to_file(url, path):
    print(f"downloading {url}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urlopen(url) as response, open(path, "wb") as file:
        shutil.copyfileobj(response, file)
    print(f"downloading {url}... done!")


def _download_drive_url_to_file(url, path, _iszipfile=False):
    print(f"downloading {url}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    output = path
    gdown.download(url, output, quiet=False)
    print(f"downloading {url}... done!")
    if _iszipfile:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path[: path.rindex("/") + 1])
        os.remove(path)
        print(f"unzip {path}... done!")


def _load_nav_graph(scan):
    """Load connectivity graph for scan"""

    def distance(pose1, pose2):
        """Euclidean distance between two graph poses"""
        return (
            (pose1["pose"][3] - pose2["pose"][3]) ** 2
            + (pose1["pose"][7] - pose2["pose"][7]) ** 2
            + (pose1["pose"][11] - pose2["pose"][11]) ** 2
        ) ** 0.5

    with open(f"data/connectivity/{scan}_connectivity.json") as f:
        G = nx.Graph()
        positions = {}
        data = json.load(f)
        for i, item in enumerate(data):
            if item["included"]:
                for j, conn in enumerate(item["unobstructed"]):
                    if conn and data[j]["included"]:
                        positions[item["image_id"]] = np.array(
                            [item["pose"][3], item["pose"][7], item["pose"][11]]
                        )
                        assert data[j]["unobstructed"][i], "Graph should be undirected"
                        G.add_edge(
                            item["image_id"],
                            data[j]["image_id"],
                            weight=distance(item, data[j]),
                        )
        nx.set_node_attributes(G, values=positions, name="position")
    return G


def _generate_distances(scan):
    g = _load_nav_graph(scan)
    d = dict(nx.all_pairs_dijkstra_path_length(g))
    with open(f"data/distances/{scan}_distances.json", "w") as fid:
        json.dump(d, fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--way_splits",
        action="store_true",
        help="only download way_splits for led task",
    )
    parser.add_argument(
        "--word_embeddings", action="store_true", help="only download word_embeddings"
    )
    parser.add_argument(
        "--floorplans", action="store_true", help="only download floorplans data"
    )
    parser.add_argument(
        "--connectivity", action="store_true", help="only download connectivity data"
    )
    parser.add_argument("--models", action="store_true", help="only download models")
    args = parser.parse_args()

    download_all = (
        not args.way_splits
        and not args.word_embeddings
        and not args.floorplans
        and not args.connectivity
        and not args.models
    )

    if download_all or args.way_splits:
        for path, url in WAY_LINKS:
            _download_drive_url_to_file(url, path, _iszipfile=True)

    if download_all or args.word_embeddings:
        for path, url in WORD_EMBEDDING_LINKS:
            _download_drive_url_to_file(url, path, _iszipfile=True)

    if download_all or args.floorplans:
        for path, url in FLOORPLAN_LINKS:
            _download_drive_url_to_file(url, path, _iszipfile=True)

    if download_all or args.models:
        for path, url in MODEL_LINKS:
            _download_drive_url_to_file(url, path)

    if download_all or args.connectivity:
        for fname in CONNECTIVITY_FILES:
            path = f"data/connectivity/{fname}"
            url = f"{CONNECTIVITY_ROOT_URL}/{fname}"
            _download_url_to_file(url, path)
        print("generating distance data...")
        os.makedirs("data/distances", exist_ok=True)
        scans = open("data/connectivity/scans.txt", "r").read().splitlines()
        for scan in tqdm(scans):
            _generate_distances(scan)
        print("generating distance data... done!")

    # complete the directory structure
    os.makedirs("data/node_feats/", exist_ok=True)
    os.makedirs("log_dir/", exist_ok=True)
    os.makedirs("log_dir/predictions", exist_ok=True)
    os.makedirs("log_dir/tensorboard", exist_ok=True)
    os.makedirs("log_dir/checkpoints", exist_ok=True)
