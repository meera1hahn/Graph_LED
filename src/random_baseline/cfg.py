import argparse
from utils import open_graph

parser = argparse.ArgumentParser(description="LED task --random baseline")

# Data/Input Paths
parser.add_argument(
    "--data_base_dir",
    type=str,
    default="/srv/share/mhahn30/Projects/LED/data/public_data/",  # "../data/",
    help="path to base data directory",
)
parser.add_argument("--data_dir", type=str, default="way_splits/")
parser.add_argument("--image_dir", type=str, default="floorplans/")
parser.add_argument("--connect_dir", type=str, default="connectivity/")
# Output Paths
parser.add_argument(
    "--predictions_dir",
    type=str,
    default="/srv/share/mhahn30/Projects/Graph_LED/model_runs/predictions/",
)


def collect_graphs(args):  # get scene graphs
    scan_graphs = {}
    scans = [s.strip() for s in open(args.connect_dir + "scans.txt").readlines()]
    for scan_id in scans:
        scan_graphs[scan_id] = open_graph(args.connect_dir, scan_id)
    return scan_graphs


def parse_args():
    args = parser.parse_args()
    args.data_dir = args.data_base_dir + args.data_dir
    args.image_dir = args.data_base_dir + args.image_dir
    args.connect_dir = args.data_base_dir + args.connect_dir
    args.scan_graphs = collect_graphs(args)
    return args
