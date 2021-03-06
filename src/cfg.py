import argparse
from src.utils import open_graph

parser = argparse.ArgumentParser(description="LED task")

# What are you doing
parser.add_argument("--attention", default=False, action="store_true")
parser.add_argument("--train", default=False, action="store_true")
parser.add_argument("--evaluate", default=False, action="store_true")
parser.add_argument("--model_save", default=True, action="store_true")

# Input Paths
parser.add_argument("--data_dir", type=str, default="../../data/way_splits/")
parser.add_argument("--image_dir", type=str, default="../../data/floorplans/")
parser.add_argument("--embedding_dir", type=str, default="../../data/word_embeddings/")
parser.add_argument("--connect_dir", type=str, default="../../data/connectivity/")
parser.add_argument("--panofeat_dir", type=str, default="../../data/node_feats/")
parser.add_argument(
    "--geodistance_file", type=str, default="../../data/geodistance_nodes.json"
)

# Output Paths
parser.add_argument(
    "--summary_dir",
    type=str,
    default="../../log_dir/tensorboard/",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="../../log_dir/checkpoints/",
)
parser.add_argument(
    "--predictions_dir",
    type=str,
    default="../../log_dir/predictions",
    help="location of generated predictions to evaluate/visualize",
)
parser.add_argument(
    "--eval_ckpt",
    type=str,
    default="ckpt.pt",
    help="a checkpoint to evaluate by either testing or generate_predictions",
)


# Logging
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--log", default=True, action="store_true", help="log losses")
parser.add_argument("--summary", default=True, action="store_true", help="tensorboard")
parser.add_argument("--name", type=str, default="no_name", help="name of the run")

# Params
parser.add_argument("--cuda", type=str, default=0, help="which GPU to use")
parser.add_argument("--max_floors", type=int, default=5)
parser.add_argument("--max_nodes", type=int, default=340)
parser.add_argument("--max_nodes_test", type=int, default=230)
parser.add_argument("--pano_embed_size", type=int, default=2048)
parser.add_argument("--rnn_embed_size", type=int, default=300)
parser.add_argument("--rnn_hidden_size", type=int, default=1024)
parser.add_argument("--bidirectional", default=True, action="store_true")
parser.add_argument("--loss_type", type=str, default="kl")
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--grad_clip", type=float, default=0.5, help="gradient clipping")
parser.add_argument("--num_epoch", type=int, default=40, help="upper epoch limit")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--early_stopping", type=int, default=10)


def collect_graphs(args):  # get scene graphs
    scan_graphs = {}
    scans = [s.strip() for s in open(args.connect_dir + "scans.txt").readlines()]
    for scan_id in scans:
        scan_graphs[scan_id] = open_graph(args.connect_dir, scan_id)
    return scan_graphs


def parse_args():
    args = parser.parse_args()
    args.run_name = args.name
    args.scan_graphs = collect_graphs(args)
    return args
