import argparse
from src.utils import open_graph

parser = argparse.ArgumentParser(description="LED task")

# What are you doing
parser.add_argument("--attention", default=False, action="store_true")
parser.add_argument("--train", default=False, action="store_true")
parser.add_argument("--evaluate", default=False, action="store_true")
parser.add_argument("--generate_predictions", default=False, action="store_true")
parser.add_argument("--eval_predictions", default=False, action="store_true")
parser.add_argument("--visualize", default=False, action="store_true")

# Input Paths
parser.add_argument(
    "--data_base_dir",
    type=str,
    default="/srv/share/mhahn30/Projects/LED/data/public_data/",  # "../data/",
    help="path to base data directory",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="way_splits/",
    help="path to data folder where train.json, dev.json, and test.json files",
)
parser.add_argument(
    "--image_dir",
    type=str,
    default="floorplans/",
    help="path to `top down maps`",
)
parser.add_argument("--embedding_dir", type=str, default="word_embeddings/")
parser.add_argument("--connect_dir", type=str, default="connectivity/")
parser.add_argument(
    "--mesh2meters", type=str, default="floorplans/pix2meshDistance.json"
)

# Output Paths
parser.add_argument(
    "--summary_dir",
    type=str,
    default="/srv/share/mhahn30/Projects/Graph_LED/model_runs/tensorboard/",
)  # /path/to/tensorboard/")
parser.add_argument(
    "--log_dir",
    type=str,
    default="/srv/share/mhahn30/Projects/Graph_LED/model_runs/logs/",
)  # /path/to/logs/")
parser.add_argument(
    "--visualization_dir",
    type=str,
    default="/srv/share/mhahn30/Projects/Graph_LED/model_runs/visualizations/",
)  # /path/to/visualizations/")
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="/srv/share/mhahn30/Projects/Graph_LED/model_runs/checkpoints/",
)  # /path/to/checkpoints/")
parser.add_argument("--model_save", default=True, action="store_true")
parser.add_argument(
    "--eval_ckpt",
    type=str,
    default="ckpt.pt",
    help="a checkpoint to evaluate by either testing or generate_predictions",
)
parser.add_argument(
    "--predictions_dir",
    type=str,
    default="/srv/share/mhahn30/Projects/Graph_LED/model_runs/predictions/",
    # default="./path/to/predictions.json",
    help="location of generated predictions to evaluate/visualize",
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
    args.data_dir = args.data_base_dir + args.data_dir
    args.image_dir = args.data_base_dir + args.image_dir
    args.embedding_dir = args.data_base_dir + args.embedding_dir
    args.connect_dir = args.data_base_dir + args.connect_dir
    args.mesh2meters = args.data_base_dir + args.mesh2meters
    args.eval_ckpt = (
        args.checkpoint_dir
        + "test_implementation_changes/Epoch1_Acc1K-0.0086.pt"  # joint_attention_shuffle_panos/Epoch5_Acc1K-0.1048.pt"
    )
    args.scan_graphs = collect_graphs(args)
    return args
