import argparse
from utils import open_graph

parser = argparse.ArgumentParser(description="LED task")

# What are you doing
parser.add_argument("--train", default=True, action="store_true")
parser.add_argument("--evaluate", default=False, action="store_true")
parser.add_argument("--generate_predictions", default=False, action="store_true")
parser.add_argument("--eval_predictions", default=False, action="store_true")
parser.add_argument("--visualize", default=False, action="store_true")
parser.add_argument("--test_multi_floor", default=False, action="store_true")
parser.add_argument(
    "--distance_metric",
    type=str,
    default="euclidean",
    help="Options: euclidean or geodesic",
)

# Data/Input Paths
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
parser.add_argument("--summary_dir", type=str, default="/path/to/tensorboard/")
parser.add_argument("--log_dir", type=str, default="/path/to/logs/")
parser.add_argument("--visualization_dir", type=str, default="/path/to/visualizations/")
parser.add_argument("--save", default=False, action="store_true")
parser.add_argument(
    "--eval_ckpt",
    type=str,
    default="/path/to/ckpt.pt",
    help="a checkpoint to evaluate by either testing or generate_predictions",
)
parser.add_argument(
    "--predictions_dir",
    type=str,
    default="./path/to/predictions.json",
    help="location of generated predictions to evaluate/visualize",
)

# Logging
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--log", default=False, action="store_true", help="log losses")
parser.add_argument("--summary", default=True, action="store_true", help="tensorboard")
parser.add_argument("--project_name", type=str, default="led-linguine", help="Comet")
parser.add_argument("--name", type=str, default="no_name", help="name of the run")
parser.add_argument("--model", type=str, default="lingunet", help="model used")

# FO Layer before lingunet and scaling for the image
parser.add_argument("--cuda", type=str, default=0, help="which GPU to use")
parser.add_argument("--freeze_resnet", default=True, action="store_true")
parser.add_argument("--max_floors", type=int, default=5)
parser.add_argument("--max_nodes", type=int, default=345)

# CNN
parser.add_argument("--pano_embed_size", type=int, default=2048)
parser.add_argument("--hidden_embed_size", type=int, default=2048)

# RNN
parser.add_argument("--rnn_embed_size", type=int, default=300)
parser.add_argument("--rnn_hidden_size", type=int, default=2048)
parser.add_argument("--num_rnn_layers", type=int, default=1)
parser.add_argument("--bidirectional", default=True, action="store_true")
parser.add_argument("--embed_dropout", type=float, default=0.5)
parser.add_argument(
    "--embedding_type",
    type=str,
    default="stratch",
    help="Options: stratch, word2vec, glove",
)
parser.add_argument("--num_rnn2conv_layers", type=int, default=1)

# Final linear layers
parser.add_argument("--num_linear_hidden_layers", type=int, default=1)
parser.add_argument("--linear_hidden_size", type=int, default=128)

# Architecture Specific Arguments
parser.add_argument("--num_lingunet_layers", type=int, default=3)
parser.add_argument("--sample_used", type=float, default=1.0)
parser.add_argument("--loss_type", type=str, default="kl")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--grad_clip", type=float, default=0.5, help="gradient clipping")
parser.add_argument("--num_epoch", type=int, default=40, help="upper epoch limit")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--early_stopping", type=int, default=10)
parser.add_argument("--data_aug", default=False, action="store_true")
parser.add_argument("--increase_train", default=False, action="store_true")
parser.add_argument(
    "--avgpool", default=False, action="store_true", help="Average pool predictions"
)
parser.add_argument(
    "--sigma_scalar",
    type=float,
    default=1.0,
    help="Sigma multiplier for the Gaussian target in meters",
)

# Language Ablation Arguments
parser.add_argument("--blind_vis", default=False, action="store_true")
parser.add_argument("--blind_lang", default=False, action="store_true")
parser.add_argument(
    "--language_change",
    type=str,
    default="none",
    help="default is none, options: shuffle,locator_only,observer_only,first_half,second_half",
)


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
    args.scan_graphs = collect_graphs(args)
    return args
