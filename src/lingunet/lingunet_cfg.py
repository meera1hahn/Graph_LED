import argparse
from src.utils import open_graph


parser = argparse.ArgumentParser(description="LED task")

# What are you doing
parser.add_argument("--train", default=False, action="store_true")
parser.add_argument("--evaluate", default=False, action="store_true")
parser.add_argument("--generate_predictions", default=False, action="store_true")
parser.add_argument("--eval_predictions", default=False, action="store_true")
parser.add_argument("--visualize", default=False, action="store_true")

# Data/Input Paths
parser.add_argument("--data_dir", type=str, default="../../data/way_splits/")
parser.add_argument("--image_dir", type=str, default="../../data/floorplans/")
parser.add_argument("--embedding_dir", type=str, default="../../data/word_embeddings/")
parser.add_argument("--connect_dir", type=str, default="../../data/connectivity/")
parser.add_argument(
    "--geodistance_file", type=str, default="../../data/geodistance_nodes.json"
)
# Output Paths
parser.add_argument("--summary_dir", type=str, default="../../log_dir/tensorboard/")
parser.add_argument("--checkpoint_dir", type=str, default="../../log_dir/checkpoints/")
parser.add_argument("--predictions_dir", type=str, default="../../log_dir/predictions")
parser.add_argument("--model_save", default=False, action="store_true")
parser.add_argument(
    "--eval_ckpt",
    type=str,
    default="/path/to/ckpt.pt",
    help="a checkpoint to evaluate by either testing or generate_predictions",
)

# FO Layer before lingunet and scaling for the image
parser.add_argument("--freeze_resnet", default=True, action="store_true")
parser.add_argument("--ds_percent", type=float, default=0.65)
parser.add_argument("--ds_scale", type=float, default=0.125)
parser.add_argument("--ds_height_crop", type=int, default=54)
parser.add_argument("--ds_width_crop", type=int, default=93)
parser.add_argument("--ds_height", type=int, default=57)
parser.add_argument("--ds_width", type=int, default=98)
parser.add_argument("--max_floors", type=int, default=5)
# CNN
parser.add_argument("--num_conv_layers", type=int, default=1)
parser.add_argument("--conv_dropout", type=float, default=0.0)
parser.add_argument("--deconv_dropout", type=float, default=0.0)
parser.add_argument("--res_connect", default=True, action="store_true")
# RNN
parser.add_argument("--encoder", type=str, default="rnn", help="Options: rnn or hrn")
parser.add_argument("--embed_size", type=int, default=300)
parser.add_argument("--rnn_hidden_size", type=int, default=300)
parser.add_argument("--num_rnn_layers", type=int, default=1)
parser.add_argument("--bidirectional", default=True, action="store_true")
parser.add_argument("--embed_dropout", type=float, default=0.5)
parser.add_argument("--num_rnn2conv_layers", type=int, default=1)
# Final linear layers
parser.add_argument("--num_linear_hidden_layers", type=int, default=1)
parser.add_argument("--linear_hidden_size", type=int, default=128)
parser.add_argument("--num_lingunet_layers", type=int, default=3)

# Params
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--summary", default=True, action="store_true", help="tensorboard")
parser.add_argument("--name", type=str, default="no_name", help="name of the run")
parser.add_argument("--cuda", type=str, default=0, help="which GPU to use")
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--grad_clip", type=float, default=0.5, help="gradient clipping")
parser.add_argument("--num_epoch", type=int, default=40, help="upper epoch limit")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--batch_size", type=int, default=8)
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
