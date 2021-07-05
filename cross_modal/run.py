import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import tqdm
import numpy as np
import os.path
import os
import cfg
import json

from loader import Loader
from model import XRN


class LEDAgent:
    def __init__(self, args):
        self.args = args
        self.device = (
            torch.device(f"cuda:{args.cuda}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.args.device = self.device
        self.loss_func = nn.KLDivLoss(reduction="batchmean")

        self.loader = None
        self.writer = None
        self.out_dir = os.path.join(args.log_dir, args.run_name)
        if (args.log and args.train) or args.save:
            if not os.path.isdir(self.out_dir):
                print("Log directory under {}".format(self.out_dir))
                os.system("mkdir {}".format(self.out_dir))
            self.writer = SummaryWriter(args.summary_dir + args.name)

        self.model = None
        self.optimizer = None
        self.node2pix = json.load(open(args.image_dir + "allScans_Node2pix.json"))
        self.args.rnn_input_size = len(
            json.load(open(self.args.embedding_dir + "word2idx.json"))
        )

    def load_data(self):
        print("Loading Data...")
        self.loader = Loader(data_dir=self.args.data_dir, args=self.args)
        self.loader.build_dataset(file="train_data.json")
        self.loader.build_dataset(file="valSeen_data.json")
        self.loader.build_dataset(file="valUnseen_data.json")
        self.train_iterator = DataLoader(
            self.loader.datasets["train"],
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        self.valseen_iterator = DataLoader(
            self.loader.datasets["valSeen"],
            batch_size=8,
            shuffle=False,
        )

        self.val_unseen_iterator = DataLoader(
            self.loader.datasets["valUnseen"],
            batch_size=8,
            shuffle=False,
        )
        if self.args.evaluate:
            self.loader.build_dataset(file="test_data.json")
            self.test_iterator = DataLoader(
                self.loader.datasets["test"],
                batch_size=8,
                shuffle=False,
            )

    def scores(self, mode, acc_k1, acc_topk, le):
        print(f"\t{mode} Acc@1k: {np.mean(acc_k1)} Acc@5k: {np.mean(acc_topk)}")
        le = np.asarray(le)
        acc0m = sum(le <= 0) * 1.0 / len(le)
        acc5m = sum(le <= 5) * 1.0 / len(le)
        acc10m = sum(le <= 10) * 1.0 / len(le)
        print(
            f"\t{mode} LE: {np.mean(le)}, Acc@0m: {acc0m}, Acc@5m: {acc5m}, Acc@10m: {acc10m}"
        )

    def evaluate(self, data_iterator, mode):
        print("Mode-", mode)
        self.model.val_start()
        acc_k1, acc_topk, le = [], [], []
        for batch_data in tqdm.tqdm(data_iterator):
            k1, topk, e = self.model.eval_emb(*batch_data)
            acc_k1.append(k1)
            acc_topk.append(topk)
            le.extend(e)
        self.scores(mode, acc_k1, acc_topk, le)

    def train(self):
        print("\nStarting Training...")
        self.model = XRN(self.args)
        for _ in range(self.args.num_epoch):
            self.model.train_start()
            loss = []
            acc_k1, acc_topk, le = [], [], []
            print("Mode-", "train")
            for batch_data in tqdm.tqdm(self.train_iterator):
                k1, topk, e, l = self.model.train_emb(*batch_data)
                loss.append(l.item())
                acc_k1.append(k1)
                acc_topk.append(topk)
                le.extend(e)
            print(f"\tTraining Loss: {np.mean(loss)}")
            self.scores("Training", acc_k1, acc_topk, le)
            self.evaluate(self.valseen_iterator, mode="ValSeen")
            self.evaluate(self.val_unseen_iterator, mode="ValUnseen")

    def run(self):
        if self.args.train:
            self.load_data()
            self.train()
            self.evaluate(self.valseen_iterator, mode="ValSeen")
            self.evaluate(self.val_unseen_iterator, mode="ValUnseen")

        if self.args.evaluate:
            self.load_data()
            self.evaluate(self.test_iterator, mode="Test")


if __name__ == "__main__":
    args = cfg.parse_args()
    agent = LEDAgent(args)
    agent.run()
