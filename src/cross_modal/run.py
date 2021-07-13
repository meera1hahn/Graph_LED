import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import tqdm
import numpy as np
import os.path
import os
import json

from loader import Loader
from model import XRN
from attention import AttentionXRN
from src.cfg import *


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
        self.log_dir = os.path.join(args.log_dir, args.run_name)
        self.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
        if (args.log and args.train) or args.model_save:
            if not os.path.isdir(self.log_dir):
                print("Log directory under {}".format(self.log_dir))
                os.system("mkdir {}".format(self.log_dir))
            if not os.path.isdir(self.checkpoint_dir):
                print("Checkpoint directory under {}".format(self.checkpoint_dir))
                os.system("mkdir {}".format(self.checkpoint_dir))
            self.writer = SummaryWriter(args.summary_dir + args.name)

        self.model = None
        self.optimizer = None
        self.node2pix = json.load(open(args.image_dir + "allScans_Node2pix.json"))
        self.args.rnn_input_size = len(
            json.load(open(self.args.embedding_dir + "word2idx.json"))
        )
        self.best_val_acc = -0.1
        self.patience = 0
        self.savename = ""

    def load_data(self):
        print("Loading Data...")
        self.loader = Loader(data_dir=self.args.data_dir, args=self.args)
        self.loader.build_dataset(file="train_expanded_data.json")
        self.loader.build_dataset(file="valSeen_data.json")
        self.loader.build_dataset(file="valUnseen_data.json")
        self.train_iterator = DataLoader(
            self.loader.datasets["train"],
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.valseen_iterator = DataLoader(
            self.loader.datasets["valSeen"],
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.val_unseen_iterator = DataLoader(
            self.loader.datasets["valUnseen"],
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if self.args.evaluate:
            self.loader.build_dataset(file="test_data_full.json")
            self.test_iterator = DataLoader(
                self.loader.datasets["test"],
                batch_size=self.args.batch_size,
                shuffle=False,
            )

    def tensorboard_writer(self, mode, epoch, loss, acc_topk, le):
        le = np.asarray(le)
        acc0m = sum(le <= 0) * 1.0 / len(le)
        acc5m = sum(le <= 5) * 1.0 / len(le)
        acc10m = sum(le <= 10) * 1.0 / len(le)
        self.writer.add_scalar("Loss/" + mode, np.mean(loss), epoch)
        self.writer.add_scalar("LE/" + mode, np.mean(le), epoch)
        self.writer.add_scalar("Acc@5k/" + mode, np.mean(acc_topk), epoch)
        self.writer.add_scalar("Acc@0m/" + mode, acc0m, epoch)
        self.writer.add_scalar("Acc@5m/" + mode, acc5m, epoch)
        self.writer.add_scalar("Acc@10m/" + mode, acc10m, epoch)

    def scores(self, mode, acc_k1, acc_topk, le):
        print(f"\t{mode} Acc@1k: {np.mean(acc_k1)} Acc@5k: {np.mean(acc_topk)}")
        le = np.asarray(le)
        acc0m = sum(le <= 0) * 1.0 / len(le)
        acc5m = sum(le <= 5) * 1.0 / len(le)
        acc10m = sum(le <= 10) * 1.0 / len(le)
        print(
            f"\t{mode} LE: {np.mean(le):.4f}, Acc@0m: {acc0m:.4f}, Acc@5m: {acc5m:.4f}, Acc@10m: {acc10m:.4f}"
        )

    def evaluate(self, epoch, data_iterator, mode):
        print("Mode-", mode)
        self.model.val_start()
        loss = []
        acc_k1, acc_topk, le = [], [], []
        for batch_data in tqdm.tqdm(data_iterator):
            k1, topk, e, l = self.model.run_emb(*batch_data)
            loss.append(l.item())
            acc_k1.append(k1)
            acc_topk.append(topk)
            le.extend(e)
        self.scores(mode, acc_k1, acc_topk, le)
        self.tensorboard_writer(mode.lower(), epoch, loss, acc_topk, le)
        if mode == "ValUnseen" and self.args.model_save:
            self.savename = (
                f"{self.checkpoint_dir}/Epoch{epoch}_Acc1K-{np.mean(acc_k1):.4f}.pt"
            )
            torch.save(self.model.get_state_dict(), self.savename)
            print(f"saved at {self.savename}")
            if np.mean(acc_k1) > self.best_val_acc:
                self.best_val_acc = np.mean(acc_k1)
                self.patience = -1
            self.patience += 1

    def train(self):
        print("\nStarting Training...")
        for epoch in range(0, self.args.num_epoch):
            print("Epoch ", epoch)
            self.model.train_start()
            loss = []
            acc_k1, acc_topk, le = [], [], []
            print("Mode-", "train")
            for batch_data in tqdm.tqdm(self.train_iterator):
                k1, topk, e, l = self.model.train_emb(*batch_data)
                del batch_data
                try:
                    loss.append(l.item())
                    acc_k1.append(k1)
                    acc_topk.append(topk)
                    le.extend(e)
                except:
                    import ipdb

                    ipdb.set_trace()
            print(f"\tTraining Loss: {np.mean(loss)}")
            self.scores("Training", acc_k1, acc_topk, le)
            self.tensorboard_writer("train", epoch, loss, acc_topk, le)
            self.evaluate(epoch, self.valseen_iterator, mode="ValSeen")
            self.evaluate(epoch, self.val_unseen_iterator, mode="ValUnseen")
            if self.patience > self.args.early_stopping:
                print(f"Patience Reached. Ending Training at Epoch {epoch}.")
                break

    def run(self):
        if self.args.train:
            self.load_data()
            if self.args.attention:
                self.model = AttentionXRN(self.args)
            else:
                self.model = XRN(self.args)
                # self.model.load_state_dict()
            self.train()
            print("Training Ended...")
            print("Last Model Saved @", self.savename)

        if self.args.evaluate:
            self.load_data()
            if self.args.attention:
                self.model = AttentionXRN(self.args)

            else:
                self.model = XRN(self.args)
            self.model.load_state_dict()
            self.evaluate(0, self.valseen_iterator, mode="ValSeen")
            self.evaluate(0, self.val_unseen_iterator, mode="ValUnseen")


if __name__ == "__main__":
    args = parse_args()
    agent = LEDAgent(args)
    agent.run()
