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
            batch_size=2,
            shuffle=False,
        )

        self.val_unseen_iterator = DataLoader(
            self.loader.datasets["valUnseen"],
            batch_size=2,
            shuffle=False,
        )
        if self.args.evaluate:
            self.loader.build_dataset(file="test_data.json")
            self.test_iterator = DataLoader(
                self.loader.datasets["test"],
                batch_size=2,
                shuffle=False,
            )

    # def validate(self, data_iterator, mode):
    #     print("Mode-", mode)
    #     self.model.val_start()
    #     distances = []
    #     correct = []
    #     for batch_data in tqdm.tqdm(data_iterator):
    #         preds, c = self.model.eval_emb(*batch_data)
    #         _, info_elem, _, _, _, _, _ = batch_data
    #         if mode != "test":
    #             distances.extend(
    #                 geo_dist_singlefloor(self.args.scan_graphs, preds, info_elem)
    #             )
    #             correct.append(c)
    #     if mode != "test":
    #         print(f"\t{0}m-Acc: {np.mean(correct)}")
    #         print(
    #             f"\t{0}m-Acc: {round(accuracy(distances, 0), 3)}, {3}m-Acc: {round(accuracy(distances, 3), 3)}, {5}m-Acc: {round(accuracy(distances, 5), 3)}, LE: {round(np.mean(distances), 3)}"
    #         )

    def run_train(self):
        print("\nStarting Training...")
        self.model = XRN(self.args)
        for _ in range(self.args.num_epoch):
            self.model.train_start()
            loss = []
            correct_all = []
            correct_top = []
            print("Mode-", "train")
            for batch_data in tqdm.tqdm(self.train_iterator):
                l, ca, ct = self.model.train_emb(*batch_data)
                loss.append(l.item())
                correct_all.append(ca)
                correct_top.append(ct)
            print(f"\tTraining Loss: {np.mean(loss)}")
            print(
                f"\tTraining acc_all: {np.mean(correct_all)}; acc_top: {np.mean(correct_top)}"
            )
            # self.validate(self.train_iterator, mode="train")
            # self.validate(self.valseen_iterator, mode="valSeen")
            # self.validate(self.val_unseen_iterator, mode="valUnseen")

    def run(self):
        if self.args.train:
            self.load_data()
            self.run_train()

        # elif self.args.evaluate:
        #     self.load_data()
        #     self.validate()


if __name__ == "__main__":
    args = cfg.parse_args()
    agent = LEDAgent(args)
    agent.run()
