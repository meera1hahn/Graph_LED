import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import tqdm
import numpy as np
import os.path
import os
import copy
import json

from src.lingunet.lingunet_cfg import parse_args
from src.lingunet.loader import Loader
from src.lingunet.lingunet_model import LingUNet, load_oldArgs, convert_model_to_state
from src.utils import accuracy, distance_from_pixels, evaluate


class LingUNetAgent:
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
        self.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
        if args.train and args.model_save:
            if not os.path.isdir(self.checkpoint_dir):
                print("Checkpoint directory under {}".format(self.checkpoint_dir))
                os.system("mkdir {}".format(self.checkpoint_dir))
            self.writer = SummaryWriter(args.summary_dir + args.run_name)

        self.model = None
        self.optimizer = None

    def run_test(self):
        print("Starting Evaluation...")
        oldArgs, rnn_args, state_dict = torch.load(self.args.eval_ckpt).values()
        self.args = load_oldArgs(self.args, oldArgs)
        self.model = LingUNet(rnn_args, self.args)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device=self.args.device)

        loss, acc0m, acc5m = self.eval_model(self.valseen_iterator, "valSeen")
        self.scores("valSeen", loss, acc0m, acc5m, 0)
        evaluate(self.args, "valSeen_data.json", self.args.run_name)
        loss, acc0m, acc5m = self.eval_model(self.val_unseen_iterator, "valUnseen")
        self.scores("valUnseen", loss, acc0m, acc5m, 0)
        evaluate(self.args, "valUnseen_data.json", self.args.run_name)
        loss, acc0m, acc5m = self.eval_model(self.test_iterator, "test")

    def run_epoch(
        self,
        info_elem,
        batch_texts,
        batch_seq_lengths,
        batch_target,
        batch_maps,
        batch_conversions,
        mode,
    ):
        B, num_maps, C, H, W = batch_maps.size()
        preds = self.model(
            batch_maps.to(device=self.args.device),
            batch_texts.to(device=self.args.device),
            batch_seq_lengths.to(device=self.args.device),
        )

        """ calculate loss """
        batch_target = batch_target.view(B * num_maps, H, W)
        batch_target = (
            nn.functional.interpolate(
                batch_target.unsqueeze(1),
                (self.args.ds_height, self.args.ds_width),
                mode="bilinear",
            )
            .squeeze(1)
            .float()
        ).to(device=self.args.device)
        batch_target = batch_target.view(
            B, num_maps, batch_target.size()[-2], batch_target.size()[-1]
        )

        loss = self.loss_func(preds, batch_target)
        le, ep = distance_from_pixels(
            args, preds.detach().cpu(), batch_conversions, info_elem, mode
        )
        return loss, accuracy(le, 0), accuracy(le, 5), ep

    def eval_model(self, data_iterator, mode):
        self.model.eval()
        loss, accuracy0m, accuracy5m = [], [], []
        submission = {}
        for (
            info_elem,
            texts,
            seq_lengths,
            target,
            maps,
            conversions,
        ) in tqdm.tqdm(data_iterator):
            l, acc0m, acc5m, ep = self.run_epoch(
                info_elem, texts, seq_lengths, target, maps, conversions, mode
            )
            loss.append(l.item())
            accuracy0m.append(acc0m)
            accuracy5m.append(acc5m)
            for i in ep:
                submission[i[0]] = {"viewpoint": i[1]}

        if self.args.evaluate:
            fileName = f"{self.args.run_name}_{mode}_submission.json"
            fileName = os.path.join(self.args.predictions_dir, fileName)
            json.dump(submission, open(fileName, "w"), indent=3)
            print("submission saved at ", fileName)
        return (
            np.mean(loss),
            np.mean(np.asarray(accuracy0m)),
            np.mean(np.asarray(accuracy5m)),
        )

    def train_model(self):
        self.model.train()
        loss, accuracy0m, accuracy5m = [], [], []
        for (
            info_elem,
            texts,
            seq_lengths,
            target,
            maps,
            conversions,
        ) in tqdm.tqdm(self.train_iterator):
            self.optimizer.zero_grad()
            l, acc0m, acc5m, _ = self.run_epoch(
                info_elem, texts, seq_lengths, target, maps, conversions, mode="train"
            )
            l.backward()
            self.optimizer.step()
            loss.append(l.item())
            accuracy0m.append(acc0m)
            accuracy5m.append(acc5m)

        return (
            np.mean(loss),
            np.mean(np.asarray(accuracy0m)),
            np.mean(np.asarray(accuracy5m)),
        )

    def tensorboard_writer(self, mode, loss, acc0m, acc5m, epoch):
        self.writer.add_scalar("Loss/" + mode, np.mean(loss), epoch)
        self.writer.add_scalar("Acc@0m/" + mode, acc0m, epoch)
        self.writer.add_scalar("Acc@5m/" + mode, acc5m, epoch)

    def scores(self, mode, loss, acc0m, acc5m, epoch):
        print(
            f"\t{mode} Epoch:{epoch} Loss:{loss} Acc@0m: {np.mean(acc0m)} Acc@5m: {np.mean(acc5m)}"
        )

    def run_train(self):
        assert self.args.num_lingunet_layers is not None
        rnn_args = {"input_size": len(self.loader.vocab)}

        self.model = LingUNet(rnn_args, args)
        num_params = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        print("Number of parameters:", num_params)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device=self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        print("Starting Training...")
        best_unseen_acc = float("-inf")
        best_model, save_path, patience = None, "", 0
        for epoch in range(self.args.num_epoch):
            loss, acc0m, acc5m = self.train_model()
            self.scores("train", loss, acc0m, acc5m, epoch)
            self.tensorboard_writer("train", loss, acc0m, acc5m, epoch)
            loss, acc0m, acc5m = self.eval_model(self.valseen_iterator, "val_seen")
            self.scores("val_seen", loss, acc0m, acc5m, epoch)
            self.tensorboard_writer("val_seen", loss, acc0m, acc5m, epoch)
            loss, acc0m, acc5m = self.eval_model(self.val_unseen_iterator, "val_unseen")
            self.scores("val_unseen", loss, acc0m, acc5m, epoch)
            self.tensorboard_writer("val_unseen", loss, acc0m, acc5m, epoch)

            if acc0m > best_unseen_acc:
                best_model = copy.deepcopy(self.model)
                if self.args.model_save:
                    save_path = os.path.join(
                        self.checkpoint_dir,
                        "{}_unseenAcc{:.4f}_epoch{}.pt".format(
                            self.args.name, acc0m, epoch
                        ),
                    )
                    state = convert_model_to_state(best_model, args, rnn_args)
                    torch.save(state, save_path)
                best_unseen_acc = acc0m
                patience = 0
                print("[Tune]: Best valUNseen accuracy:", best_unseen_acc)
            else:
                patience += 1
                if patience >= self.args.early_stopping:
                    break
            print("Patience:", patience)
        print(f"Best model saved at: {save_path}")

    def load_data(self):
        self.loader = Loader(args)
        self.loader.build_dataset(file="train_expanded_data.json")
        self.loader.build_dataset(file="valSeen_data.json")
        self.loader.build_dataset(file="valUnseen_data.json")
        self.train_iterator = DataLoader(
            self.loader.datasets["train"],
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        self.valseen_iterator = DataLoader(
            self.loader.datasets["valSeen"],
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        self.val_unseen_iterator = DataLoader(
            self.loader.datasets["valUnseen"],
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        if self.args.evaluate:
            self.loader.build_dataset(file="test_data.json")
            self.test_iterator = DataLoader(
                self.loader.datasets["test"],
                batch_size=self.args.batch_size,
                shuffle=False,
            )

    def run(self):
        self.load_data()
        if self.args.train:
            self.run_train()

        elif self.args.evaluate:
            self.run_test()


if __name__ == "__main__":
    args = parse_args()
    agent = LingUNetAgent(args)
    agent.run()
