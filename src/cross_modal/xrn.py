import numpy as np
import json
import torch
import torch.nn as nn

from src.utils import get_geo_dist
from src.cross_modal.models import AttentionModel, BasicModel


class XRN(object):
    def __init__(self, opt):
        print("using attention model")
        # Cuda
        self.device = (
            torch.device(f"cuda:{opt.cuda}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # Build Models
        self.opt = opt
        if opt.attention:
            self.model = AttentionModel(opt)
        else:
            self.model = BasicModel(opt)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        num_params = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        print("Number of parameters:", num_params)

        # Loss and Optimizer
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        self.geodistance_nodes = json.load(open("../geodistance_nodes.json"))

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self):
        print("=> loading checkpoint '{}'".format(self.opt.eval_ckpt))
        self.model.load_state_dict(torch.load(self.opt.eval_ckpt))

    def train_start(self):
        # switch to train mode
        self.model.train()

    def val_start(self):
        # switch to evaluate mode
        self.model.eval()

    def forward_loss(self, anchor, predict):
        loss = self.kl_criterion(predict, anchor.float().to(self.device))
        return loss

    def run_emb(
        self,
        text,
        seq_length,
        node_feats,
        anchor,
        node_names,
        info_elem,
        mode="eval",
        *args,
    ):

        batch_size = node_feats.size()[0]
        predict = self.model(
            node_feats.to(self.device),
            text.to(self.device),
            seq_length.to(self.device),
        ).squeeze(-1)

        loss = self.forward_loss(anchor, predict)

        # measure acc
        predict = predict.detach().cpu()
        anchor = anchor.detach().cpu()
        top_true = anchor.argmax(dim=1)
        node_names = np.transpose(np.asarray(node_names))
        k, topk_correct, k1_correct = 5, 0.0, 0.0
        le, episode_predictions = [], []
        for i in range(batch_size):
            non_null = np.where(node_names[i, :] != "null")[0]
            topk1 = non_null[torch.tensor(np.asarray(predict[i, :][non_null])).argmax()]
            pred_vp = node_names[i, :][topk1]
            topk5 = torch.topk(predict[i, :], k)[1]
            graph = self.opt.scan_graphs[info_elem[1][i]]
            if mode != "test":
                k1_correct += torch.eq(torch.tensor(topk1), top_true[i])
                topk_correct += top_true[i] in topk5
                true_vp = node_names[i, :][top_true[i]]
                le.append(get_geo_dist(graph, pred_vp, true_vp))
            episode_predictions.append([info_elem[2][i], pred_vp])

        accuracy_k1 = k1_correct * 1.0 / batch_size
        accuracy_topk = topk_correct * 1.0 / batch_size
        return accuracy_k1, accuracy_topk, le, loss, episode_predictions

    def train_emb(
        self,
        text,
        seq_length,
        node_feats,
        anchor,
        node_names,
        info_elem,
        *args,
    ):
        self.optimizer.zero_grad()
        accuracy_k1, accuracy_topk, le, loss, _ = self.run_emb(
            text, seq_length, node_feats, anchor, node_names, info_elem, mode="train"
        )
        loss.backward()
        self.optimizer.step()

        return accuracy_k1, accuracy_topk, le, loss
