import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils import get_geo_dist


class AttentionModel(nn.Module):
    def __init__(self, opt):
        super(AttentionModel, self).__init__()
        # Create Models
        self.txt_encoder = EncoderText(opt)
        self.img_encoder = EncoderImage(opt)
        self.predict = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )
        self.img_encoder.img_lin1.apply(self.init_weights)
        self.predict.apply(self.init_weights)

    def forward(self, node_feats, text, seq_length):
        """Compute the image and caption embeddings"""
        cap_emb = self.txt_encoder(text, seq_length)
        img_emb = self.img_encoder(node_feats, cap_emb)
        joint_emb = torch.mul(
            img_emb,
            cap_emb.unsqueeze(1).expand(
                node_feats.size()[0], node_feats.size()[1], 2048
            ),
        )
        joint_emb = self.predict(joint_emb)
        predict = F.log_softmax(joint_emb, 1)
        return predict

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class EncoderImage(nn.Module):
    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        # self.img_fc = nn.Sequential(
        #     nn.Linear(opt.pano_embed_size, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 64),
        # )
        self.img_lin1 = nn.Linear(opt.pano_embed_size, 2048)
        # self.img_lin2 = nn.Linear(opt.pano_embed_size, 1024)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images, cap_emb):
        """self attention"""
        # query = self.img_lin1(images)  # (N,340,36,1024)
        # key = self.img_lin2(images)  # (N,340,36,1024)
        # batch, pano, patch, feat = query.size()
        # query = query.view(batch * pano, patch, feat).permute(
        #     0, 2, 1
        # )  # (N,340,1024, 36)
        # key = key.view(batch * pano, patch, feat)  # (N*340,36,1024)
        # attention = self.softmax(torch.bmm(query, key))# (N*340,36,36)
        # proj = self.img_fc(images).view(batch * pano, patch, feat)
        # img_emb = torch.bmm(proj, attention.permute(0, 2, 1))
        # img_emb = img_emb.view(batch, pano, patch, feat)

        """dialog attention"""
        img_emb = self.img_lin1(images)
        batch, pano, patch, feat = img_emb.size()
        img_emb = img_emb.view(batch * pano, patch, feat)
        cap_emb = torch.repeat_interleave(cap_emb, pano, dim=0)
        attention = self.softmax(torch.bmm(img_emb, cap_emb.unsqueeze(2)))
        img_emb = torch.bmm(attention.permute(0, 2, 1), img_emb)
        img_emb = img_emb.view(batch, pano, feat)
        return img_emb


class EncoderText(nn.Module):
    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.bidirectional = opt.bidirectional
        self.input_size = opt.rnn_input_size
        self.embed_size = opt.rnn_embed_size
        self.hidden_size = opt.rnn_hidden_size
        self.num_layers = opt.num_rnn_layers
        self.reduce = "last" if not opt.bidirectional else "mean"
        self.embedding_type = opt.embedding_type
        self.embedding_dir = opt.embedding_dir

        glove_weights = torch.FloatTensor(
            np.load(self.embedding_dir + "glove_weights_matrix.npy", allow_pickle=True)
        )
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.embedding.from_pretrained(glove_weights)

        self.lstm = nn.LSTM(
            self.embed_size,
            1024,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.0,
            num_layers=self.num_layers,
        )
        self.dropout = nn.Dropout(p=opt.embed_dropout)

    def forward(self, x, seq_lengths):
        embed = self.embedding(x)
        embed = self.dropout(embed)
        embed_packed = pack_padded_sequence(
            embed, seq_lengths, enforce_sorted=False, batch_first=True
        )

        out_packed = embed_packed
        self.lstm.flatten_parameters()
        out_packed, _ = self.lstm(out_packed)
        out, _ = pad_packed_sequence(out_packed)

        # reduce the dimension
        if self.reduce == "last":
            out = out[seq_lengths - 1, np.arange(len(seq_lengths)), :]
        elif self.reduce == "mean":
            seq_lengths_ = seq_lengths.unsqueeze(-1)
            out = torch.sum(out[:, np.arange(len(seq_lengths_)), :], 0) / seq_lengths_

        return out


class AttentionXRN(object):
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
        self.model = AttentionModel(opt)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        num_params = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        print("Number of parameters:", num_params)

        # Loss and Optimizer
        self.criterion = nn.BCELoss()
        self.mrl_criterion = nn.MarginRankingLoss(margin=1)
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        self.geodistance_nodes = json.load(open("geodistance_nodes.json"))

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self):
        print("=> loading checkpoint '{}'".format(self.opt.eval_ckpt))
        self.model.load_state_dict(self.opt.eval_ckpt)

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
        k = 5
        k1_correct = 0.0
        topk_correct = 0.0
        LE = []
        for i in range(batch_size):
            non_null = np.where(node_names[i, :] != "null")[0]
            topk1 = torch.tensor(np.asarray(predict[i, :][non_null])).argmax()
            k1_correct += torch.eq(topk1, top_true[i])
            topk5 = torch.topk(predict[i, :], k)[1]
            topk_correct += top_true[i] in topk5
            graph = self.opt.scan_graphs[info_elem[2][i]]
            pred_vp = node_names[i, :][topk1]
            true_vp = node_names[i, :][top_true[i]]
            LE.append(get_geo_dist(graph, pred_vp, true_vp))

        accuracy_k1 = k1_correct * 1.0 / batch_size
        accuracy_topk = topk_correct * 1.0 / batch_size
        return accuracy_k1, accuracy_topk, LE, loss

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
        accuracy_k1, accuracy_topk, LE, loss = self.run_emb(
            text, seq_length, node_feats, anchor, node_names, info_elem, mode="train"
        )
        loss.backward()
        self.optimizer.step()

        return accuracy_k1, accuracy_topk, LE, loss
