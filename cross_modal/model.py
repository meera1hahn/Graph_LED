import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from collections import OrderedDict
import ipdb
from utils import get_geo_dist
import json


class EncoderImage(nn.Module):
    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(opt.pano_embed_size, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
        )
        self.predict = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )
        self.sig = nn.Sigmoid()

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)

    def forward(self, images, cap_emb, batch_size):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)
        features = features.flatten(1)
        features = torch.cat((features, cap_emb), -1)
        predict = self.predict(features)
        predict = F.log_softmax(predict.view(batch_size, -1), 1)
        # predict = self.sig(predict)
        return predict


class EncoderImageAttend(nn.Module):
    def __init__(self, opt):
        super(EncoderImageAttend, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(opt.pano_embed_size, 512),
            nn.ReLU(True),
        )
        self.attend_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12), num_layers=4
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
        )

    def forward(self, images, cap_emb):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)  # (N,36,512)
        cap_emb = cap_emb.unsqueeze(1).expand(-1, 36, -1)  # (N,36,256)
        features = torch.cat((features, cap_emb), -1)
        features = features.permute(1, 0, 2)
        features = self.attend_layers(features)
        features = features.permute(1, 0, 2)
        features = features.mean(1)
        return features


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
            128,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0,
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


class XRN(object):
    def __init__(self, opt):
        # Cuda
        if torch.cuda.is_available():
            self.device = (
                torch.device(f"cuda:{opt.cuda}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        # Build Models
        self.opt = opt
        # self.margin = opt.margin
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt)
        # self.img_enc = EncoderImageAttend(opt)
        self.txt_enc = EncoderText(opt)
        self.img_enc.to(self.device)
        self.txt_enc.to(self.device)
        # Parameters
        self.params = list(self.txt_enc.parameters())
        self.params += list(self.img_enc.fc.parameters())
        num_params = sum([p.numel() for p in self.params if p.requires_grad])
        print("Number of parameters:", num_params)

        # Loss and Optimizer
        self.criterion = nn.BCELoss()
        self.mrl_criterion = nn.MarginRankingLoss(margin=1)
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = torch.optim.Adam(self.params, lr=opt.lr)
        self.weight_init()
        self.geodistance_nodes = json.load(open("geodistance_nodes.json"))

    def weight_init(self):
        if self.opt.evaluate:
            init_path = "save/coco_resnet_restval/model_best.pth.tar"
            print("=> loading checkpoint '{}'".format(init_path))
            checkpoint = torch.load(init_path)
            self.load_state_dict(checkpoint["model"])

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        # switch to train mode
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        # switch to evaluate mode
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, node_feats, text, seq_length, batch_size):
        """Compute the image and caption embeddings"""
        # Set mini-batch dataset

        node_feats = node_feats.to(self.device)
        text = text.to(self.device)
        seq_length = seq_length.to(self.device)

        # Forward
        cap_emb = self.txt_enc(text, seq_length)
        img_emb = self.img_enc(node_feats, cap_emb, batch_size)
        return img_emb

    def forward_loss(self, anchor, predict):
        loss = self.kl_criterion(predict, anchor.float().to(self.device))
        return loss

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
        batch_size = node_feats.size()[0]
        neg_examples = node_feats.size()[1]
        node_feats = node_feats.view(batch_size * neg_examples, 36, 2048)
        text = torch.repeat_interleave(text, repeats=neg_examples, dim=0)
        seq_length = torch.repeat_interleave(seq_length, repeats=neg_examples, dim=0)
        predict = self.forward_emb(node_feats, text, seq_length, batch_size)
        loss = self.forward_loss(anchor, predict)
        loss.backward()
        self.optimizer.step()

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
            topk5 = torch.topk(predict[i, :][non_null], k)[1]
            topk_correct += top_true[i] in topk5

            graph = self.opt.scan_graphs[info_elem[2][i]]
            pred_vp = node_names[i, :][topk1]
            true_vp = node_names[i, :][top_true[i]]
            LE.append(get_geo_dist(graph, pred_vp, true_vp))

        accuracy_k1 = k1_correct * 1.0 / batch_size
        accuracy_topk = topk_correct * 1.0 / batch_size
        return accuracy_k1, accuracy_topk, LE, loss

    def eval_emb(
        self,
        text,
        seq_length,
        node_feats,
        anchor,
        node_names,
        info_elem,
        *args,
    ):
        batch_size = node_feats.size()[0]

        node_feats = node_feats.view(batch_size * self.opt.max_nodes, 36, 2048)
        text = torch.repeat_interleave(text, repeats=self.opt.max_nodes, dim=0)
        seq_length = torch.repeat_interleave(
            seq_length, repeats=self.opt.max_nodes, dim=0
        )

        predict = (
            self.forward_emb(node_feats, text, seq_length, batch_size).detach().cpu()
        )
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
            topk5 = torch.topk(predict[i, :][non_null], k)[1]
            topk_correct += top_true[i] in topk5

            graph = self.opt.scan_graphs[info_elem[2][i]]
            pred_vp = node_names[i, :][topk1]
            true_vp = node_names[i, :][top_true[i]]
            LE.append(get_geo_dist(graph, pred_vp, true_vp))

        accuracy_k1 = k1_correct * 1.0 / batch_size
        accuracy_topk = topk_correct * 1.0 / batch_size

        return accuracy_k1, accuracy_topk, LE
