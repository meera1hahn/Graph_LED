import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils import get_geo_dist

"""MODEL WITH CROSSMODAL ATTENTION"""


class AttentionModel(nn.Module):
    def __init__(self, opt):
        super(AttentionModel, self).__init__()
        # Create Models
        self.txt_encoder = EncoderTextAttn(opt)
        self.img_encoder = EncoderImageAttn(opt)
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


class EncoderImageAttn(nn.Module):
    def __init__(self, opt):
        super(EncoderImageAttn, self).__init__()
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


class EncoderTextAttn(nn.Module):
    def __init__(self, opt):
        super(EncoderTextAttn, self).__init__()
        self.bidirectional = opt.bidirectional
        self.input_size = opt.rnn_input_size
        self.embed_size = opt.rnn_embed_size
        self.hidden_size = opt.rnn_hidden_size
        self.reduce = "last" if not opt.bidirectional else "mean"
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
            num_layers=1,  # self.num_layers,
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, seq_lengths):
        embed = self.embedding(x)
        embed = self.dropout(embed)
        embed_packed = pack_padded_sequence(
            embed, seq_lengths.cpu(), enforce_sorted=False, batch_first=True
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


##################### MODEL 2 #####################
"""BASIC MODEL"""


class BasicModel(nn.Module):
    def __init__(self, opt):
        super(BasicModel, self).__init__()
        # Create Models
        self.txt_encoder = EncoderText(opt)
        self.img_encoder = EncoderImage(opt)

    def forward(self, node_feats, text, seq_length):
        """Compute the image and caption embeddings"""
        cap_emb = self.txt_encoder(text, seq_length)
        img_emb = self.img_encoder(node_feats, cap_emb)
        return img_emb


class EncoderImage(nn.Module):
    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.ffc = nn.Sequential(
            nn.Linear(opt.pano_embed_size, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
        )
        self.predict = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )
        self.sig = nn.Sigmoid()
        self.ffc.apply(self.init_weights)
        self.predict.apply(self.init_weights)

    def forward(self, images, cap_emb):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        batch_size, num_nodes, patches, feats = images.size()
        features = self.ffc(images)  # (N,340,36,64)
        features = features.flatten(2)  # (N,340,2304)
        feat_size = cap_emb.size()[1]
        cap_emb = cap_emb.unsqueeze(1).expand(
            batch_size, num_nodes, feat_size
        )  # (N,340,cap_embed_size)
        features = torch.mul(features, cap_emb)
        predict = self.predict(features)  # (N,340,2560)
        predict = F.log_softmax(predict, 1)
        return predict

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class EncoderText(nn.Module):
    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.bidirectional = opt.bidirectional
        self.input_size = opt.rnn_input_size
        self.embed_size = opt.rnn_embed_size
        self.hidden_size = opt.rnn_hidden_size
        self.reduce = "last" if not opt.bidirectional else "mean"
        self.embedding_dir = opt.embedding_dir

        glove_weights = torch.FloatTensor(
            np.load(self.embedding_dir + "glove_weights_matrix.npy", allow_pickle=True)
        )
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.embedding.from_pretrained(glove_weights)

        self.lstm = nn.LSTM(
            self.embed_size,
            1152,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.0,
            num_layers=1,
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, seq_lengths):
        embed = self.embedding(x)
        embed = self.dropout(embed)
        embed_packed = pack_padded_sequence(
            embed, seq_lengths.cpu(), enforce_sorted=False, batch_first=True
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
