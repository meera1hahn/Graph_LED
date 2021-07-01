import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import copy
from collections import OrderedDict
import ipdb

from loader import Loader


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

    def forward(self, images, cap_emb):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)
        features = features.flatten(1)
        features = torch.cat((features, cap_emb), -1)
        predict = self.sig(self.predict(features))
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
        ipdb.set_trace()
        features = self.fc(images)  # (N,36,512)
        cap_emb = cap_emb.unsqueeze(1).expand(-1, 36, -1)  # (N,36,256)
        features = torch.cat((features, cap_emb), -1)
        features = features.permute(1, 0, 2)
        features = self.attend_layers(features)
        features = features.permute(1, 0, 2)
        features = features.mean(1)
        ipdb.set_trace()
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
        self.optimizer = torch.optim.Adam(self.params, lr=opt.lr)
        self.Eiters = 0
        self.weight_init()

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

    def forward_emb(self, node_feats, text, seq_length):
        """Compute the image and caption embeddings"""
        # Set mini-batch dataset
        node_feats = node_feats.to(self.device)
        text = text.to(self.device)
        seq_length = seq_length.to(self.device)

        # Forward
        cap_emb = self.txt_enc(text, seq_length)
        img_emb = self.img_enc(node_feats, cap_emb)
        return img_emb

    def forward_loss(self, anchor, predict):
        # Compute the loss given pairs of image and caption embeddings
        loss = self.criterion(predict, anchor.float().flatten().to(self.device))
        return loss

    def train_emb(
        self,
        viz_elem,
        info_elem,
        text,
        seq_length,
        node_feats,
        node_names,
        anchor,
        *args,
    ):
        # One training step given images and captions.
        self.Eiters += 1
        self.optimizer.zero_grad()
        node_feats = node_feats.view(node_feats.size()[0] * 2, 36, 2048)

        text = torch.repeat_interleave(text, repeats=2, dim=0)
        seq_length = torch.repeat_interleave(seq_length, repeats=2, dim=0)
        predict = self.forward_emb(node_feats, text, seq_length).squeeze(1)
        # measure and record loss
        loss = self.forward_loss(anchor, predict)
        # compute gradient and do SGD step
        loss.backward()
        self.optimizer.step()
        # measure acc
        correct_all = (
            torch.eq(
                anchor.float().flatten(), np.round(predict.detach().cpu()).flatten()
            ).sum()
            * 1.0
            / anchor.float().flatten().size()[0]
        )
        predict = predict.view(-1, 2, 1).detach().cpu()
        top_pred = predict.squeeze(2).argmax(dim=1)
        top_true = anchor.argmax(dim=1)
        correct_top = torch.eq(top_pred, top_true).sum() * 1.0 / top_pred.size()[0]
        return loss, correct_all, correct_top

    def eval_emb(
        self,
        viz_elem,
        info_elem,
        text,
        seq_length,
        node_feats,
        node_names,
        anchor,
        *args,
    ):
        batch_size = len(info_elem[0])
        node_names = np.asarray(node_names)
        node_feats = node_feats.view(batch_size * self.opt.max_nodes, 36, 2048)
        text = torch.repeat_interleave(text, repeats=self.opt.max_nodes, dim=0)
        seq_length = torch.repeat_interleave(
            seq_length, repeats=self.opt.max_nodes, dim=0
        )
        predict = self.forward_emb(node_feats, text, seq_length)

        predict = predict.view(-1, self.opt.max_nodes, 1).detach().cpu()
        top_pred = predict.squeeze(2).argmax(dim=1)
        top_true = anchor.argmax(dim=1)
        correct_top = torch.eq(top_pred, top_true).sum() * 1.0 / batch_size

        top_k_viewpoints = []
        for i in range(batch_size):
            non_null = np.where(node_names[:, i] != "null")[0]
            top = torch.tensor(np.asarray(predict[i, :, :][non_null])).argmax()
            top_k_viewpoints.append(node_names[top, i])

        # correct_names = (
        #     np.sum(np.asarray(top_k_viewpoints) == np.asarray(info_elem[-1]))
        #     * 1.0
        #     / batch_size
        # )
        # try:
        #     assert round(correct_names, 2) == round(
        #         correct_top.item(), 2
        #     ), "correct names should equal correct indices"
        # except:
        #     ipdb.set_trace()

        return top_k_viewpoints, correct_top
