import torch
from torch import nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from utils import get_geo_dist
import json


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.features_att = weight_norm(
            nn.Linear(features_dim, attention_dim)
        )  # linear layer to transform encoded image
        self.decoder_att = weight_norm(
            nn.Linear(decoder_dim, attention_dim)
        )  # linear layer to transform decoder's output
        self.full_att = weight_norm(
            nn.Linear(attention_dim, 1)
        )  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.features_att(image_features)  # (batch_size, 36, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(
            2
        )  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 36)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, features_dim)

        return attention_weighted_encoding


class FullModel(nn.Module):
    def __init__(self, opt):
        super(FullModel, self).__init__()
        # Create Models
        self.txt_encoder = EncoderText(opt)
        self.img_encoder = EncoderImage(opt)
        # self.img_encoder = EncoderImageAttend(opt)

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

    def forward(self, images, cap_emb):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        batch_size, num_nodes = images.size()[0:2]
        features = self.ffc(images)  # (N,340,36,64)
        features = features.flatten(2)  # (N,340,2304)
        cap_emb = cap_emb.unsqueeze(1).expand(
            batch_size, num_nodes, 2304
        )  # (N,340,cap_embed_size)
        features = torch.mul(features, cap_emb)
        # features = torch.cat((features, cap_emb), -1)
        predict = self.predict(features)  # (N,340,2560)
        predict = F.log_softmax(predict, 1)
        return predict


# class EncoderImage(nn.Module):
#     def __init__(self, opt):
#         super(EncoderImage, self).__init__()
#         self.ffc = nn.Sequential(
#             nn.Linear(opt.pano_embed_size, 512),
#             nn.ReLU(True),
#         )
#         self.predict = nn.Sequential(
#             nn.Linear(512, 1),
#         )
#         self.sig = nn.Sigmoid()

#     def forward(self, images, cap_emb):
#         """Extract image feature vectors."""
#         # assuming that the precomputed features are already l2-normalized
#         batch_size = node_feats.size()[0]
#         features = self.ffc(images)
#         features = torch.mul((features, cap_emb), -1)
#         predict = self.predict(features)
#         predict = F.log_softmax(predict.view(batch_size, -1), 1)
#         return predict


# class EncoderImageAttend(nn.Module):
#     def __init__(self, opt):
#         super(EncoderImageAttend, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(opt.pano_embed_size, 512),
#             nn.ReLU(True),
#         )
#         self.attend_layers = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=768, nhead=12), num_layers=3
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(768, 1),
#         )

#     def forward(self, images, cap_emb, batch_size):
#         """Extract image feature vectors."""
#         # assuming that the precomputed features are already l2-normalized
#         features = self.fc(images)  # (N,36,512)
#         cap_emb = cap_emb.unsqueeze(1).expand(-1, 36, -1)  # (N,36,256)
#         features = torch.mul((features, cap_emb), -1)  # (N,36,768)
#         features = features.permute(1, 0, 2)
#         features = self.attend_layers(features)
#         features = features.permute(1, 0, 2)
#         features = features.mean(1)
#         predictions = F.log_softmax(self.fc2(features).view(batch_size, -1), 1)
#         return predictions


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
            1152,
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


class XRN(object):
    def __init__(self, opt):
        # Cuda
        self.device = (
            torch.device(f"cuda:{opt.cuda}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # Build Models
        self.opt = opt
        self.model = FullModel(opt)
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
        self.weight_init()
        self.geodistance_nodes = json.load(open("geodistance_nodes.json"))

    def weight_init(self):
        if self.opt.evaluate:
            init_path = "save/coco_resnet_restval/model_best.pth.tar"
            print("=> loading checkpoint '{}'".format(init_path))
            checkpoint = torch.load(init_path)
            self.load_state_dict(checkpoint["model"])

    def state_dict(self):
        state_dict = self.model.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

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
