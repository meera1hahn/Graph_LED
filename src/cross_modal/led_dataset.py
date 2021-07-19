from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import json


class LEDDataset(Dataset):
    def __init__(
        self,
        mode,
        args,
        scan_names,
        episode_ids,
        viewpoints,
        texts,
        seq_lengths,
        dialogs,
    ):
        self.mode = mode
        self.args = args
        self.scan_names = scan_names
        self.episode_ids = episode_ids
        self.viewpoints = viewpoints
        # self.pano_feats = pano_feats
        self.texts = texts
        self.seq_lengths = seq_lengths
        self.dialogs = dialogs
        self.node_levels = None
        ## if visualization
        self.node_levels = json.load(open(args.image_dir + "allScans_Node2pix.json"))
        self.geodistance_nodes = json.load(open("../geodistance_nodes.json"))
        self.feat_dir = "/srv/flash1/mhahn30/LED/node_feats/"
        # max nodes
        self.max_nodes = self.args.max_nodes
        if self.mode == "test" or self.mode == "valUnseen":
            self.max_nodes = self.args.max_nodes_test

    def get_info(self, index):
        viz_elem = None
        ## if visualization
        level = self.node_levels[self.scan_names[index]][self.viewpoints[index]][-1]
        pretty_image = Image.open(
            "{}floor_{}/{}_{}.png".format(
                self.args.image_dir,
                level,
                self.scan_names[index],
                level,
            )
        )
        #
        viz_elem = np.asarray(pretty_image)[:, :, :3]

        info_elem = [
            self.dialogs[index],
            level,
            self.scan_names[index],
            self.episode_ids[index],
            self.viewpoints[index],
        ]
        return viz_elem, info_elem

    def get_test_items(self, index):
        scan_id = self.scan_names[index]
        node_feats = torch.load(self.feat_dir + scan_id + ".pt")
        node_names = sorted([n for n in self.args.scan_graphs[scan_id].nodes()])
        anchor_index = node_names.index(self.viewpoints[index])
        for _ in range(len(node_names), self.max_nodes):
            node_names.append("null")
        return node_names, node_feats, anchor_index

    def get_train_items(self, index):
        scan_id = self.scan_names[index]
        vp = self.viewpoints[index]
        dists = self.geodistance_nodes[scan_id][vp]
        target_feats = torch.load(self.feat_dir + scan_id + ".pt")
        target_names = sorted([n for n in self.args.scan_graphs[scan_id].nodes()])
        target_probabilites = torch.zeros((self.max_nodes))
        for i, node in enumerate(target_names):
            if node == vp:
                target_probabilites[i] = 1.0
            elif dists[node] <= 1.5:
                target_probabilites[i] = 0.45
            elif dists[node] <= 3.25:
                target_probabilites[i] = 0.0
            else:
                target_probabilites[i] = 0.0
        target_probabilites = target_probabilites / target_probabilites.sum()
        for _ in range(len(target_names), self.max_nodes):
            target_names.append("null")

        # shuffle train indices
        shuffle_indices = np.arange(self.max_nodes)
        np.random.shuffle(shuffle_indices)
        return (
            np.asarray(target_names)[shuffle_indices].tolist(),
            target_feats[shuffle_indices],
            target_probabilites[shuffle_indices],
        )
        # return target_names,target_feats,target_probabilites

    def __getitem__(self, index):
        text = torch.LongTensor(self.texts[index])
        seq_length = np.array(self.seq_lengths[index])
        viz_elem, info_elem = self.get_info(index)
        if "train" in self.mode:
            node_names, node_feats, target_probabilites = self.get_train_items(index)
        else:
            node_names, node_feats, anchor_index = self.get_test_items(index)
            target_probabilites = torch.zeros(self.max_nodes)
            target_probabilites[anchor_index] = 1

        return (
            text,
            seq_length,
            node_feats,
            target_probabilites,
            node_names,
            info_elem,
            viz_elem,
        )

    def __len__(self):
        return len(self.texts)
