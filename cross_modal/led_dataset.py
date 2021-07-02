from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import json
import random


class LEDDataset(Dataset):
    def __init__(
        self,
        mode,
        args,
        scan_names,
        episode_ids,
        viewpoints,
        pano_feats,
        texts,
        seq_lengths,
        dialogs,
    ):
        self.mode = mode
        self.args = args
        self.scan_names = scan_names
        self.episode_ids = episode_ids
        self.viewpoints = viewpoints
        self.pano_feats = pano_feats
        self.texts = texts
        self.seq_lengths = seq_lengths
        self.dialogs = dialogs
        self.min_length = 345
        self.node_levels = None
        ## if visualization
        self.node_levels = json.load(open(args.image_dir + "allScans_Node2pix.json"))

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

    def get_contrastive_items(self, index):
        scan_id = self.scan_names[index]
        level = self.node_levels[self.scan_names[index]][self.viewpoints[index]][-1]
        vp = self.viewpoints[index]
        neg_examples = []
        for node in sorted(self.args.scan_graphs[scan_id].nodes()):
            if node == vp:
                continue
            if int(self.node_levels[scan_id][node][-1]) != int(level):
                continue
            neg_examples.append(node)
        contrastive = random.choice(neg_examples)

        node_names = [vp, contrastive]
        key1 = scan_id + "-" + vp
        key2 = scan_id + "-" + contrastive
        node_feats = [
            torch.tensor(self.pano_feats[key1.encode()]),
            torch.tensor(self.pano_feats[key2.encode()]),
        ]
        node_feats = torch.stack(node_feats, dim=0)

        return node_feats, node_names

    def get_test_items(self, index):
        scan_id = self.scan_names[index]
        node_feats = torch.zeros(self.args.max_nodes, 36, 2048)
        node_names = []
        i = 0
        anchor_index = 0
        for node in sorted(self.args.scan_graphs[scan_id].nodes()):
            node_names.append(node)
            key = scan_id + "-" + node
            node_feats[i, :, :] = torch.tensor(self.pano_feats[key.encode()])
            if node == self.viewpoints[index]:
                anchor_index = i
            i += 1
        for _ in range(i, self.args.max_nodes):
            node_names.append("null")
        return node_feats, node_names, anchor_index

    def __getitem__(self, index):
        text = torch.LongTensor(self.texts[index])
        seq_length = np.array(self.seq_lengths[index])
        viz_elem, info_elem = self.get_info(index)

        if "train" in self.mode:
            node_feats, node_names = self.get_contrastive_items(index)
            anchor = torch.tensor([1, 0], dtype=float)
        else:
            node_feats, node_names, anchor_index = self.get_test_items(index)
            anchor = torch.zeros(self.args.max_nodes)
            anchor[anchor_index] = 1

        return (
            text,
            seq_length,
            node_feats,
            anchor,
            node_names,
            info_elem,
            viz_elem,
        )

    def __len__(self):
        return len(self.texts)
