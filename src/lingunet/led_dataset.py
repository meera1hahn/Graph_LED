from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import torch
import numpy as np
from PIL import Image
import copy
import json


class LEDDataset(Dataset):
    def __init__(
        self,
        mode,
        args,
        texts,
        seq_lengths,
        mesh_conversions,
        locations,
        viewPoint_location,
        dialogs,
        scan_names,
        levels,
        annotation_ids,
    ):
        self.mode = mode
        self.args = args
        self.texts = texts
        self.seq_lengths = seq_lengths
        self.mesh_conversions = mesh_conversions
        self.locations = locations
        self.viewPoint_location = viewPoint_location
        self.dialogs = dialogs
        self.scan_names = scan_names
        self.levels = levels
        self.annotation_ids = annotation_ids
        self.mesh2meters = json.load(open(args.image_dir + "pix2meshDistance.json"))
        self.image_size = [
            3,
            int(700 * self.args.ds_percent),
            int(1200 * self.args.ds_percent),
        ]

        self.preprocess_data_aug = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5, hue=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.555],
                    std=[0.229, 0.224, 0.225, 0.222],
                ),
            ]
        )
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.555],
                    std=[0.229, 0.224, 0.225, 0.222],
                ),
            ]
        )

    def gather_all_floors(self, index):
        all_maps = torch.zeros(
            self.args.max_floors,
            self.image_size[0],
            self.image_size[1],
            self.image_size[2],
        )
        all_conversions = torch.zeros(self.args.max_floors, 1)
        sn = self.scan_names[index]
        floors = self.mesh2meters[sn].keys()
        for enum, f in enumerate(floors):
            img = Image.open(
                "{}floor_{}/{}_{}.png".format(self.args.image_dir, f, sn, f)
            )
            img = img.resize((self.image_size[2], self.image_size[1]))
            if "train" in self.mode:
                all_maps[enum, :, :, :] = self.preprocess_data_aug(img)[:3, :, :]
            else:
                all_maps[enum, :, :, :] = self.preprocess(img)[:3, :, :]
            all_conversions[enum, :] = self.mesh2meters[sn][f]["threeMeterRadius"] / 3.0
        return all_maps, all_conversions

    def get_info(self, index):
        info_elem = [
            self.dialogs[index],
            self.levels[index],
            self.scan_names[index],
            self.annotation_ids[index],
            self.viewPoint_location[index],
        ]
        return info_elem

    def create_target(self, index, location, mesh_conversion):
        gaussian_target = np.zeros(
            (self.args.max_floors, self.image_size[1], self.image_size[2])
        )
        gaussian_target[int(self.levels[index]), location[0], location[1]] = 1
        gaussian_target[int(self.levels[index]), :, :] = gaussian_filter(
            gaussian_target[int(self.levels[index]), :, :],
            sigma=mesh_conversion,
        )
        gaussian_target[int(self.levels[index]), :, :] = (
            gaussian_target[int(self.levels[index]), :, :]
            / gaussian_target[int(self.levels[index]), :, :].sum()
        )
        gaussian_target = torch.tensor(gaussian_target)
        return gaussian_target

    def __getitem__(self, index):
        location = copy.deepcopy(self.locations[index])
        location = np.round(np.asarray(location) * self.args.ds_percent).astype(int)
        mesh_conversion = self.mesh_conversions[index] * self.args.ds_percent
        text = torch.LongTensor(self.texts[index])
        seq_length = np.array(self.seq_lengths[index])
        maps, conversions = self.gather_all_floors(index)
        target = self.create_target(index, location, mesh_conversion)
        info_elem = self.get_info(index)

        return (
            info_elem,
            text,
            seq_length,
            target,
            maps,
            conversions,
        )

    def __len__(self):
        return len(self.annotation_ids)
