import numpy as np
import json
from nltk.tokenize import word_tokenize
import numpy as np
import copy
import re
from led_dataset import LEDDataset
import base64
import csv
import sys
import numpy as np
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

_TSV_FIELDNAMES = ["scanId", "viewpointId", "image_w", "image_h", "vfov", "features"]


class Loader:
    def __init__(self, data_dir, args):
        self.data_dir = data_dir
        self.datasets = {}
        self.max_length = 0
        self.args = args
        self.pano_feats, self.keys = self.load_pano_feats(args)

    def _convert_item(self, item):
        item["image_w"] = int(item["image_w"])  # pixels
        item["image_h"] = int(item["image_h"])  # pixels
        item["vfov"] = int(item["vfov"])  # degrees
        item["features"] = np.frombuffer(
            base64.b64decode(item["features"]), dtype=np.float32
        ).reshape(
            (-1, 2048)
        )  # 36 x 2048 region features
        return item

    def load_pano_feats(self, args):
        keys = []
        data = {}
        path = "/srv/flash1/mhahn30/LED/ResNet-152-imagenet.tsv"
        with open(path, "rt") as fid:
            reader = csv.DictReader(fid, delimiter="\t", fieldnames=_TSV_FIELDNAMES)
            for item in tqdm(reader):
                item = self._convert_item(item)
                key = item["scanId"] + "-" + item["viewpointId"]
                keys.append(key.encode())
                data[key.encode()] = item["features"]
        return data, keys

    def load_info(self, data, mode):
        scan_names = []
        episode_ids = []
        viewpoints = []
        dialogs = []
        for data_obj in data:
            scan_names.append(data_obj["scanName"])
            episode_ids.append(data_obj["annotationId"])
            if "test" in mode:
                viewpoints.append(data_obj["finalLocation"]["viewPoint"])
            else:
                viewpoints.append(data_obj["finalLocation"]["viewPoint"])
            dialogs.append(self.add_tokens(data_obj["dialogArray"]))
        return scan_names, episode_ids, viewpoints, dialogs

    def add_tokens(self, message_arr):
        new_dialog = ""
        for enum, message in enumerate(message_arr):
            if enum % 2 == 0:
                new_dialog += "SOLM " + message + " EOLM "
            else:
                new_dialog += "SOOM " + message + " EOOM "
        return new_dialog

    def build_pretrained_vocab(self, texts):
        word2idx = json.load(open(self.args.embedding_dir + "word2idx.json"))
        ids = []
        seq_lengths = []
        for text in texts:
            text = re.sub(r"\.\.+", ". ", text)
            line_ids = []
            words = word_tokenize(text.lower())
            self.max_length = max(self.max_length, len(words))
            for word in words:
                line_ids.append(word2idx[word])
            ids.append(line_ids)
            seq_lengths.append(len(words))
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        return text_ids, seq_lengths

    def build_dataset(self, file):
        mode = file.split("_")[0]
        print("[{}]: Loading JSON file...".format(mode))
        data = json.load(open(self.data_dir + file))

        if isinstance(self.args.sample_used, tuple):
            start, end = self.args.sample_used
            data = data[start:end]
            num_samples = end - start
        elif isinstance(self.args.sample_used, float):
            num_samples = int(len(data) * self.args.sample_used)
            data = data[:num_samples]
        print(
            "[{}]: Using {} ({}%) samples".format(
                mode, num_samples, num_samples / len(data) * 100
            )
        )

        scan_names, episode_ids, viewpoints, dialogs = self.load_info(data, mode)
        texts = copy.deepcopy(dialogs)
        texts, seq_lengths = self.build_pretrained_vocab(texts)

        print("[{}]: Building dataset...".format(mode))
        dataset = LEDDataset(
            mode,
            self.args,
            scan_names,
            episode_ids,
            viewpoints,
            self.pano_feats,
            texts,
            seq_lengths,
            dialogs,
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))
