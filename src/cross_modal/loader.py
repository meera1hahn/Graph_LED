import numpy as np
import json
from nltk.tokenize import word_tokenize
import numpy as np
import copy
import re
import csv
import sys
import numpy as np

csv.field_size_limit(sys.maxsize)
from src.cross_modal.led_dataset import LEDDataset


class Loader:
    def __init__(self, data_dir, args):
        self.data_dir = data_dir
        self.datasets = {}
        self.max_length = 0
        self.args = args

    def load_info(self, data, mode):
        scan_names = []
        episode_ids = []
        viewpoints = []
        dialogs = []
        for data_obj in data:
            scan_names.append(data_obj["scanName"])
            episode_ids.append(data_obj["episodeId"])
            if "test" in mode:
                viewpoints.append("")
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

        num_samples = int(len(data))
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
            texts,
            seq_lengths,
            dialogs,
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))
