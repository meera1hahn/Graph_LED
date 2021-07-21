import numpy as np
import json
from nltk.tokenize import word_tokenize
import copy
import re
from src.lingunet.led_dataset import LEDDataset


class Loader:
    def __init__(self, args):
        self.mesh2meters = json.load(open(args.image_dir + "pix2meshDistance.json"))
        self.vocab = Vocabulary()
        self.max_length = 0
        self.max_dialog_length = 0
        self.datasets = {}
        self.args = args

    def load_image_paths(self, data, mode):
        episode_ids, scan_names, levels, mesh_conversions, dialogs = [], [], [], [], []
        for data_obj in data:
            episode_ids.append(data_obj["episodeId"])
            scan_names.append(data_obj["scanName"])
            dialogs.append(self.add_tokens(data_obj["dialogArray"]))
            level = 0
            if mode != "test":
                level = str(data_obj["finalLocation"]["floor"])
            levels.append(level)
            mesh_conversions.append(
                self.mesh2meters[data_obj["scanName"]][str(level)]["threeMeterRadius"]
                / 3.0
            )
        return episode_ids, scan_names, levels, mesh_conversions, dialogs

    def add_tokens(self, message_arr):
        new_dialog = ""
        for enum, message in enumerate(message_arr):
            if enum % 2 == 0:
                new_dialog += "SOLM " + message + " EOLM "
            else:
                new_dialog += "SOOM " + message + " EOOM "
        return new_dialog

    def load_locations(self, data, mode):
        if "test" in mode:
            return [[0, 0] for _ in data], ["" for _ in data]

        x = [
            [
                data_obj["finalLocation"]["pixel_coord"][1],
                data_obj["finalLocation"]["pixel_coord"][0],
            ]
            for data_obj in data
        ]

        y = [data_obj["finalLocation"]["viewPoint"] for data_obj in data]

        return x, y

    def build_pretrained_vocab(self, texts):
        self.vocab.word2idx = json.load(open(self.args.embedding_dir + "word2idx.json"))
        self.vocab.idx2word = json.load(open(self.args.embedding_dir + "idx2word.json"))
        ids = []
        seq_lengths = []
        for text in texts:
            text = re.sub(r"\.\.+", ". ", text)
            line_ids = []
            words = word_tokenize(text.lower())
            self.max_length = max(self.max_length, len(words))
            for word in words:
                line_ids.append(self.vocab.word2idx[word])
            ids.append(line_ids)
            seq_lengths.append(len(words))
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        return text_ids, seq_lengths

    def build_dataset(self, file):
        mode = file.split("_")[0]
        print("[{}]: Loading JSON file...".format(mode))
        data = json.load(open(self.args.data_dir + file))
        print("[{}]: Using {} samples".format(mode, len(data)))
        locations, viewPoint_location = self.load_locations(data, mode)
        (
            episode_ids,
            scan_names,
            levels,
            mesh_conversions,
            dialogs,
        ) = self.load_image_paths(data, mode)
        texts = copy.deepcopy(dialogs)
        texts, seq_lengths = self.build_pretrained_vocab(texts)

        print("[{}]: Building dataset...".format(mode))
        dataset = LEDDataset(
            mode,
            self.args,
            texts,
            seq_lengths,
            mesh_conversions,
            locations,
            viewPoint_location,
            dialogs,
            scan_names,
            levels,
            episode_ids,
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}

    def add_word(self, word, mode):
        if word not in self.word2idx and mode in ("train"):
            idx = len(self.idx2word)
            self.idx2word[idx] = word
            self.word2idx[word] = idx
            return word
        elif word not in self.word2idx and mode != "train":
            return "<unk>"
        else:
            return word

    def __len__(self):
        return len(self.idx2word)
