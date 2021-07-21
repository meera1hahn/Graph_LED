import numpy as np
import json
from nltk.tokenize import word_tokenize
import copy
import re
from src.lingunet.led_dataset import LEDDataset


class Loader:
    def __init__(self, args, data_dir, image_dir):
        self.mesh2meters = json.load(open(args.image_dir + "pix2meshDistance.json"))
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.vocab = Vocabulary()
        self.max_length = 0
        self.max_dialog_length = 0
        self.datasets = {}
        self.args = args

    def load_image_paths(self, data):
        image_paths = []
        scan_names = []
        levels = []
        annotation_ids = []
        for data_obj in data:
            scan_name = data_obj["scanName"]
            scan_names.append(scan_name)
            annotation_ids.append(data_obj["episodeId"])
            level = str(data_obj["finalLocation"]["floor"])
            levels.append(level)

            image_paths.append(
                "{}floor_{}/{}_{}.png".format(self.image_dir, level, scan_name, level)
            )
        return image_paths, scan_names, levels, annotation_ids

    def add_tokens(self, message_arr):
        new_dialog = ""
        for enum, message in enumerate(message_arr):
            if enum % 2 == 0:
                new_dialog += "SOLM " + message + " EOLM "
            else:
                new_dialog += "SOOM " + message + " EOOM "
        return new_dialog

    def load_dialogs(self, data):
        dialogs = []
        for data_obj in data:
            dialogs.append(self.add_tokens(data_obj["dialogArray"]))
        return dialogs

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

    def load_mesh_conversion(self, data):
        mesh_conversions = []
        for data_obj in data:
            mesh_conversions.append(
                self.mesh2meters[data_obj["scanName"]][
                    str(data_obj["finalLocation"]["floor"])
                ]["threeMeterRadius"]
                / 3.0
            )
        return mesh_conversions

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
        data = json.load(open(self.data_dir + file))
        print("[{}]: Using {} samples".format(mode, len(data)))
        locations, viewPoint_location = self.load_locations(data, mode)
        mesh_conversions = self.load_mesh_conversion(data)
        image_paths, scan_names, levels, annotation_ids = self.load_image_paths(data)
        dialogs = self.load_dialogs(data)
        texts = copy.deepcopy(dialogs)
        texts, seq_lengths = self.build_pretrained_vocab(texts)

        print("[{}]: Building dataset...".format(mode))
        dataset = LEDDataset(
            mode,
            self.args,
            image_paths,
            texts,
            seq_lengths,
            mesh_conversions,
            locations,
            viewPoint_location,
            dialogs,
            scan_names,
            levels,
            annotation_ids,
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
