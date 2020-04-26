from os.path import join, exists
import numpy as np
from tqdm import tqdm
import torch
from bi_lstm_crf.app.preprocessing.utils import *

FILE_VOCAB = "vocab.json"
FILE_TAGS = "tags.json"
FILE_DATASET = "dataset.txt"
FILE_DATASET_CACHE = "dataset_cache_{}.npz"


class Preprocessor:
    def __init__(self, config_dir, save_config_dir=None, verbose=True):
        self.config_dir = config_dir
        self.verbose = verbose

        self.vocab, self.vocab_dict = self.__load_list_file(FILE_VOCAB, offset=1, verbose=verbose)
        self.tags, self.tags_dict = self.__load_list_file(FILE_TAGS, verbose=verbose)
        if save_config_dir:
            self.__save_config(save_config_dir)

        self.PAD_IDX = 0
        self.OOV_IDX = len(self.vocab)
        self.__adjust_vocab()

    def __load_list_file(self, file_name, offset=0, verbose=False):
        file_path = join(self.config_dir, file_name)
        if not exists(file_path):
            raise ValueError('"{}" file does not exist.'.format(file_path))
        else:
            elements = load_json_file(file_path)
            elements_dict = {w: idx + offset for idx, w in enumerate(elements)}
            if verbose:
                print("config {} loaded".format(file_path))
            return elements, elements_dict

    def __adjust_vocab(self):
        self.vocab.insert(0, PAD)
        self.vocab_dict[PAD] = 0

        self.vocab.append(OOV)
        self.vocab_dict[OOV] = len(self.vocab) - 1

    def __save_config(self, dst_dir):
        char_file = join(dst_dir, FILE_VOCAB)
        save_json_file(self.vocab, char_file)

        tag_file = join(dst_dir, FILE_TAGS)
        save_json_file(self.tags, tag_file)

        if self.verbose:
            print("tag dict file => {}".format(tag_file))
            print("tag dict file => {}".format(char_file))

    @staticmethod
    def __cache_file_path(corpus_dir, max_seq_len):
        return join(corpus_dir, FILE_DATASET_CACHE.format(max_seq_len))

    def load_dataset(self, corpus_dir, val_split, test_split, max_seq_len):
        """load the train set

        :return: (xs, ys)
            xs: [B, L]
            ys: [B, L, C]
        """
        ds_path = self.__cache_file_path(corpus_dir, max_seq_len)
        if not exists(ds_path):
            xs, ys = self.__build_corpus(corpus_dir, max_seq_len)
        else:
            print("loading dataset {} ...".format(ds_path))
            dataset = np.load(ds_path)
            xs, ys = dataset["xs"], dataset["ys"]

        xs, ys = map(
            torch.tensor, (xs, ys)
        )

        # split the dataset
        total_count = len(xs)
        assert total_count == len(ys)
        val_count = int(total_count * val_split)
        test_count = int(total_count * test_split)
        train_count = total_count - val_count - test_count
        assert train_count > 0 and val_count > 0

        indices = np.cumsum([0, train_count, val_count, test_count])
        datasets = [(xs[s:e], ys[s:e]) for s, e in zip(indices[:-1], indices[1:])]
        print("datasets loaded:")
        for (xs_, ys_), name in zip(datasets, ["train", "val", "test"]):
            print("\t{}: {}, {}".format(name, xs_.shape, ys_.shape))
        return datasets

    def decode_tags(self, batch_tags):
        batch_tags = [
            [self.tags[t] for t in tags]
            for tags in batch_tags
        ]
        return batch_tags

    def sent_to_vector(self, sentence, max_seq_len=0):
        max_seq_len = max_seq_len if max_seq_len > 0 else len(sentence)
        vec = [self.vocab_dict.get(c, self.OOV_IDX) for c in sentence[:max_seq_len]]
        return vec + [self.PAD_IDX] * (max_seq_len - len(vec))

    def tags_to_vector(self, tags, max_seq_len=0):
        max_seq_len = max_seq_len if max_seq_len > 0 else len(tags)
        vec = [self.tags_dict[c] for c in tags[:max_seq_len]]
        return vec + [0] * (max_seq_len - len(vec))

    def __build_corpus(self, corpus_dir, max_seq_len):
        file_path = join(corpus_dir, FILE_DATASET)
        xs, ys = [], []
        with open(file_path, encoding="utf8") as f:
            for idx, line in tqdm(enumerate(f), desc="parsing {}".format(file_path)):
                fields = line.strip().split("\t")
                if len(fields) != 2:
                    raise ValueError("format error in line {}, tabs count: {}".format(idx + 1, len(fields) - 1))

                sentence, tags = fields
                try:
                    if sentence[0] == "[":
                        sentence = json.loads(sentence)
                    tags = json.loads(tags)
                    xs.append(self.sent_to_vector(sentence, max_seq_len=max_seq_len))
                    ys.append(self.tags_to_vector(tags, max_seq_len=max_seq_len))
                    if len(sentence) != len(tags):
                        raise ValueError('"sentence length({})" != "tags length({})" in line {}"'.format(
                            len(sentence), len(tags), idx + 1))
                except Exception as e:
                    raise ValueError("exception raised when parsing line {}\n\t{}\n\t{}".format(idx + 1, line, e))

        xs, ys = np.asarray(xs), np.asarray(ys)

        # save train set
        cache_file = self.__cache_file_path(corpus_dir, max_seq_len)
        np.savez(cache_file, xs=xs, ys=ys)
        print("dataset cache({}, {}) => {}".format(xs.shape, ys.shape, cache_file))
        return xs, ys
