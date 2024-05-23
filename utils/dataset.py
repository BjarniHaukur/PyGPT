from typing import Literal
from functools import lru_cache
from pathlib import Path

from .tokenizer import BPETokenizer

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

DATA_PATH = Path("data/py150k")


class Py150kDataset(Dataset):
    """ Reads and tokenizes each file in the Py150k dataset. """
    def __init__(self, split: Literal["train", "eval"], tokenizer_name: str):
        self.files = open(
            DATA_PATH / ("python100k_train.txt" if split == "train" else "50k_eval.txt"),
            "r", encoding='utf-8'
        ).read().split("\n")[:-1]  # last is empty line
        self.tokenizer = BPETokenizer.load(tokenizer_name)

    @lru_cache()  # creates a dictionary behind the scenes which maps idx to the data, i.e. only tokenize once
    def __getitem__(self, idx: int):
        tokens = self.tokenizer.tokenize(open(DATA_PATH / self.files[idx], encoding="iso-8859-1").read())
        return torch.tensor(tokens)

    def __len__(self):
        return len(self.files)


class MemmapDataset(Dataset):
    """ Reads tokens from a memmap file. """
    def __init__(self, memmap_name:str, tokenizer_name: str, num_tokens: int = 4096):
        self.memmap = np.memmap(
            DATA_PATH / (memmap_name + ".dat"),
            dtype="uint16", mode="r"
        )
        self.tokenizer = BPETokenizer.load(tokenizer_name)
        self.num_tokens = num_tokens

    @lru_cache()
    def __getitem__(self, idx: int):
        if idx < 0: idx += len(self)
        tokens = self.memmap[idx * self.num_tokens: (idx + 1) * self.num_tokens]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.memmap) // self.num_tokens


if __name__ == "__main__":
    from argparse import ArgumentParser
    from concurrent.futures import ProcessPoolExecutor
    
    parser = ArgumentParser(description="Creating a memmap from the Py150k dataset and a given tokenizer")
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("memmap_name", type=str)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer_name)

    def read_file(file_path):
        with open(file_path, "r", encoding="iso-8859-1") as file:
            return file.read()

    def tokenize_text(text):
        return tokenizer.tokenize(text)

    def process_files(files):
        with ProcessPoolExecutor() as executor:
            texts = list(tqdm(executor.map(read_file, [DATA_PATH / x for x in files]), desc="Reading data...", total=len(files)))
            tokens = list(tqdm(executor.map(tokenize_text, texts), desc="Tokenizing data...", total=len(texts)))
        return tokens
    


    files_train = open(DATA_PATH / "python100k_train.txt", "r").read().split("\n")[:-1]  # last is empty line
    tokens_train = np.concatenate(process_files(files_train))
    memmap_train = np.memmap(DATA_PATH / (args.memmap_name + "_train.dat"), mode="w+", dtype='uint16', shape=(len(tokens_train),))
    memmap_train[:] = tokens_train
    memmap_train.flush()

    del files_train, tokens_train

    files_eval = open(DATA_PATH / "python50k_eval.txt", "r").read().split("\n")[:-1]  # last is empty line
    tokens_eval = np.concatenate(process_files(files_eval))
    memmap_eval = np.memmap(DATA_PATH / (args.memmap_name + "_val.dat"), mode="w+", dtype='uint16', shape=(len(tokens_eval),))
    memmap_eval[:] = tokens_eval
    memmap_eval.flush()
