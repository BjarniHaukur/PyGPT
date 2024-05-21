from typing import Literal
from functools import lru_cache
from pathlib import Path

from .tokenizer import BPETokenizer

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_PATH = Path("data/py150k")


class Py150kDataset(Dataset):
    """ Reads and tokenizes each file in the Py150k dataset. """
    def __init__(self, split:Literal["train","eval"], tokenizer_name:str):
        self.files = open(
            DATA_PATH / "python" / ("100k_train.txt" if split=="train" else "50k_eval.txt"),
            "r", encoding='utf-8'
        ).read().split("\n")[:-1] # last is empty line
        self.tokenizer = BPETokenizer.load(tokenizer_name)

    @lru_cache() # creates a dictionary behind the scenes which maps idx to the data, i.e. only tokenize once
    def __getitem__(self, idx:int):
        tokens = self.tokenizer.tokenize(open(DATA_PATH / self.files[idx], encoding="iso-8859-1").read())
        return torch.tensor(tokens)
    
    def __len__(self):
        return len(self.files)
    
class MemmapDataset(Dataset):
    """ Reads tokens from a memmap file. """
    def __init__(self, split:Literal["train","eval"], tokenizer_name:str, num_chars:int=4096):
        self.memmap = np.memmap(
            DATA_PATH / ("train.dat" if split=="train" else "eval.dat"),
            dtype="uint8", mode="r"
        )
        self.tokenizer = BPETokenizer.load(tokenizer_name)
        self.num_chars = num_chars
    
    @lru_cache()
    def __getitem__(self, idx:int):
        if idx < 0: idx += len(self)
        encoded = self.memmap[idx * self.num_chars: (idx + 1) * self.num_chars]
        text = encoded.tobytes().decode('iso-8859-1')
        text = text[text.find("\n"):text.rfind("\n")] # start after newline and end at newline, shouldnt drop too much data
        # remove from start until you reach a space, remove end until you reach a space
        tokens = self.tokenizer.tokenize(text)
        return torch.tensor(tokens)
        
    def __len__(self):
        return len(self.memmap) // self.num_chars
    

if __name__ == "__main__":
    # This script is used to create the memmap files for the MemmapDataset
    # inspired by Karpathy's nanoGPT
    
    files_train = open(DATA_PATH / "python100k_train.txt", "r").read().split("\n")[:-1] # last is empty line
    texts_train = [open(DATA_PATH / x, "r", encoding="iso-8859-1").read() for x in files_train]
        
    files_eval = open(DATA_PATH / "python50k_eval.txt", "r").read().split("\n")[:-1] # last is empty line
    texts_eval = [open(DATA_PATH / x, "r", encoding="iso-8859-1").read() for x in files_eval]

    chrs_train = "".join(texts_train).encode('iso-8859-1')
    chrs_eval = "".join(texts_eval).encode('iso-8859-1')
    
    memmap_train = np.memmap(DATA_PATH / "train.dat", mode="w+", dtype='uint8', shape=(len(chrs_train),))
    memmap_eval = np.memmap(DATA_PATH / "eval.dat", mode="w+", dtype='uint8', shape=(len(chrs_eval),))
    
    memmap_train[:] = np.array(list(chrs_train))
    memmap_eval[:] = np.array(list(chrs_eval))
    
    memmap_train.flush()
    memmap_eval.flush()
    




