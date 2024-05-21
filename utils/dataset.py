from .tokenizer import BPETokenizer

from typing import Literal
from functools import lru_cache

import numpy as np

import torch
from torch.utils.data import Dataset


class Py150kDataset(Dataset):
    """ Reads and tokenizes each file in the PY150k dataset. """
    def __init__(self, split:Literal["train","eval"], tokenizer_name:str):
        self.files = open("data/PY150k/python" + ("100k_train.txt" if split=="train" else "50k_eval.txt"), "r", encoding='utf-8').read().split("\n")[:-1] # last is empty line
        self.tokenizer = BPETokenizer.load(tokenizer_name)

    @lru_cache() # creates a dictionary behind the scenes which maps idx to the data, i.e. only tokenize once
    def __getitem__(self, idx:int):
        tokens = self.tokenizer.tokenize(open("data/PY150k/" + self.files[idx], encoding="iso-8859-1").read())
        return torch.tensor(tokens)
    
    def __len__(self):
        return len(self.files)
    
    
class MemmapDataset(Dataset):
    """ Reads tokens from a memmap file. """
    def __init__(self, split:Literal["train","eval"], tokenizer_name:str, num_chars:int=4096):
        self.memmap = np.memmap("data/PY150k/" + ("train.dat" if split=="train" else "eval.dat"), dtype="uint8", mode="r")
        self.tokenizer = BPETokenizer.load(tokenizer_name)
        self.num_chars = num_chars
    
    @lru_cache()
    def __getitem__(self, idx:int):
        if idx < 0: idx += len(self)
        encoded = self.memmap[idx * self.num_chars: (idx + 1) * self.num_chars]
        text = encoded.tobytes().decode('iso-8859-1')
        tokens = self.tokenizer.tokenize(text)
        return torch.tensor(tokens)
        
    def __len__(self):
        return len(self.memmap) // self.num_chars
    

if __name__ == "__main__":
    # This script is used to create the memmap files for the MemmapDataset
    # inspired by Karpathy's nanoGPT
    import numpy as np
    
    files_train = open("data/PY150K/python100k_train.txt", "r").read().split("\n")[:-1] # last is empty line
    texts_train = [open("data/PY150K/" + x, "r", encoding="iso-8859-1").read() for x in files_eval]
        
    files_eval = open("data/PY150K/python50k_eval.txt", "r").read().split("\n")[:-1] # last is empty line
    texts_eval = [open("data/PY150K/" + x, "r", encoding="iso-8859-1").read() for x in files_eval]

    chrs_train = "".join(texts_train).encode('iso-8859-1')
    chrs_eval = "".join(texts_eval).encode('iso-8859-1')
    
    memmap_train = np.memmap("data/PY150k/train.dat", mode="w+", dtype='uint8', shape=(len(chrs_train),))
    memmap_eval = np.memmap("data/PY150k/eval.dat", mode="w+", dtype='uint8', shape=(len(chrs_eval),))
    
    memmap_train[:] = np.array(list(chrs_train))
    memmap_eval[:] = np.array(list(chrs_eval))
    
    memmap_train.flush()
    memmap_eval.flush()
    




