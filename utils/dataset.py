from .tokenizer import BPETokenizer

from typing import Literal
from functools import lru_cache

import torch
from torch.utils.data import Dataset


class PY150kDataset(Dataset):
    def __init__(self, split:Literal["train","eval"], tokenizer_name:str):
        self.files = open("data/PY150K/python" + "100k_train.txt" if split=="train" else "50k_eval.txt", "r",encoding='utf-8').read().split("\n")
        self.tokenizer = BPETokenizer.load(tokenizer_name)

    @lru_cache() # creates a dictionary behind the scenes which maps idx to the data, i.e. only tokenize once
    def __getitem__(self, idx:int):
        tokens = self.tokenizer.tokenize(open("data/PY150K/" + self.files[idx], encoding='iso-8859-1').read())
        return torch.tensor(tokens)
    
    def __len__(self): return len(self.files)

    


    




