import json

from tqdm import tqdm
from pathlib import Path

CHECKPOINT_PATH = Path("checkpoints/tokenizer/")

    
def most_common_pair(ids:list[int])->tuple[int,int]:
    counts = {}
    for a, b in zip(ids[:-1], ids[1:]):
        counts[(a,b)] = counts.get((a,b), 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]

def replace_pair(ids:list[int], pair:tuple[int,int])->list[int]:
    new_id = max(ids) + 1
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(new_id)
            i += 2 # we accounted for two tokens
        else:
            new_ids.append(ids[i])
            i += 1
    
    return new_ids

class BPETokenizer:
    def __init__(self):
        self.chr_to_ids = {}
        self.ids_to_chr = {}

    def __len__(self): return len(self.chr_to_ids)

    def save(self, filename: str):
        CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
        filepath = CHECKPOINT_PATH / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'chr_to_ids': self.chr_to_ids,
                'ids_to_chr': self.ids_to_chr
            }, f, indent=4)

    @classmethod
    def load(cls, filename: str):
        filepath = CHECKPOINT_PATH / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.chr_to_ids = {c:int(i) for c,i in data['chr_to_ids'].items()}
        tokenizer.ids_to_chr = {int(i):c for i,c in data['ids_to_chr'].items()}
        return tokenizer
    
    @classmethod
    def fit(cls, text:str, iterations:int):
        tok = cls()
        tok.ids_to_chr = {i:c for i,c in enumerate(set(text))}
        tok.chr_to_ids = {c:i for i,c in tok.ids_to_chr.items()}

        ids = tok.tokenize(text)
        for _ in tqdm(range(iterations), desc="Fitting tokenizer ..."):
            pair = most_common_pair(ids)
            ids = replace_pair(ids, pair)
            new_id = max(tok.ids_to_chr) + 1
            tok.ids_to_chr[new_id] = tok.ids_to_chr[pair[0]] + tok.ids_to_chr[pair[1]]

        tok.chr_to_ids = {c:i for i,c in tok.ids_to_chr.items()} # ugly I know
        return tok

    def tokenize(self, text:str)->list[int]:
        tokens = []
        longest_token = max(len(x) for x in self.ids_to_chr.values())

        i = 0
        while i < len(text): # find the longest substring included in the vocabulary
            token = ""
            token_idx = -1
            for j in range(i+1, min(len(text)+1, i+longest_token+1)):
                substring = text[i:j]
                if substring in self.chr_to_ids:
                    token = substring
                    token_idx = j - i
                
            if token_idx == -1: raise RuntimeError() # should always find some token unless fit on data that does not have a character being tokenized

            tokens.append(self.chr_to_ids[token])
            i += token_idx

        return tokens

    def print_tokens(self, text):
        tokens = self.tokenize(text)
        for i, token in enumerate(tokens):
            color_print(self.ids_to_chr[token], i)


RESET_BG = '\x1b[0m'  # ANSI code to reset background color
COLORS = [
    (194, 224, 255),  # light blue
    (255, 218, 194),  # light orange
    (194, 255, 208),  # light green
    (255, 194, 224),  # light pink
    (218, 255, 194),  # light lime
]

def bg_color(rgb:tuple[int, int, int]) -> str:
    """Return ANSI escape code for a custom RGB background color."""
    return f'\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m'

def color_print(text:str, color_idx:int):
     color = COLORS[color_idx % len(COLORS)]
     print(bg_color(color) + text, end='')