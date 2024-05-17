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
    BOS, BOS_ID = "<bos>", 0
    EOS, EOS_ID = "<eos>", 1
    PAD, PAD_ID = "<pad>", 2
    UNK, UNK_ID = "<unk>", 3
    
    def __init__(self, text:str=""):
        self.__initialize_tokens(text)
        
    def __initialize_tokens(self, text:str):
        assert not hasattr(self, "chr_to_ids"), "Cannot override existing vocabulary"
        self.chr_to_ids = {self.BOS:0, self.EOS:1, self.PAD:2, self.UNK:3}
        
        for c in sorted(set(text)): self.chr_to_ids[c] = len(self.chr_to_ids)
        self.ids_to_chr = {i:c for c,i in self.chr_to_ids.items()}
        
    def __len__(self): return len(self.chr_to_ids) 
    def __getitem__(self, idx:int): return self.ids_to_chr[idx]

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
        tok = cls(text)
        
        ids = tok.tokenize(text)
        for _ in tqdm(range(iterations), desc="Fitting tokenizer ..."):
            pair = most_common_pair(ids)
            ids = replace_pair(ids, pair)
            new_id = len(tok)
            tok.ids_to_chr[new_id] = tok.ids_to_chr[pair[0]] + tok.ids_to_chr[pair[1]]

        tok.chr_to_ids = {c:i for i,c in tok.ids_to_chr.items()} # ugly I know
        return tok

    def tokenize(self, text:str)->list[int]:
        tokens = []
        longest_token = max(len(x) for x in self.ids_to_chr.values())

        i = 0
        while i < len(text): # find each longest substring included in the vocabulary
            token = ""
            token_idx = -1
            for j in range(i+1, min(len(text)+1, i+longest_token+1)):
                substring = text[i:j]
                if substring in self.chr_to_ids:
                    token = substring
                    token_idx = j - i
                else:
                    break # as soon as a substring is not in the vocabulary we stop
                
            if token_idx == -1:
                token = self.UNK
                token_idx = 1
                
            tokens.append(self.chr_to_ids[token])
            i += token_idx

        return tokens
    
    def detokenize(self, tokens:list[int])->str:
        return "".join(self.ids_to_chr[token] for token in tokens)

    def color_text_ansi(self, text:str)->str:
        tokens = self.tokenize(text)
        colored = ""
        for i, token in enumerate(tokens):
            color = COLORS[i % len(COLORS)]
            colored += ansi_color(color) + self.ids_to_chr[token]
        return colored
    
    def color_text_html(self, text:str)->str:
        tokens = self.tokenize(text)
        colored = ""
        for i, token in enumerate(tokens):
            color = COLORS[i % len(COLORS)]
            token_text = self.ids_to_chr[token]
            token_text = token_text.replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
            colored += f'<span {html_color(color)}>{token_text}</span>'
        return colored


RESET_BG = '\x1b[0m'  # ANSI code to reset background color
COLORS = [
    (194, 224, 255),  # light blue
    (255, 218, 194),  # light orange
    (194, 255, 208),  # light green
    (255, 194, 224),  # light pink
    (218, 255, 194),  # light lime
]

def ansi_color(rgb:tuple[int, int, int]) -> str:
    """Return ANSI escape code for a custom RGB background color."""
    return f'\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m'

def html_color(rgb:tuple[int, int, int]) -> str:
    """Return HTML style for a custom RGB text color."""
    return f'style="color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});"'


if __name__ == "__main__":
    import random
    random.seed(1337) # do not change

    # We use ISO-8859-1 instead of UTF-8 because it encodes each character as a single byte. In contrast, UTF-8’s variable-length encoding significantly increase the vocabulary size.
    train_files = open("data/PY150K/python100k_train.txt", "r", encoding="utf-8").read().split("\n")[:-1] # remove the last empty line
    train_texts = [open("data/PY150K/" + path, encoding="iso-8859-1").read() for path in train_files]

    # Our starting vocabulary size is around 130, tokens which do not appear here are all considered <unk>
    tok = BPETokenizer.fit("".join(random.sample(train_texts, 100)), 100) # after fitting we should have 130 + number_of_iterations tokens
    # tok = BPETokenizer.load("py150k")
    tok.save("my_tokenizer")
