import json
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

CHECKPOINT_PATH = Path("checkpoints/tokenizer/")

BOS, BOS_ID = "<bos>", 0
EOS, EOS_ID = "<eos>", 1
PAD, PAD_ID = "<pad>", 2
UNK, UNK_ID = "<unk>", 3

def chunk_list(data, chunk_size):
    """Yield successive chunks of data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def count_pairs(ids_chunk):
    """Count pairs in a chunk of ids."""
    counts = Counter()
    for a, b in zip(ids_chunk[:-1], ids_chunk[1:]):
        counts[(a, b)] += 1
    return counts

def most_common_pair(ids:list[int], num_chunks:int) -> tuple[int, int]:
    chunk_size = len(ids) // num_chunks
    chunks = list(chunk_list(ids, chunk_size))
    
    with ThreadPoolExecutor() as executor:
        chunk_counts = list(executor.map(count_pairs, chunks))
    
    combined_counts = Counter()
    for count in chunk_counts:
        combined_counts.update(count)
    
    return combined_counts.most_common(1)[0][0]

def replace_pair(ids:list[int], pair:tuple[int, int]) -> list[int]:
    new_id = max(ids) + 1
    i, length = 0, len(ids)
    result = []
    while i < length:
        if i < length - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result

class BPETokenizer:
    def __init__(self, text:str = ""):
        self.__initialize_tokens(text)

    def __initialize_tokens(self, text:str):
        assert not hasattr(self, "chr_to_ids"), "Cannot override existing vocabulary"
        self.chr_to_ids = {BOS: BOS_ID, EOS: EOS_ID, PAD: PAD_ID, UNK: UNK_ID}

        for c in sorted(set(text)):
            self.chr_to_ids[c] = len(self.chr_to_ids)
        self.ids_to_chr = {i: c for c, i in self.chr_to_ids.items()}

    def __len__(self):
        return len(self.ids_to_chr)

    def __getitem__(self, idx:int):
        return self.ids_to_chr[idx]

    def save(self, filename:str):
        CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
        filepath = CHECKPOINT_PATH / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'chr_to_ids': self.chr_to_ids,
                'ids_to_chr': self.ids_to_chr
            }, f, indent=4)

    @classmethod
    def load(cls, filename:str):
        filepath = CHECKPOINT_PATH / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.chr_to_ids = {c: int(i) for c, i in data['chr_to_ids'].items()}
        tokenizer.ids_to_chr = {int(i): c for i, c in data['ids_to_chr'].items()}
        return tokenizer

    @classmethod
    def fit(cls, text:str, iterations:int, num_chunks:int = 4):
        tok = cls(text)
        ids = tok.tokenize(text)

        for _ in tqdm(range(iterations), desc="Fitting tokenizer ..."):
            pair = most_common_pair(ids, num_chunks)
            ids = replace_pair(ids, pair)
            new_id = len(tok)
            tok.ids_to_chr[new_id] = tok.ids_to_chr[pair[0]] + tok.ids_to_chr[pair[1]]

        tok.chr_to_ids = {c: i for i, c in tok.ids_to_chr.items()}
        return tok

    def tokenize(self, text:str) -> list[int]:
        tokens = []
        longest_token = max(len(x) for x in self.ids_to_chr.values())

        i = 0
        while i < len(text):
            token = ""
            token_idx = -1
            for j in range(i + 1, min(len(text) + 1, i + longest_token + 1)):
                substring = text[i:j]
                if substring in self.chr_to_ids:
                    token = substring
                    token_idx = j - i

            if token_idx == -1:
                token = UNK
                token_idx = 1

            tokens.append(self.chr_to_ids[token])
            i += token_idx

        return tokens

    def detokenize(self, tokens:list[int]) -> str:
        return "".join(self.ids_to_chr[token] for token in tokens)

    def color_text_ansi(self, text:str) -> str:
        tokens = self.tokenize(text)
        colored = ""
        for i, token in enumerate(tokens):
            color = COLORS[i % len(COLORS)]
            colored += ansi_color(color) + self.ids_to_chr[token]
        return colored

    def color_text_html(self, text:str) -> str:
        tokens = self.tokenize(text)
        colored = ""
        for i, token in enumerate(tokens):
            color = COLORS[i % len(COLORS)]
            token_text = self.ids_to_chr[token]
            token_text = token_text.replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
            colored += f'<span {html_color(color)}>{token_text}</span>'
        return colored

RESET_BG = '\x1b[0m'
COLORS = [
    (194, 224, 255),
    (255, 218, 194),
    (194, 255, 208),
    (255, 194, 224),
    (218, 255, 194),
]

def ansi_color(rgb:tuple[int, int, int]) -> str:
    return f'\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m'

def html_color(rgb:tuple[int, int, int]) -> str:
    return f'style="background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"'

if __name__ == "__main__":
    import random
    import argparse

    random.seed(1337)

    parser = argparse.ArgumentParser(description="Fit a Byte-Pair Encoding tokenizer")
    parser.add_argument("--num_train_files", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--tokenizer_name", type=str, default="my_tokenizer")
    args = parser.parse_args()

    train_files = open("data/py150k/python100k_train.txt", "r", encoding="utf-8").read().split("\n")[:-1]
    train_texts = [
        open("data/py150k/" + path, encoding="iso-8859-1").read()
        for path in random.sample(train_files, args.num_train_files)
    ]

    tok = BPETokenizer.fit("".join(train_texts), args.iterations, args.num_chunks)
    tok.save(args.tokenizer_name)
