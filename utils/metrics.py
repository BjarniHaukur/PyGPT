import ast
import math
from collections import defaultdict

import numpy as np

# def bleu_score(predicted_tokens:list[int], label_tokens:list[int], n_gram:int=4):
#     len_pred, len_label = len(predicted_tokens), len(label_tokens)
#     score = math.exp(min(0, 1 - len_label / len_pred))
#     for n in range(1, min(n_gram, len_pred) + 1):
#         num_matches, label_subs = 0, defaultdict(int)
#         for i in range(len_label - n + 1):
#             label_subs[' '.join(label_tokens[i: i + n])] += 1
#         for i in range(len_pred - n + 1):
#             if label_subs[' '.join(predicted_tokens[i: i + n])] > 0:
#                 num_matches += 1
#                 label_subs[' '.join(predicted_tokens[i: i + n])] -= 1
#         score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
#     return score

def bleu_score(predicted_tokens: list[int], label_tokens: list[int], n_gram: int = 4):
    len_pred, len_label = len(predicted_tokens), len(label_tokens)
    if len_pred == 0 or len_label == 0:
        return 0.0  # Handle edge cases where length is zero

    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(n_gram, len_pred) + 1):
        num_matches, label_subs = 0, defaultdict(int)
        
        # Create n-grams for label tokens
        for i in range(len_label - n + 1):
            label_subs[tuple(label_tokens[i: i + n])] += 1
        
        # Count matches with predicted tokens
        for i in range(len_pred - n + 1):
            n_gram_tuple = tuple(predicted_tokens[i: i + n])
            if label_subs[n_gram_tuple] > 0:
                num_matches += 1
                label_subs[n_gram_tuple] -= 1
        
        if len_pred - n + 1 == 0:
            precision = 0.0
        else:
            precision = num_matches / (len_pred - n + 1)
        
        score *= math.pow(precision, math.pow(0.5, n))
    
    return score
    
def syntax_error_score(programs:list[str]):
    has_syntax_error = []
    for program in programs:
        try: ast.parse(program)
        except SyntaxError: has_syntax_error.append(True)
        else: has_syntax_error.append(False)
    return np.mean(has_syntax_error)


if __name__ == "__main__":
    programs = [
        "def foo(x): return x + 1",
        "def foo(x) return x + 1",
        "de foo(x): return x + 1",
        str(open(__file__).read())
    ]
    print(f"The programs listed have a score of {syntax_error_score(programs):.2%}, lower is better")
    pred = [0, 1, 1, 0]
    label = [0, 1, 1, 1]
                       
    bleu = bleu_score(pred, label, 2)
    
    print(bleu)
    