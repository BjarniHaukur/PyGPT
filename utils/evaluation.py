from dataclasses import dataclass
import torch
from tqdm import tqdm
import numpy as np
import utils.metrics as metrics

@dataclass
class GenerationScores:
    bleu: float
    syntaxError: float

def evaluate_generation(model, val_dl, tokenizer, input_len=1000, output_len=10) -> GenerationScores:
    total_len = input_len + output_len

    bleu_scores = []
    gens = []

    with torch.no_grad():
        for batch in tqdm(val_dl):
            x = batch[:, :-input_len]
            y = batch[:, input_len:total_len]

            # placeholder for what is assumed to be generated
                # batch_size = x.size(0)
                # gen = torch.randint(0, 100, (batch_size, output_len))

            gen = model.generate(starting_tokens=x, max_length=output_len)

            gens.append(gen)
            y_hat = gen[:, -output_len:]

            for i in range(gen.size(0)):
                bleu_scores.append(metrics.bleu_score(y_hat[i].tolist(), y[i].tolist()))

    flattened_gens = [gen_seq for batch_gen in gens for gen_seq in batch_gen]
    programs = [tokenizer.detokenize(gen_seq.tolist()) for gen_seq in flattened_gens]
    
    syntax_error_score = metrics.syntax_error_score(programs)
    avg_bleu = np.mean(bleu_scores)

    return GenerationScores(bleu=avg_bleu, syntaxError=syntax_error_score)


