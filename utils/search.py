import torch
import torch.nn.functional as F

from tokenizer import BOS_ID, EOS_ID

def beam_search(model, beam_width:int=3, max_length:int=50)->list[int]:
    sequences = [[BOS_ID]]
    scores = [0]
    
    for _ in range(max_length):
        all_candidates = []
        for i in range(len(sequences)):
            seq = sequences[i]
            x = torch.tensor(seq).unsqueeze(0)
            logits = model(x)
            p = F.log_softmax(logits[:, -1, :], dim=-1)
            
            top_probs, top_indices = torch.topk(p, beam_width)
            for j in range(beam_width):
                candidate = seq + [top_indices[0, j].item()]
                score = scores[i] + top_probs[0, j].item()
                all_candidates.append((candidate, score))
                
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences, scores = zip(*ordered[:beam_width])

        if sequences[0][-1] == EOS_ID:
            break
        
    return sequences[0]