import torch
import torch.nn.functional as F

def sample_with_temp(logits:torch.Tensor, temperature:float=1.0)->torch.Tensor:
    p = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(p, 1)

def nucleus_sample(logits:torch.Tensor, nucleus_threshold:float=0.9)->torch.Tensor:
    p = F.softmax(logits, dim=-1).squeeze()
    sorted_probs, sorted_indices = torch.sort(p, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = torch.searchsorted(cumulative_probs, nucleus_threshold).item()
    top_indices = sorted_indices[:cutoff + 1]
    top_probs = sorted_probs[:cutoff + 1]
    top_probs /= top_probs.sum()

    sampled_index = torch.multinomial(top_probs, 1)
    return top_indices[sampled_index].unsqueeze(0)

def nucleus_sample_with_temp(logits:torch.Tensor, nucleus_threshold:float=0.9, temperature:float=1.0)->torch.Tensor:
    return nucleus_sample(logits / temperature, nucleus_threshold)

def top_k_sample(logits:torch.Tensor, k:int=50)->torch.Tensor:
    p = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(p, k)
    top_probs /= top_probs.sum()
    sampled_index = torch.multinomial(top_probs, 1)
    return top_indices[sampled_index]

