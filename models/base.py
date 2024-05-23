from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

CONFIG_PATH = Path("configs/")

@dataclass
class ModelConfig(ABC):
    """Abstract base class for model configurations."""
    wandb_project: str
    wandb_group: str
    tokenizer_name: str
    vocab_size: int

@dataclass
class PyRNNConfig(ModelConfig):
    """Configuration for RNN models."""
    model_type: str = "PyRNN"
    hidden_size: int = 256
    num_layers: int = 2

@dataclass
class PyGRUConfig(ModelConfig):
    """Configuration for GRU models."""
    model_type: str = "PyGRU"
    hidden_size: int = 256
    num_layers: int = 2

@dataclass
class PyLSTMConfig(ModelConfig):
    """Configuration for LSTM models."""
    model_type: str = "PyLSTM"
    hidden_size: int = 256
    num_layers: int = 2

@dataclass
class PyTransformerConfig(ModelConfig):
    """Configuration for Transformer models."""
    model_type: str = "PyTransformer"
    wandb_project: str = "PyGPT"
    wandb_group: str = "Transformers"
    tokenizer_name: str = "py150k_large"
    block_size: int = 1024
    vocab_size: int = 578 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    
class PyGenerator(ABC, nn.Module):
    @abstractmethod
    def generate(self, batch_size:int, max_len:int, nucleus_threshold:float=0.9, starting_tokens:torch.Tensor=None):
        pass