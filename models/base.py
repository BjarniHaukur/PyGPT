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
class PyLSTMConfig(ModelConfig):
    """Configuration for LSTM models."""
    model_type: str = "PyLSTM"
    hidden_size: int = 256
    num_layers: int = 2

@dataclass
class PyTransformerConfig(ModelConfig):
    """Configuration for Transformer models."""
    model_type: str = "PyTransformer"
    context_window_size: int = 512
    d_model: int = 512
    d_feedforward: int = 2048
    num_attn_heads: int = 8
    num_decoder_layers: int = 6
    dropout: float = 0.1
    activation: str = "relu"
    norm_type: str = "prenorm"
    
class PyGenerator(ABC, nn.Module):
    @abstractmethod
    def generate(self, batch_size:int, max_len:int, nucleus_threshold:float=0.9, starting_tokens:torch.Tensor=None):
        pass