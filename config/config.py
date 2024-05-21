import yaml
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path("config/")


def save_config(config: "ModelConfig", file_name: str):
    with open(CONFIG_PATH / file_name, 'w') as file:
        yaml.dump(config.__dict__, file)

def load_config(file_name: str) -> 'ModelConfig':
    file_name += ".yaml" if not file_name.endswith(".yaml") else ""
    with open(CONFIG_PATH / file_name, 'r') as file:
        config_dict = yaml.safe_load(file)

    match config_dict["model_type"]:
        case "PyRNN":
            return PyRNNConfig(**config_dict)
        case "PyLSTM":
            return PyLSTMConfig(**config_dict)
        case "PyTransformer":
            return PyTransformerConfig(**config_dict)
        case _:
            raise ValueError(f"Invalid model_type, got {config_dict["model_type"]}")

@dataclass
class ModelConfig(ABC):
    """Abstract base class for model configurations."""
    wandb_project: str
    wandb_group: str
    tokenizer_name: str
    vocab_size: int # i.e. the vocab size

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
