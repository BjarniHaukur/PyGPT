from .rnn import PyRNN
from .lstm import PyLSTM
from .gru import PyGRU
from .transformer import PyTransformer
from .base import ModelConfig, PyRNNConfig, PyGRUConfig, PyLSTMConfig, PyTransformerConfig, PyGenerator, CONFIG_PATH

import yaml
from pathlib import Path

import torch

CHECKPOINT_PATH = Path("checkpoints/models")


def save_config(config: "ModelConfig", file_name: str):
    with open(CONFIG_PATH / file_name, 'w') as file:
        yaml.dump(config.__dict__, file)


def load_config(file_name: str) -> 'ModelConfig':
    file_name += ".yaml" if not file_name.endswith(".yaml") else ""
    with open(CONFIG_PATH / file_name, 'r') as file:
        config_dict = yaml.safe_load(file)

    if config_dict["model_type"] == "PyRNN":
        return PyRNNConfig(**config_dict)
    elif config_dict["model_type"] == "PyGRU":
        return PyGRUConfig(**config_dict)
    elif config_dict["model_type"] == "PyLSTM":
        return PyLSTMConfig(**config_dict)
    elif config_dict["model_type"] == "PyTransformer":
        return PyTransformerConfig(**config_dict)
    else:
        raise ValueError(f"Invalid model_type, got {config_dict['model_type']}")


def model_from_config(config: "ModelConfig", device="cpu") -> "PyGenerator":
    if config.model_type == "PyRNN":
        model = PyRNN(**config.__dict__)
    elif config.model_type == "PyGRU":
        model = PyGRU(**config.__dict__)
    elif config.model_type == "PyLSTM":
        model = PyLSTM(**config.__dict__)
    elif config.model_type == "PyTransformer":
        model = PyTransformer(**config.__dict__)
    else:
        raise ValueError(f"Invalid model_type, got {config.model_type}")
    
    model.to(device)
    return model


def model_from_checkpoint(checkpoint_path: str, device="cpu") -> "PyGenerator":
    checkpoint = torch.load(CHECKPOINT_PATH / checkpoint_path, map_location=device)
    config = load_config(checkpoint["config_name"])
    model = model_from_config(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model