import yaml

from .rnn import PyRNN
from .lstm import PyLSTM
from .transformer import PyTransformer
from .base import ModelConfig, PyRNNConfig, PyLSTMConfig, PyTransformerConfig, PyGenerator, CONFIG_PATH


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
            raise ValueError(f"Invalid model_type, got {config_dict['model_type']}")
        
        
# Things that can be "quantified" are handled by the config, things like architecture changes should be different classes (reduces boilerplate a tone)
# i.e. storing things like "prenorm" or "postnorm" in the config is nice but is effectively ignored by the __init__ 
def model_from_config(config: "ModelConfig")-> "PyGenerator":
    match config.model_type:
        case "PyRNN":
            return PyRNN(**config.__dict__)
        case "PyLSTM":
            return PyLSTM(**config.__dict__)
        case "PyTransformer":
            return PyTransformer(**config.__dict__)
        case _:
            raise ValueError(f"Invalid model_type, got {config.model_type}")
