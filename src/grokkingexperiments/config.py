import tomllib
from pydantic import BaseModel, PositiveInt, PositiveFloat
from typing import Literal, Union
from pathlib import Path

class ModelConfig(BaseModel):
    actv_func : Literal["relu", "tanh", "sigmoid"]  # non linear activation func used by model
    embed_dim : PositiveInt                         # models embedding dimension
    hidden_dim : PositiveInt                        # models hidden dimension

class TrainingConfig(BaseModel):
    epochs : PositiveInt                            # Max number of passes over train set

class GroupConfig(BaseModel):
    name : Literal["Zn"]                      # Name of group - see selection.py
    n : PositiveInt                                 # Number of elements
    split : PositiveFloat                           # proportion of data used for trainin

class SettingsConfig(BaseModel):
    seed : PositiveInt
    print_progress : bool
    progress_interval : PositiveInt
    model_dir : str
    model_filename  : str

class Config(BaseModel):
    model : ModelConfig
    train : TrainingConfig
    group  : GroupConfig
    settings : SettingsConfig

def load_config(file_path = None):
    if not file_path:
        ROOT = Path(__file__).resolve().parents[2]
        file_path = ROOT / "config.toml"


    with open(file_path, "rb") as f:
        config_data = tomllib.load(f)
    return Config.model_validate(config_data)

def print_config(config):
    print("Config")
    # Turn config to dict and loop over entries
    for section, field in config.model_dump().items():
        print(f"  {section}")
        for key, value in field.items():
            print(f"    | {key : <20} {value}")

