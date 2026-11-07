import matplotlib.pyplot as plt
import tomllib
from pydantic import BaseModel, PositiveInt, PositiveFloat
from typing import Literal, Union

def plot_metrics(train, test, title, ylabel):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.plot(train, c='b', label='train')
    plt.plot(test, c='r', label='test')

    plt.ylabel(ylabel)
    plt.xlabel('100 Epoch')
    plt.legend()
    plt.show()

def plot_grokking_results(train_loss, test_loss, train_acc, test_acc):
    """Plots Loss and Accuracy side-by-side in a single window tab."""
    # Create 1 row with 2 columns side-by-side (14 inches wide, 5 inches high)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Left Chart: Loss
    ax1.plot(train_loss, c='b', label='train')
    ax1.plot(test_loss, c='r', label='test')
    ax1.set_title("Train Loss and Test Loss per Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epochs")
    ax1.legend()
    
    # 2. Right Chart: Accuracy
    ax2.plot(train_acc, c='b', label='train')
    ax2.plot(test_acc, c='r', label='test')
    ax2.set_title("Train Accuracy and Test Accuracy per Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.legend()
    
    # Clean up the spacing and display a single window
    plt.tight_layout()
    plt.show()

class ModelConfig(BaseModel):
    actv_func : Literal["relu", "tanh", "sigmoid"]
    embed_dim : PositiveInt
    hidden_dim : PositiveInt

class TrainingConfig(BaseModel):
    # this lets us specify either a single value or a list of values for n and split
    n : Union[PositiveInt, list[PositiveInt]]
    op : Literal["mod_n", "bitwise_and"]
    split : PositiveFloat
    epochs : PositiveInt

class Config(BaseModel):
    model: ModelConfig
    train: TrainingConfig

def load_config(file_path: str) -> Config:
    with open(file_path, "rb") as f:
        config_data = tomllib.load(f)
    return Config.model_validate(config_data)