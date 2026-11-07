import torch
from grokkingexperiments.model import GrokkingModel
from grokkingexperiments.train import train_model
from grokkingexperiments.tasks import select_task
from grokkingexperiments.config import load_config


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create the model
    model = GrokkingModel(config).to(device)

    # Create training task and get the splits
    task = select_task(config)
    x_train, y_train, x_test, y_test = task.generate_train_test_data()

    # Move data to the same device as the model
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Train the model, and store the four training metric histories in `metrics`
    train_model(config, model, x_train, y_train, x_test, y_test)

    # Plot model metrics
    model.plot_metrics()


