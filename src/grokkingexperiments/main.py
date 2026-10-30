import torch 
from grokkingexperiments.model import GrokkingModel
from grokkingexperiments.train import train_model
from grokkingexperiments.data import generate_train_test_data
from grokkingexperiments.utils import plot_metrics

import matplotlib.pyplot as plt

def main(n, op, split,embed_dim, hidden_dim, epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the model
    model = GrokkingModel(n=n, embed_dimension=embed_dim, hidden_dimension=hidden_dim).to(device) 
    
    # Get the data splits
    x_train, y_train, x_test, y_test = generate_train_test_data(n, op, split)
    
    # Move data to the same device as the model
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Train the model, and store the four training metric histories in `metrics`
    metrics = train_model(epochs, model, x_train, y_train, x_test, y_test)

    # Unpack the metrics
    training_losses = metrics[0]
    test_losses = metrics[1]
    training_accuracies = metrics[2]
    test_accuracies = metrics[3]

    # Plot the metrics 
    plot_metrics(training_losses, test_losses, 'Train Loss and Test Loss per eporch', 'loss')
    plot_metrics(training_accuracies, test_accuracies, 'Train Accuracy and Test Accuracy per epoch', 'accuracy')
