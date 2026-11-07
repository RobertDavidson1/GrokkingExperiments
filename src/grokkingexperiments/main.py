import torch
from grokkingexperiments.model import GrokkingModel
from grokkingexperiments.train import train_model
<<<<<<< HEAD
from grokkingexperiments.tasks import select_task
=======
from grokkingexperiments.data import generate_train_test_data
from grokkingexperiments.utils import plot_metrics, plot_grokking_results
from grokkingexperiments.utils import load_config
>>>>>>> origin/config-pipeline


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

<<<<<<< HEAD
    # Create the model
    model = GrokkingModel(n=config.train.n, embed_dimension=config.model.embed_dim, hidden_dimension=config.model.hidden_dim).to(device)

    # Create training task and get the splits
    task = select_task(config.task.type, config.task.name, config.task.n)
    x_train, y_train, x_test, y_test = task.generate_train_test_data(config)

    # Move data to the same device as the model
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Train the model, and store the four training metric histories in `metrics`
    train_model(config.train.epochs, model, x_train, y_train, x_test, y_test)

    # Plot model metrics
    model.plot_metrics()
=======
    n_values = config.train.n

    if isinstance(n_values, int):
            n_values = [n_values]

    for current_n in n_values:
        print(f"\n--- Starting Experiment: n = {current_n} ---")
        # Create the model
        model = GrokkingModel(n=current_n, embed_dimension=config.model.embed_dim, hidden_dimension=config.model.hidden_dim).to(device)

        # Get the data splits
        x_train, y_train, x_test, y_test = generate_train_test_data(current_n, config.train.op, config.train.split)

        # Move data to the same device as the model
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        # Train the model, and store the four training metric histories in `metrics`
        metrics = train_model(config.train.epochs, model, x_train, y_train, x_test, y_test)

        # Unpack the metrics
        training_losses = metrics[0]
        test_losses = metrics[1]
        training_accuracies = metrics[2]
        test_accuracies = metrics[3]

        # Plot the metrics
        plot_grokking_results(training_losses, test_losses, training_accuracies, test_accuracies)
        # plot_metrics(training_losses, test_losses, 'Train Loss and Test Loss per epoch', 'loss')
        # plot_metrics(training_accuracies, test_accuracies, 'Train Accuracy and Test Accuracy per epoch', 'accuracy')
        print(f"--- Finished Experiment: n = {current_n} ---\n")
>>>>>>> origin/config-pipeline
