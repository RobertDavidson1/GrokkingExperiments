from grokkingexperiments.mlp import MLP
from grokkingexperiments.train import train_model
import matplotlib.pyplot as plt
import torch 
import numpy as np
from pathlib import Path
from grokkingexperiments.config import Config
import plotly.express as px
import scienceplots


plt.style.use(["science", "notebook"])

class GrokkingModel(MLP):
    """
    Extend the MLP with training metrics and model visualization tools.
    """
    def __init__(self, config):
        self.config = config

        # To store metric histories
        self.epochs = []
        self.training_losses = []
        self.test_losses = []
        self.training_accuracies = []
        self.test_accuracies = []

        # Init parent MLP class
        super().__init__(config)


    def save_model(self):
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / self.config.settings.model_dir / self.config.settings.model_filename
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model_state_dict" : self.state_dict(), 
            "config" : self.config.model_dump(), 
            "metrics": {
              "epochs": self.epochs,
              "training_losses": self.training_losses,
              "test_losses": self.test_losses,
              "training_accuracies": self.training_accuracies,
              "test_accuracies": self.test_accuracies,
          },
            
        }

        torch.save(model_data, model_path)
        
        return model_path

    def load_model(self, model_path):

      device = next(self.parameters()).device

      model_data = torch.load(
          model_path,
          map_location=device,
          weights_only=True,
      )

      saved_config = Config.model_validate(model_data["config"])
      self.load_state_dict(model_data["model_state_dict"])
      self.config = saved_config

      metrics = model_data["metrics"]
      self.epochs = metrics["epochs"]
      self.training_losses = metrics["training_losses"]
      self.test_losses = metrics["test_losses"]
      self.training_accuracies = metrics["training_accuracies"]
      self.test_accuracies = metrics["test_accuracies"]
      return self

    def fit(self, x_train, y_train, x_test, y_test):
        train_model(self.config, self, x_train, y_train, x_test, y_test)

    def plot_metrics(self, end_epoch=None):
        """
        Plot training and test loss and accuracy over epochs.
        """

        fig, axs = plt.subplots(2,1, figsize=(8, 5))
        # 1. Left Chart: Loss
        axs[0].plot(self.epochs, self.training_losses, c='b', label='train')
        axs[0].plot(self.epochs, self.test_losses, c='r', label='test')
        axs[0].set_title('Loss per epoch')
        axs[0].set_ylabel("Loss")
        axs[0].set_xlabel("Epochs")
        axs[0].legend()

        # 2. Right Chart: Accuracy
        axs[1].plot(self.epochs, self.training_accuracies, c='b', label='train')
        axs[1].plot(self.epochs, self.test_accuracies, c='r', label='test')
        axs[1].set_title('Accuracy per epoch')
        axs[1].set_ylabel("Accuracy")
        axs[1].set_xlabel("Epochs")
        axs[1].legend()

        if end_epoch is not None:
          for ax in axs:
               ax.set_xlim(0, end_epoch)


        plt.tight_layout()
        plt.show()
    
    def plot_weights(self):
        """Visualize the model's weight matrices as heatmaps."""

        weights = {
            "Left Embedding Matrix": self.E_L.detach().cpu(),
            "Right Embedding Matrix": self.E_R.detach().cpu(),
            "Hidden Matrix": self.W_H.detach().cpu(),
            "Unembedding Matrix": self.W_U.detach().cpu(),
        }

        fig, axs = plt.subplots(2, 2, figsize=(20, 8))

        for i, (name, matrix) in enumerate(weights.items()):
            row = i // 2
            col = i % 2

            limit = matrix.abs().max().item()

            axs[row, col].imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit
            )

            axs[row, col].set_title(name)
            axs[row, col].set_xticks([0, matrix.shape[1] - 1])
            axs[row, col].set_yticks([0, matrix.shape[0] - 1])

        plt.tight_layout()
        plt.show()

    def plot_logit(self, logit_index):
        """
        Plot a logits scores for all possible outputs a,b 
        """

        # Get n and device
        n = self.config.group.n
        device = next(self.parameters()).device


        # Generate all possible inputs a,b
        x = torch.zeros((n**2, 2), dtype=torch.int64)
        x = x.to(device)
        for i in range(n):
            for j in range(n):
                x[i * n + j] = torch.tensor([i, j])

        # Get the all logits
        self.eval()
        with torch.no_grad():
            logits = self(x.to(device))

        # Select logits and reshape into grid of inputs a,b
        grid = logits[:, logit_index].reshape(n, n).cpu().numpy()

        plt.imshow(grid, cmap="hot", origin="lower")
        plt.title(f"Model's Score for Answer {logit_index}")
        plt.xlabel("Input a")
        plt.ylabel("Input b")
        plt.xticks(range(0, n, 10))
        plt.yticks(range(0, n, 10))
        plt.colorbar(label="Model Score (Logit)")
        plt.show()
    

    def plot_hidden_neuron_values(self, neuron, side="all", relu=False, title=None):
        
        if relu:
            activation_type = "activation"
            values = torch.relu(values)
        else:
            activation_type = "preactivation"

        if side == "all":
            both = self.compute_hidden_preactivations("both")
            left = self.compute_hidden_preactivations("left")
            right = self.compute_hidden_preactivations("right")

            grid = both[neuron, :, :]
            left_values = left[neuron, :]
            right_values = right[neuron, :]

            if relu:
                grid = torch.relu(grid)

            fig, axes = plt.subplots(1, 3, figsize=(18, 4), width_ratios=[1, 2, 2])

            # Full preactivation grid
            im = axes[0].imshow(grid, cmap="magma")
            axes[0].set_xlabel("Input b")
            axes[0].set_ylabel("Input a")
            axes[0].set_title("Both inputs")

            # Left contributions
            axes[1].plot(left_values)
            axes[1].set_xlabel("Input a")
            axes[1].set_ylabel("Preactivation contribution")
            axes[1].set_title("Left contributions")

            # Right contributions
            axes[2].plot(right_values)
            axes[2].set_xlabel("Input b")
            axes[2].set_ylabel("Preactivation contribution")
            axes[2].set_title("Right contributions")

        
            fig.suptitle(title if title is not None else f"Neuron {neuron} {activation_type}")
            plt.tight_layout()
            plt.show()
            return
            
        values = self.compute_hidden_preactivations(side)

        if side == "both":
            plt.figure(figsize=(5, 5))
            matrix = values[neuron, :, :]

            if relu:
                plt.imshow(matrix, cmap="magma", vmin=0)
            else:
                limit = matrix.abs().max().item()
                plt.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)

            plt.ylabel("a input")
            plt.xlabel("b input")

            default_title = f"Neuron {neuron} {activation_type} across all input pairs (a,b)"

        else:
            plt.figure(figsize=(40, 2))
            plt.plot(values[neuron, :])
            plt.xticks(np.arange(values.shape[1]))
            plt.xticks(rotation=-45)
            plt.ylabel(f"{activation_type.capitalize()} value")
            plt.xlabel("Input number")

            default_title = f"{side} {activation_type} for neuron {neuron}"

        plt.suptitle(title if title is not None else default_title)
        plt.show();

    def plot_all_hidden_neurons(self, side="both", relu=False):
        # Subplot code source:
        # https://stackoverflow.com/a/43733899

        values = self.compute_hidden_preactivations(side)

        if relu:
            values = torch.relu(values)

        num_neurons = self.W_H.shape[0]
        n = int(np.ceil(np.sqrt(num_neurons)))

        if side == "both":
            fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
        else:
            fig, axes = plt.subplots(n, n, figsize=(n * 5, n * 2))

        for i in range(n):
            for j in range(n):
                neuron = i * n + j

                if neuron >= num_neurons:
                    axes[i, j].axis("off")
                    continue

                if side == "both":
                    matrix = values[neuron, :, :]

                    if relu:
                        axes[i, j].imshow(matrix, cmap="magma", vmin=0)
                    else:
                        limit = matrix.abs().max().item()
                        axes[i, j].imshow(
                            matrix,
                            cmap="RdBu_r",
                            vmin=-limit,
                            vmax=limit
                        )

                    axes[i, j].set(
                        xticks=[0, self.config.group.n - 1],
                        yticks=[0, self.config.group.n - 1]
                    )

                else:
                    axes[i, j].plot(values[neuron, :])
                    axes[i, j].set_xticks([0, self.config.group.n - 1])

                axes[i, j].set_title(f"Neuron {neuron}")

        plt.tight_layout()
        plt.show()

    def compute_hidden_preactivations(self, side, neuron=None):
        E_L = self.E_L.detach().cpu()
        E_R = self.E_R.detach().cpu()
        W_H = self.W_H.detach().cpu()

        d = E_L.shape[0]

        W_H_L = W_H[:, :d] @ E_L
        W_H_R = W_H[:, d:] @ E_R

        if side == "left":
            values = W_H_L

        elif side == "right":
            values = W_H_R

        elif side == "both":
            values = W_H_L.unsqueeze(2) + W_H_R.unsqueeze(1)

        else:
            raise ValueError("side must be 'left', 'right', or 'both'")

        if neuron is not None:
            return values[neuron]

        return values

        
