from grokkingexperiments.mlp import MLP
from grokkingexperiments.train import train_model
import matplotlib.pyplot as plt
import torch 
import numpy as np

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

        # Init parent Multi Layer Perceptron class
        super().__init__(config)

    def fit(self, x_train, y_train, x_test, y_test):
        train_model(self.config, self, x_train, y_train, x_test, y_test)

    def plot_metrics(self):
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

        plt.tight_layout()
        plt.show()

    def plot_weights(self):
        """
        Visualize the model's weight matrices as heatmaps.
        """

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 8))

        axs[0,0].imshow(self.left_embed.detach().to("cpu"), aspect='auto')
        axs[0,0].set_title("Left Embedding Matrix")


        axs[0,1].imshow(self.right_embed.detach().to("cpu"), aspect='auto')
        axs[0,1].set_title("Right Embedding Matrix")

        axs[1,0].imshow(self.hidden.detach().to("cpu"), aspect='auto')
        axs[1,0].set_title("Hidden Matrix")

        axs[1,1].imshow(self.unembedding.detach().to("cpu"), aspect='auto')
        axs[1,1].set_title("Unembedding Matrix")

        plt.tight_layout()
        plt.show()
    


    def plot_hidden_neuron_values(self, neuron, side="both", relu=False):  
        values = self.__compute_hidden_preactivations(side)
        
        if relu :
            activation_type = "activation"
            values = torch.relu(values)
        else:
            activation_type = "preactivation"
     
        if side == "both":
            plt.figure(figsize=(5, 5))
            plt.imshow(values[:, :, neuron])
            plt.ylabel("a input")
            plt.xlabel("b input")
            plt.title(f"Neuron {neuron} {activation_type} across all input pairs (a,b)")
        
        else:
            plt.figure(figsize=(30, 2))
            plt.plot(values[:, neuron])
            plt.title(f"{side} {activation_type} for neuron {neuron}" )
            plt.xticks(np.arange(values.shape[0]))
            plt.xticks(rotation=-45)
            plt.ylabel(f"{activation_type.capitalize()} value")
            plt.xlabel("Input number")

        plt.show();

    def plot_all_hidden_neurons(self, side="both"):
        # Subplot code source:
        # Source - https://stackoverflow.com/a/43733899
        # Posted by ImportanceOfBeingErnest
        preactivations = self.__compute_hidden_preactivations(side)
        
        num_neurons = self.hidden.shape[1]
        n = int(np.ceil(np.sqrt(num_neurons)))
        fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))

        for i in range(n):
            for j in range(n):
                neuron = (i*n) + j

                if neuron >= num_neurons:
                    axes[i, j].axis("off")
                    continue

                if side == "both":
                    axes[i, j].imshow(preactivations[:, :, neuron])
                else:
                    axes[i, j].plot(preactivations[:, neuron])

                axes[i, j].set_title(f"Neuron {neuron}")
        
        plt.tight_layout()
        plt.show()


    def __compute_hidden_preactivations(self, side):
        left = self.left_embed.detach().to("cpu")
        right = self.right_embed.detach().to("cpu")
        hidden = self.hidden.detach().to("cpu")
        embed_dim = left.shape[1]

        if side == "left":
            values = torch.matmul(left, hidden[:embed_dim, :])
        elif side == "right":
                values = torch.matmul(right, hidden[embed_dim:, :])
        elif side == "both":
            broadcast_left = torch.matmul(left, hidden[:embed_dim, :]).unsqueeze(1)
            broadcast_right = torch.matmul(right, hidden[embed_dim:, :]).unsqueeze(0)
            values = broadcast_left + broadcast_right 

        return values
        

