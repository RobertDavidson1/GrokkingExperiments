from grokkingexperiments.mlp import MLP
import matplotlib.pyplot as plt

class GrokkingModel(MLP):
    def __init__(self, config):
        # To store metric histories
        self.epochs = []
        self.training_losses = []
        self.test_losses = []
        self.training_accuracies = []
        self.test_accuracies = []

        # Init parent Multi Layer Perceptron class
        super().__init__(config)

    def plot_metrics(self):


        # Plot Losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.training_losses, c='b', label='train')
        plt.plot(self.epochs, self.test_losses, c='r', label='test')
        plt.title('Loss per epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

        # Plot accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.training_accuracies, c='b', label='train')
        plt.plot(self.epochs, self.test_accuracies, c='r', label='test')
        plt.title('Accuracy per epoch')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()
