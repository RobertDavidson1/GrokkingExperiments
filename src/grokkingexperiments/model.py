from grokkingexperiments.mlp import MLP
import matplotlib.pyplot as plt

class GrokkingModel(MLP):
    """

    """
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

        # Clean up the spacing and display a single window
        plt.tight_layout()
        plt.show()


