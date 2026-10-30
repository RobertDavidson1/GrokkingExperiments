import torch
import torch.nn as nn
import torch.optim as optim
from grokkingexperiments.model import GrokkingModel


def calculate_accuracy(logits, true_values):
    # Adapted from:
    # Source - https://stackoverflow.com/a/59229237
    # Posted by alercelik, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-10-30, License - CC BY-SA 4.0

    # Get our predictions
    predictions = logits.argmax(dim=1)

    # (predictions == true_values) returns an array True / False values 
    # depending if we predicted correctly we then
    # turn to float and get the average
    return (predictions == true_values).float().mean()


def train_model(epochs, model, x_train, y_train, x_test, y_test):
    # To store metric histories
    training_losses, test_losses = [], []
    training_accuracies, test_accuracies = [], []

    optimizer  = optim.AdamW(model.parameters(), weight_decay=1)
    loss_func = nn.CrossEntropyLoss()

    # Iterate over the dataset
    for i in range(epochs):
        model.train()                       # Put the model in training mode
        optimizer.zero_grad()               # Clear the previous training steps gradients
        logits = model(x_train)             # Preict the entire training dataset 
        loss = loss_func(logits, y_train)   # Calculuate the loss / error of our prediction vs true values
        loss.backward()                     # Calculate the derivatives
        optimizer.step()                    # Update the weights according to the derivatives

        if i % 100 == 0:
            model.eval()                                                # Put the model in evaluation mode so its doesnt update weights
            with torch.no_grad():                                       # Disable derivative calculation
                test_logits = model(x_test)                             # Predict the entire test dataset
                test_loss = loss_func(test_logits, y_test)              # Calculate the loss for the test predictions vs true values
                train_accuracy = calculate_accuracy(logits, y_train)    # Calculate train accuracy
                test_accuracy = calculate_accuracy(test_logits, y_test) # Calculate test accuracy

            # Record losses
            training_losses.append(loss.item())             
            test_losses.append(test_loss.item())

            # Record accuracy 
            training_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)


            # Print information
            print(f"Epoch {i : < 7} | Loss {loss : <6.3f} | Test loss {test_loss: <6.3f} | Train accuracy {train_accuracy : <6.3f} | Test accuracy {test_accuracy : <6.3f}")
    
            # Stop training when our model achieves 100% accuracy on the test data
            if test_accuracy == 1.0:
                break
    
    return training_losses, test_losses, training_accuracies, test_accuracies