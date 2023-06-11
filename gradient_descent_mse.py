import pandas as pd
import numpy as np


class GradientDescentMse:

    # Base class for implementing gradient descent
    # in a linear regression task

    # Create initialization method, which gets samples,
    # targets, learning rate (step), threshold
    # and a copy as arguments
    # learning rate, threshold and copy with default values
    def __init__(self, samples: pd.DataFrame,
                 targets: pd.DataFrame,
                 learning_rate: float = 1e-3,
                 threshold=1e-6,
                 copy: bool = True):

        self.samples = samples
        self.targets = targets
        self.learning_rate = learning_rate
        self.threshold = threshold

        # If copy equal True, then create
        # copy of samples and targets
        if copy:
            self.samples = samples.copy()
        else:
            self.samples = samples

        self.iteration_loss_dict = {}

    # We can add B0 to betas with this method
    # Create column with constant feature
    # Add this column to the samples
    # Add 1 to betas
    def add_constant_feature(self):

        constant_feature = np.ones((self.samples.shape[0], 1))
        self.samples = np.hstack((self.samples, constant_feature))
        self.beta = np.ones(self.samples.shape[1])

    # Method to determine the MSE
    # Find all predictions
    # Figure out loss of our predictions
    # Find MSE and return it
    def calculate_mse_loss(self):

        predictions = np.dot(self.samples, self.beta)
        loss = (predictions - self.targets) ** 2
        mse = np.mean(loss)

        return (mse)

    # Implementing a math formula to determine
    # the gradient of a function in numpy
    def calculate_gradient(self) -> np.ndarray:

        value_in_tuple = np.dot(self.samples, self.beta).ravel() - self.targets.ravel()
        gradient = 2 * np.mean(value_in_tuple.reshape(-1, 1) * self.samples, axis=0)

        return gradient

    # Rewrite beta versions with new values
    def iteration(self):

        self.gradient = self.calculate_gradient()
        self.beta = self.beta - self.learning_rate * self.gradient

    # Iterative training of model weights until a threshold is triggered
    # Write MSE and iteration number in iteration_loss_dict
    def learn(self):

        self.previous_mse = self.calculate_mse_loss()
        self.iteration()
        self.new_mse = self.calculate_mse_loss()

        while abs(self.previous_mse - self.new_mse) > self.threshold:
            self.previous_mse = self.calculate_mse_loss()
            self.iteration()
            self.new_mse = self.calculate_mse_loss()
            self.iteration_loss_dict[len(self.iteration_loss_dict) + 1] = self.new_mse

        return "Learning process completed successfully"
