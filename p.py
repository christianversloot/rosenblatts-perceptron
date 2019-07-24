import numpy as np

# Basic Rosenblatt Perceptron implementation
class RBPerceptron:

  # Constructor
  def __init__(self, number_of_epochs = 100, learning_rate = 0.1):
    self.number_of_epochs = number_of_epochs
    self.learning_rate = learning_rate

  # Train perceptron
  def train(self, X, D):
    # Initialize weights vector with zeroes
    num_features = X.shape[1]
    self.w = np.zeros(num_features + 1)
    # Perform the epochs
    for i in range(self.number_of_epochs):
      # For every combination of (X_i, D_i)
      for sample, desired_outcome in zip(X, D):
        # Generate prediction and compare with desired outcome
        prediction    = self.predict(sample)
        difference    = (desired_outcome - prediction)
        # Compute weight update via Perceptron Learning Rule
        weight_update = self.learning_rate * difference
        self.w[1:]    += weight_update * sample
        self.w[0]     += weight_update
    return self

  # Generate prediction
  def predict(self, sample):
    outcome = np.dot(sample, self.w[1:]) + self.w[0]
    return np.where(outcome > 0, 1, 0)