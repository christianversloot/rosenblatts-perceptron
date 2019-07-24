# Import libraries
import numpy as np
from p import RBPerceptron
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
# Generate target classes {0, 1}
zeros = np.zeros(50)
ones = zeros + 1
targets = np.concatenate((zeros, ones))
# Generate data
small = np.random.normal(5, 0.25, (50,2))
large = np.random.normal(6.5, 0.25, (50,2))
# Prepare input data
X = np.concatenate((small,large))
D = targets
# Initialize Perceptron
# 2000 epochs, 0.1 learning rate
rbp = RBPerceptron(2000, 0.1)
# Train Perceptron
trained_model = rbp.train(X, D)
# Plot results
plot_decision_regions(X, D.astype(np.integer), clf=trained_model)
plt.title('Perceptron')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()