"""
Single-layer perceptron to do binary classification using 2 labels (0 and 1).
(see https://en.wikipedia.org/wiki/Perceptron)
"""

import numpy as np

class Perceptron:
    """Simple Perceptron implementation"""
    def __init__(self, learning_rate=0.01, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._weights = []
        self._bias = 0.0


    def fit(self, inputs, labels):
        """Calibrates perceptron's weights and bias unit."""
        self._weights = np.zeros(inputs.shape[1])
        self._bias = 0.0

        for _ in range(self.epochs):
            for features, label in zip(inputs, labels):
                # update is the difference (float) between binary prediction and actual truth
                # learning rate is used to converge slowly to the sweet spot value
                update = self.learning_rate * (label - self.predict(features))
                # update weight proportionally to current example feature values
                self._weights += update * features
                # bias does not depend on any features
                self._bias += update

        return self

    def predict(self, features):
        """Returns the output using current weights, bias unit and simple threshold function."""
        return np.where(self._net_input(features) >= 0.0, 1, 0)

    def _net_input(self, features):
        return np.dot(features, self._weights) + self._bias
