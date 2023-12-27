import os
import numpy as np
import pandas as pd
import matplotlib as plt

class Perceptron:
  def __init__(self, learning_rate=0.01, epochs=500, random_seed=1):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.random_seed = random_seed
  
  def fit(self, X, y):
    self.weights_ = np.random.default_rng().random(X.shape[1])
    self.bias_ = 0.0
    self.errors_ = []

    for _ in range(self.epochs):
      errors = 0
      for xi, target in zip(X, y):
        # update is the difference (float) between binary prediction and actual truth
        # learning rate is used to converge slowly to the sweet spot value
        update = self.learning_rate * (target - self.predict(xi))
        # update weight proportionally to current example feature values
        self.weights_ += update * xi 
        # bias does not depend on any features
        self.bias_ += update
        errors += int(update != 0.0)
      self.errors_.append(errors)

    return self              
  
  def predict(self, X):
    return np.where(self.net_input_(X) >= 0.0, 1, 0)
  
  def net_input_(self, X):
    return np.dot(X, self.weights_) + self.bias_

def main():
  training_data_features = np.array([
    [10, 10], [20, 16], [30, 12], [40, 13], [50, 15], [60, 8], [70, 10], [80, 11], [90, 15], [100, 13],
    [10, 40], [20, 44], [30, 50], [40, 52], [50, 60], [60, 55], [70, 41], [80, 57], [90, 47], [100, 50]
  ])

  training_data_classifications = np.array(
    [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
  )

  p = Perceptron()
  p.fit(training_data_features, training_data_classifications)
  print(p.errors_)
  print(p.weights_)
  print(p.bias_)

  # should return [0, 1, 0, 1]
  test_results = p.predict(
    np.array([
      [10, 12], [10, 50],
      [30, 16], [30, 60]
    ])
  )
  print(test_results)

if __name__ == '__main__':
  main()