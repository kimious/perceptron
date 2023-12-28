"""Basic perceptron usage"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from perceptron import Perceptron

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

if __name__ == '__main__':
    df = pd.read_csv(DATASET_URL, header=None, encoding='utf-8')
    inputs = df.iloc[:100, [0, 2]].values
    labels = np.where(df.iloc[:100, 4].values == 'Iris-setosa', 1, 0)

    model = Perceptron()
    model.fit(inputs, labels)

    test_data = np.array([[7, 4.2], [5, 1]])
    test_results = model.predict(test_data)

    _, ax = plt.subplots()
    ax.scatter(
        inputs[:, 0],
        inputs[:, 1],
        color=np.where(labels == 0, 'orange', 'pink')
    )

    ax.scatter(
        test_data[:, 0],
        test_data[:, 1],
        s=100, edgecolor='black', color=np.where(test_results == 0, 'orange', 'pink')
    )

    plt.show()
