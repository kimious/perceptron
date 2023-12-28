"""Basic perceptron usage"""

import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron


if __name__ == '__main__':
    training_data_features = np.array([
        [10, 10], [20, 16], [30, 12], [40, 13], [50, 15], [60, 8], [70, 10], [80, 11],
        [10, 40], [20, 44], [30, 50], [40, 52], [50, 60], [60, 55], [70, 41], [80, 57],
    ])

    training_data_classifications = np.array(
        [
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
        ]
    )

    p = Perceptron()
    p.fit(training_data_features, training_data_classifications)

    test_data = np.array([
        [10, 12], [10, 50],
        [30, 16], [30, 60]
    ])
    test_results = p.predict(test_data)

    _, ax = plt.subplots()
    ax.scatter(
        training_data_features[:, 0],
        training_data_features[:, 1],
        color=np.where(training_data_classifications == 0, 'orange', 'pink')
    )

    ax.scatter(
        test_data[:, 0],
        test_data[:, 1],
        s=100, edgecolor='black', color=np.where(test_results == 0, 'orange', 'pink')
    )

    plt.show()
