import numpy as np
import math


class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000, function_option=1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.function_option = function_option
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        # learn weights
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        result = []
        for x_i in X:
            linear_output = np.dot(x_i, self.weights) + self.bias
            y_predicted = self.activation_func(linear_output)
            result = np.append(result, y_predicted)
        return result

    def activation_func(self, x):
        if self.function_option == 0:  # Heaviside
            if x < 0:
                return 0
            else:
                return 1
        elif self.function_option == 1:  # Logistic
            return 1 / (1 + math.exp(-x))
        elif self.function_option == 2:  # sin
            return math.sin(x)
        elif self.function_option == 3:  # tanh
            return math.tanh(x)
        elif self.function_option == 4:  # Sign
            if x < 0:
                return -1
            elif x == 0:
                return 0
            else:
                return 1
        elif self.function_option == 5:  # ReLu
            if x <= 0:
                return 0
            else:
                return x
        elif self.function_option == 6:  # Leaky ReLu
            if x > 0:
                return x
            else:
                return 0.01 * x
