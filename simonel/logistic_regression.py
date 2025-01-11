import numpy as np

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.1, iterations=5000, tol=1e-6, penalty="l2", C=1.0):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tol = tol
        self.penalty = penalty
        self.C = C
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, y, predictions):
        """Compute the binary cross-entropy cost."""
        n_samples = len(y)
        cost = -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
        if self.penalty == "l2":
            l2_term = (self.C / (2 * n_samples)) * np.sum(self.weights ** 2)
            cost += l2_term
        return cost

    def fit(self, X, y):
        """Train the Logistic Regression model."""
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(-0.1, 0.1, n_features)  # Wider initialization
        self.bias = 0
        prev_cost = float("inf")

        for i in range(self.iterations):
            # Linear model
            model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(model)

            # Compute class weights
            class_weights = {0: 1.0 / sum(y == 0), 1: 1.0 / sum(y == 1)}
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y) * np.vectorize(class_weights.get)(y))
            db = (1 / n_samples) * np.sum((predictions - y) * np.vectorize(class_weights.get)(y))

            # Scale gradients to prevent vanishing
            dw /= np.linalg.norm(dw) + 1e-8

            # Add regularization penalty after 500 iterations
            if self.penalty == "l2" and i > 500:
                dw += (self.C / n_samples) * self.weights

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute the cost
            cost = self.compute_cost(y, predictions)

            # Debug gradients and sigmoid input range
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}, Gradients (dw): {np.linalg.norm(dw):.6f}")
                print(f"Sigmoid Input Range: {model.min()} to {model.max()}")

            # Convergence check
            if i > 100 and abs(prev_cost - cost) < self.tol:
                print(f"Convergence achieved at iteration {i}")
                break

            prev_cost = cost

    def predict(self, X):
        """Predict labels for the input data."""
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)
        return [1 if i > 0.5 else 0 for i in predictions]