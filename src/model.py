import numpy as np
from src.layers import DenseLayer, SoftmaxLayer
from src.loss import cross_entropy_loss


class NeuralNetwork:
    """Simple 2-layer neural network: Dense(ReLU) -> Dense(Softmax)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.hidden = DenseLayer(input_size, hidden_size)
        self.output = SoftmaxLayer(hidden_size, output_size)

    def forward(self, X: np.ndarray):
        A_hidden = self.hidden.forward(X)
        A_output = self.output.forward(A_hidden)
        return A_hidden, A_output

    def backward(self, y_true: np.ndarray, A_hidden: np.ndarray, A_output: np.ndarray, learning_rate: float = 0.01):
        # Softmax + cross-entropy gradient simplified: (A - y) / batch
        batch_size = A_output.shape[0]
        dZ_output = (A_output - y_true) / batch_size

        dA_hidden = self.output.backward(dZ_output)
        _ = self.hidden.backward(dA_hidden)

        # Update parameters
        self.output.update(learning_rate)
        self.hidden.update(learning_rate)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return cross_entropy_loss(y_true, y_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, probs = self.forward(X)
        return np.argmax(probs, axis=1)