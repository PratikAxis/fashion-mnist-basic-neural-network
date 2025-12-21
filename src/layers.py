                    #-----------------hidden layer (linear act. + ReLU)---------------------
import numpy as np


class DenseLayer:
    """A fully-connected layer with ReLU activation.

    Weight shapes:
      W: (output_size, input_size)
      b: (output_size,)

    Inputs/outputs are 2D arrays with shape (batch_size, features).
    """

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        # small random initialization
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size,))

        # placeholders for gradients
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: linear -> ReLU

        X shape: (batch, input_size)
        returns A shape: (batch, output_size)
        """
        self.X = X
        self.Z = X.dot(self.W.T) + self.b[np.newaxis, :]
        self.A = np.maximum(0, self.Z)
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Backward pass receiving dA (batch, output_size).

        Returns dX (batch, input_size) to propagate to previous layer.
        Also computes gradients dW and db (averaged over batch).
        """
        batch_size = dA.shape[0]
        dZ = dA * (self.Z > 0)
        self.dW = dZ.T.dot(self.X) / batch_size
        self.db = np.sum(dZ, axis=0) / batch_size
        dX = dZ.dot(self.W)
        return dX

    def update(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class SoftmaxLayer:
    """Output layer implementing linear + softmax. Keeps gradients for SGD updates."""

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size,))

        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities.

        X shape: (batch, input_size)
        returns probs shape: (batch, output_size)
        """
        self.X = X
        logits = X.dot(self.W.T) + self.b[np.newaxis, :]
        # numeric stability
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.A = exp / np.sum(exp, axis=1, keepdims=True)
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Backward when dA is gradient at softmax output (batch, output_size).

        Returns dX to propagate to previous layer and stores dW, db for update.
        """
        batch_size = dA.shape[0]
        # dA is expected to be (A - y) / batch or similar
        dZ = dA
        self.dW = dZ.T.dot(self.X) / batch_size
        self.db = np.sum(dZ, axis=0) / batch_size
        dX = dZ.dot(self.W)
        return dX

    def update(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db