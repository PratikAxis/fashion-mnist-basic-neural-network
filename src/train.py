"""
Training script for the NumPy-only Fashion-MNIST classifier.
"""

import numpy as np
from src.dataset import load_data, batch_iterator
from src.model import NeuralNetwork
from src.loss import cross_entropy_loss
from src.utils import plot_metrics, plot_confusion


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


def train_model(
    hidden_size: int = 128,
    learning_rate: float = 0.1,
    batch_size: int = 64,
    epochs: int = 5,
):
    (X_train, y_train), (X_test, y_test) = load_data(one_hot=True)
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=y_train.shape[1])

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        # Shuffle each epoch
        for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size):
            A_hidden, A_output = model.forward(X_batch)
            model.backward(y_batch, A_hidden, A_output, learning_rate)

        # Evaluation after epoch
        train_hidden, train_output = model.forward(X_train)
        test_hidden, test_output = model.forward(X_test)

        train_loss = cross_entropy_loss(y_train, train_output)
        train_acc = accuracy(y_train, train_output)
        test_acc = accuracy(y_test, test_output)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_acc: {test_acc:.4f}")

    plot_metrics(history)
    plot_confusion(y_test, test_output)
    return model, history


if __name__ == "__main__":
    train_model()

