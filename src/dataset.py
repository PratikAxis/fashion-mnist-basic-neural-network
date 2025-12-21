import numpy as np

try:
    from tensorflow.keras.datasets import fashion_mnist
except Exception:
    # If tensorflow isn't available, try keras (standalone) as a fallback
    from keras.datasets import fashion_mnist


def load_data(one_hot: bool = True):
    """Load, normalize and reshape Fashion-MNIST.

    Returns: (X_train, y_train), (X_test, y_test)
      X shapes: (n_samples, 784)
      y (one_hot=True): (n_samples, 10) else integer labels (n_samples,)
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    if one_hot:
        y_train_ohe = np.eye(10)[y_train]
        y_test_ohe = np.eye(10)[y_test]
        return (X_train, y_train_ohe), (X_test, y_test_ohe)
    return (X_train, y_train), (X_test, y_test)


def batch_iterator(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True):
    """Yield mini-batches of (X_batch, y_batch)."""
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


print("Train set: ", X_train.shape, y_train.shape) # --->Train set:  (60000, 28, 28) (60000,)
print("Train set: ", X_test.shape, y_test.shape) # --->Train set:  (10000, 28, 28) (10000,)

# devide both for normalize the pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# flatten each image (28x28) into a vector of length of 784
X_train = X_train.reshape(X_train.shape[0], -1)
print("x train after flatten",X_train)
X_test = X_test.reshape(X_test.shape[0], -1)
print("x test after flatten", X_test)

# for convert classes into the vectors
def ohe(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train = ohe(y_train)
y_test = ohe(y_test)

print(y_train)
print(y_test)