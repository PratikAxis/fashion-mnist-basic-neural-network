import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_metrics(history: dict):
    """Plot training loss and accuracy curves.

    Expects `history` to contain keys: 'train_loss', 'train_acc', 'test_acc'.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'r-', label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], 'b-', label='train acc')
    plt.plot(epochs, history.get("test_acc", []), 'g--', label='test acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray, class_names=None):
    """Plot confusion matrix given one-hot true labels and predicted probabilities.

    y_true_onehot: (n, num_classes)
    y_pred_probs: (n, num_classes)
    """
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    if class_names is None:
        class_names = [str(i) for i in range(y_pred_probs.shape[1])]

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
