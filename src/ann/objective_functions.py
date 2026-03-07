import numpy as np


class CrossEntropyLoss:
    """
    Cross-Entropy loss with numerically stable softmax applied internally.
    The model outputs raw logits; softmax is applied here, not in the network.
    """

    def __init__(self):
        self.y_true = None
        self.y_pred = None  # stores softmax probabilities after forward()

    def forward(self, logits, y_true):
        """
        Args:
            logits: raw output of network, shape (N, C)
            y_true: one-hot labels, shape (N, C)
        Returns:
            scalar loss value
        """
        self.y_true = y_true

        # Numerically stable softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.y_pred = exp / np.sum(exp, axis=1, keepdims=True)

        # Cross-entropy: -sum(y_true * log(softmax)) / N
        loss = -np.sum(y_true * np.log(self.y_pred + 1e-12)) / y_true.shape[0]
        return loss

    def backward(self):
        """
        Combined gradient of softmax + cross-entropy w.r.t. logits.
        dL/d_logits = (softmax - y_true) / N
        This is the correct gradient to pass back to the output layer.
        """
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


class MSELoss:
    """
    Mean Squared Error loss.
    No softmax — y_pred are raw logits compared directly to y_true.
    """

    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: raw logits, shape (N, C)
            y_true: one-hot labels, shape (N, C)
        Returns:
            scalar loss = mean over all elements
        """
        self.y_true = y_true
        self.y_pred = y_pred
        # np.mean divides by N*C (all elements)
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def backward(self):
        """
        Gradient of MSE loss w.r.t. y_pred.
        d/dy_pred mean((y_true - y_pred)^2)
        = -2*(y_true - y_pred) / (N*C)
        = 2*(y_pred - y_true) / (N*C)

        Divide by N*C to be consistent with np.mean in forward().
        """
        N, C = self.y_true.shape
        return 2.0 * (self.y_pred - self.y_true) / (N * C)