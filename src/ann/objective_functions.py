import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        self.y_true = None
        self.y_pred = None

    # def forward(self, y_pred, y_true):
    #     # loss = sum for all point(-sum(y-y_true * log(y_pred)))/ batch size
    #     self.y_true = y_true
    #     self.y_pred = y_pred
    #
    #     val = 1e-12
    #     y_pred = np.clip(y_pred, val, 1.0 - val)
    #     loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    #     return loss

    def forward(self, logits, y_true):

        # numerical stability
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.y_true = y_true

        loss = -np.sum(y_true * np.log(self.probs + 1e-9)) / logits.shape[0]

        return loss


    def backward(self):

        batch_size = self.y_true.shape[0]

        dz = (self.probs - self.y_true) / batch_size

        return dz


class MSELoss:

    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred, y_true):

        self.y_true = y_true
        self.y_pred = y_pred
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def backward(self):
        # dz = 2(y_pred - y_true)/N
        return 2*(self.y_pred - self.y_true)/self.y_true.shape[0]