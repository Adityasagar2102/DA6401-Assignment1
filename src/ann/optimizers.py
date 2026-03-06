import numpy as np

class SGD:
    # W = W - eta * grad_w
    def __init__(self,layers,lr):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

class Momentum:
    # v = beta*v + eta * grad_W
    # W = W - v
    def __init__(self, layers, lr, beta = 0.9):
        self.layers = layers
        self.lr = lr
        self.beta = beta

        # we are creating empty list to store velocity values for each layer
        self.v_W = []
        self.v_b = []

        for layer in layers:
            zero_W = np.zeros_like(layer.W) # create a zero array with the same shape as layer.b
            zero_b = np.zeros_like(layer.b)

            self.v_W.append(zero_W)
            self.v_b.append(zero_b)

    def step(self):
        for i, layer in enumerate(self.layers):
            self.v_W[i] = self.beta * self.v_W[i] + self.lr*layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + self.lr*layer.grad_b

            layer.W -= self.v_W[i]
            layer.b -= self.v_b[i]

class NAG:
    def __init__(self, layers, lr, beta=0.9):
        self.layers = layers
        self.lr = lr
        self.beta = beta

        self.v_W = []
        self.v_b = []

        for layer in layers:
            self.v_W.append(np.zeros_like(layer.W))
            self.v_b.append(np.zeros_like(layer.b))

    def step(self):
        for i, layer in enumerate(self.layers):

            v_W_prev = self.v_W[i]
            v_b_prev = self.v_b[i]

            # Update velocity
            self.v_W[i] = self.beta * self.v_W[i] - self.lr * layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] - self.lr * layer.grad_b

            # Nesterov update
            layer.W += -self.beta * v_W_prev + (1 + self.beta) * self.v_W[i]
            layer.b += -self.beta * v_b_prev + (1 + self.beta) * self.v_b[i]

class RMSProp:
    # s = beta * s + (1-beta)grad_w^2
    # W = W - (eta*grad_w)/(sqrt(s) + eps)
    def __init__(self, layers, lr, beta=0.9, epsilon=1e-8):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        self.eps = epsilon

        # we are creating empty list to store velocity values for each layer
        self.s_W = []
        self.s_b = []

        for layer in layers:
            zero_W = np.zeros_like(layer.W)  # create a zero array with the same shape as layer.b
            zero_b = np.zeros_like(layer.b)

            self.s_W.append(zero_W)
            self.s_b.append(zero_b)

    def step(self):
        for i, layer in enumerate(self.layers):

            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta)*(layer.grad_W ** 2)
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta)*(layer.grad_b ** 2)

            layer.W -= self.lr*layer.grad_W / (np.sqrt(self.s_W[i]) + self.eps)
            layer.b -= self.lr*layer.grad_b / (np.sqrt(self.s_b[i]) + self.eps)

# class Adam:
#     # m = beta1*m + (1 - beta1)*grad_W
#     # v = beta2*v + (1 - beta2)*grad_W^2
#     #
#     # bias correction
#     # mt = m/(1-beta1^T) T is time
#     # vt = v/(1-beta2^T)
#     #
#     # update
#     # w = w - eta*mt / (sqrt{vt} + eps)
#
#     def __init__(self, layers, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
#         self.layers = layers
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = epsilon
#
#         # we are creating empty list to store velocity values for each layer
#         self.v_W = []
#         self.v_b = []
#         self.m_W = []
#         self.m_b = []
#
#         for layer in layers:
#             self.v_W.append(np.zeros_like(layer.W))
#             self.v_b.append(np.zeros_like(layer.b))
#
#             self.m_W.append(np.zeros_like(layer.W))
#             self.m_b.append(np.zeros_like(layer.b))
#         self.t = 0
#     def step(self):
#         self.t +=1
#         for i, layer in enumerate(self.layers):
#             self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
#             self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b
#
#             self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.grad_W ** 2)
#             self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.grad_b ** 2)
#
#             # bias correction
#             m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
#             m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
#
#             v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
#             v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
#
#             # update
#             layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
#             layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
#
# class Nadam:
#     def __init__(self, layers, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
#         self.layers = layers
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = epsilon
#
#         # we are creating empty list to store velocity values for each layer
#         self.v_W = []
#         self.v_b = []
#         self.m_W = []
#         self.m_b = []
#
#         for layer in layers:
#             self.v_W.append(np.zeros_like(layer.W))
#             self.v_b.append(np.zeros_like(layer.b))
#
#             self.m_W.append(np.zeros_like(layer.W))
#             self.m_b.append(np.zeros_like(layer.b))
#         self.t = 0
#     def step(self):
#         self.t += 1
#         for i, layer in enumerate(self.layers):
#             self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
#             self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b
#
#             self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.grad_W ** 2)
#             self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.grad_b ** 2)
#
#             # bias correction
#             m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
#             m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
#
#             v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
#             v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
#
#             # Nadam update
#             nes_W = self.beta1 * m_W_hat + (1 - self.beta1)*layer.grad_W/(1 - self.beta1 ** self.t)
#             nes_b = self.beta1 * m_b_hat + (1 - self.beta1)*layer.grad_b/(1 - self.beta1 ** self.t)
#
#             layer.W -= self.lr * nes_W / (np.sqrt(v_W_hat) + self.eps)
#             layer.b -= self.lr * nes_b / (np.sqrt(v_b_hat) + self.eps)





