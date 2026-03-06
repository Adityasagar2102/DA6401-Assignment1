import numpy as np
from .neural_layer import NeuralLayer
from .activations import ReLU, Sigmoid, Tanh
from .objective_functions import CrossEntropyLoss, MSELoss
from .optimizers import SGD, Momentum, NAG, RMSProp



class NeuralNetwork:

    
    def __init__(self, cli_args):
        self.args = cli_args
        self.args.optimizer = self.args.optimizer.lower()
        self.layers = []
        self.activations = []

        input_dim = 784
        output_dim = 10

        hidden_sizes = self.args.hidden_size
        activation_name = self.args.activation
        weight_init = self.args.weight_init

        prev_dim = input_dim

        # BUILD HIDDEN LAYERS
        for hidden_dim in hidden_sizes:
            layer = NeuralLayer(prev_dim, hidden_dim, weight_init)
            self.layers.append(layer)

            if activation_name == "relu":
                self.activations.append(ReLU())
            elif activation_name == "sigmoid":
                self.activations.append(Sigmoid())
            elif activation_name == "tanh":
                self.activations.append(Tanh())
            else:
                raise ValueError("Invalid Activation Name")
            prev_dim = hidden_dim

        # OUTPUT LAYER
        self.layers.append(NeuralLayer(prev_dim,output_dim,weight_init))
        # self.activations.append(Softmax())

        # LOSS FUNCTION
        if self.args.loss == "cross_entropy":
            self.loss = CrossEntropyLoss()
        elif self.args.loss == "mse":
            self.loss = MSELoss()
        else:
            raise ValueError("Invalid loss function")

        # OPTIMIZER
        lr = self.args.learning_rate

        if self.args.optimizer == "sgd":
            self.optimizer = SGD(self.layers, lr)
        elif self.args.optimizer == "momentum":
            self.optimizer = Momentum(self.layers, lr)
        elif self.args.optimizer == "nag":
            self.optimizer = NAG(self.layers, lr)
        elif self.args.optimizer == "rmsprop":
            self.optimizer = RMSProp(self.layers, lr)
        # elif self.args.optimizer == "adam":
        #     self.optimizer = Adam(self.layers, lr)
        # elif self.args.optimizer == "nadam":
        #     self.optimizer = Nadam(self.layers, lr)
        else:
            raise ValueError("Invalid optimizer")

        pass

    def forward(self, X):

        a = X

        for layer, activation in zip(self.layers[:-1], self.activations):
            z = layer.forward(a)

            a = activation.forward(z)

        logits = self.layers[-1].forward(a)

        return logits

    def backward(self, y_true, y_pred):

        dz = self.loss.backward()

        # output layer gradient
        dz = self.layers[-1].backward(dz)

        # hidden layers
        for layer, activation in reversed(list(zip(self.layers[:-1], self.activations))):
            dz = activation.backward(dz)

            dz = layer.backward(dz)

        grad_W = []
        grad_b = []

        for layer in self.layers:

            if self.args.weight_decay > 0:
                layer.grad_W += self.args.weight_decay * layer.W

            grad_W.append(layer.grad_W)
            grad_b.append(layer.grad_b)

        return grad_W, grad_b

    def update_weights(self):
        self.optimizer.step()

    def train(self, X_train, y_train, epochs, batch_size):
        for epoch in range(epochs):
            indices = np.random.permutation(X_train.shape[0])
            X_train = X_train[indices]
            y_train = y_train[indices]
            # print("First 5 shuffled indices:", indices[:5])

            total_loss = 0
            num_batches = 0

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                y_pred = self.forward(X_batch)

                loss = self.loss.forward(y_pred,y_batch)
                total_loss +=loss
                num_batches += 1

                self.backward(y_batch,y_pred)
                first_layer_grad_norm = np.linalg.norm(self.layers[0].grad_W)

                self.update_weights()
            avg_loss = total_loss/num_batches
            # print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        return avg_loss,first_layer_grad_norm

    
    def evaluate(self, X, y):
        y_pred = self.forward(X)

        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == true_labels)

        return accuracy

    def get_weights(self):

        weights = {}

        for i, layer in enumerate(self.layers):
            weights[f"W{i}"] = layer.W

            weights[f"b{i}"] = layer.b

        return weights

    def set_weights(self, weights):

        for i, layer in enumerate(self.layers):
            layer.W = weights[f"W{i}"]

            layer.b = weights[f"b{i}"]
