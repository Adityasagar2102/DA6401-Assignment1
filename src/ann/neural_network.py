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
                raise ValueError(f"Invalid activation: {activation_name}")

            prev_dim = hidden_dim

        # OUTPUT LAYER — NO activation here.
        # Per updated spec: model must return LOGITS (raw linear combination),
        # NOT softmax-activated outputs. Softmax is applied inside the loss function.
        self.layers.append(NeuralLayer(prev_dim, output_dim, weight_init))

        # LOSS FUNCTION
        if self.args.loss == "cross_entropy":
            self.loss = CrossEntropyLoss()
        elif self.args.loss == "mse":
            self.loss = MSELoss()
        else:
            raise ValueError(f"Invalid loss: {self.args.loss}")

        # OPTIMIZER — updated spec: sgd, momentum, nag, rmsprop only
        lr = self.args.learning_rate
        opt = self.args.optimizer

        if opt == "sgd":
            self.optimizer = SGD(self.layers, lr)
        elif opt == "momentum":
            self.optimizer = Momentum(self.layers, lr)
        elif opt == "nag":
            self.optimizer = NAG(self.layers, lr)
        elif opt == "rmsprop":
            self.optimizer = RMSProp(self.layers, lr)
        else:
            raise ValueError(f"Invalid optimizer: {opt}")

    def forward(self, X):
        """
        Forward pass through all layers.
        Returns RAW LOGITS from the output layer (no softmax).
        Per updated spec: model must return logits, not softmax outputs.
        """
        a = X
        for layer, activation in zip(self.layers[:-1], self.activations):
            z = layer.forward(a)
            a = activation.forward(z)

        # Final layer: linear output (logits only, no softmax)
        logits = self.layers[-1].forward(a)
        return logits

    def backward(self, y_true, y_pred):
        """
        Backward pass. Computes and STORES gradients in each layer's
        self.grad_W and self.grad_b.

        Per updated spec: must compute and RETURN gradients from
        last layer to first (output layer first, input layer last).

        Returns:
            grad_W: list of weight gradients [output_layer, ..., first_hidden_layer]
            grad_b: list of bias gradients   [output_layer, ..., first_hidden_layer]
        """
        # loss.backward() uses y_pred and y_true stored during loss.forward()
        dz = self.loss.backward()

        # Output layer backward
        dz = self.layers[-1].backward(dz)

        # Hidden layers backward (reverse order)
        for layer, activation in reversed(list(zip(self.layers[:-1], self.activations))):
            dz = activation.backward(dz)
            dz = layer.backward(dz)

        # Apply L2 weight decay to all layer gradients
        for layer in self.layers:
            if self.args.weight_decay > 0:
                layer.grad_W += self.args.weight_decay * layer.W

        # Return gradients from LAST layer to FIRST (output -> input)
        # per updated spec requirement
        grad_W = [layer.grad_W for layer in reversed(self.layers)]
        grad_b = [layer.grad_b for layer in reversed(self.layers)]

        return grad_W, grad_b

    def update_weights(self):
        self.optimizer.step()

    def train(self, X_train, y_train, epochs, batch_size):
        """Train for given number of epochs. Returns (avg_loss, grad_norm_layer1)."""
        avg_loss = 0.0
        first_layer_grad_norm = 0.0

        for epoch in range(epochs):
            # Shuffle training data each epoch
            indices = np.random.permutation(X_train.shape[0])
            X_train = X_train[indices]
            y_train = y_train[indices]

            total_loss = 0.0
            num_batches = 0

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward
                y_pred = self.forward(X_batch)

                # Loss (stores y_pred and y_true for backward)
                loss = self.loss.forward(y_pred, y_batch)
                total_loss += loss
                num_batches += 1

                # Backward — gradients stored in each layer
                self.backward(y_batch, y_pred)

                # Track first hidden layer gradient norm
                first_layer_grad_norm = np.linalg.norm(self.layers[0].grad_W)

                # Update weights
                self.update_weights()

            avg_loss = total_loss / num_batches

        return avg_loss, first_layer_grad_norm

    def evaluate(self, X, y):
        """Returns classification accuracy."""
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)

    def get_weights(self):
        """Return all layer weights as a flat dictionary."""
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"W{i}"] = layer.W
            weights[f"b{i}"] = layer.b
        return weights

    def set_weights(self, weights):
        """Load weights from dictionary into all layers."""
        for i, layer in enumerate(self.layers):
            layer.W = weights[f"W{i}"]
            layer.b = weights[f"b{i}"]