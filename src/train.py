from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork

import wandb
import argparse
import os
import json
import numpy as np
from argparse import Namespace
from sklearn.metrics import f1_score


def save_model(model, args):

    os.makedirs("../models", exist_ok=True)

    best_weights = model.get_weights()

    np.save("../models/best_model.npy", best_weights)

    config = vars(args)

    with open("../models/best_config.json", "w") as f:
        json.dump(config, f, indent=4)


def build_parser():
    """
    Shared argument parser used by both train.py and inference.py.
    Per updated instructions: both CLIs must be identical,
    with best config values as defaults.
    """
    parser = argparse.ArgumentParser(description='Train / Evaluate a MLP neural network')

    parser.add_argument("-wp", "--wandb_project",
                        type=str,
                        default="da6401_assignment1",
                        help="Weights and Biases Project ID")

    parser.add_argument("-d", "--dataset",
                        type=str,
                        default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to use")

    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=10,
                        help="Number of training epochs")

    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=64,
                        help="Mini-batch size")

    parser.add_argument("-l", "--loss",
                        type=str,
                        default="cross_entropy",
                        choices=["cross_entropy", "mse"],
                        help="Loss Function")

    # Updated spec only requires: sgd, momentum, nag, rmsprop
    parser.add_argument("-o", "--optimizer",
                        type=str,
                        default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        help="Optimizer type")

    parser.add_argument("-lr", "--learning_rate",
                        type=float,
                        default=0.001,
                        help="Learning rate")

    parser.add_argument("-wd", "--weight_decay",
                        type=float,
                        default=0.0,
                        help="L2 regularization strength")

    parser.add_argument("-nhl", "--num_layers",
                        type=int,
                        default=3,
                        help="Number of hidden layers")

    parser.add_argument("-sz", "--hidden_size",
                        type=str,
                        default="128 128 128",
                        help="Hidden layer sizes e.g. '128 128 128' or '[128,128,128]'")

    parser.add_argument("-a", "--activation",
                        type=str,
                        default="relu",
                        choices=["relu", "sigmoid", "tanh"],
                        help="Activation function")

    parser.add_argument("-w_i", "--weight_init",
                        type=str,
                        default="xavier",
                        choices=["random", "xavier"],
                        help="Weight initialization method")

    # Shared with inference.py (ignored by train.py)
    parser.add_argument("-mp", "--model_path",
                        type=str,
                        default=None,
                        help="Path to saved model weights (.npy file)")

    return parser


def parse_hidden_size(raw):
    """Handle hidden_size as string '128 128', '[128,128]', or already a list."""
    if isinstance(raw, list):
        return raw
    clean = str(raw).replace('[', '').replace(']', '').replace(',', ' ')
    return [int(x) for x in clean.split()]


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.hidden_size = parse_hidden_size(args.hidden_size)

    if args.num_layers != len(args.hidden_size):
        raise ValueError(
            f"num_layers ({args.num_layers}) must match "
            f"length of hidden_size ({len(args.hidden_size)})"
        )

    try:
        wandb.init(project=args.wandb_project, config=vars(args))
        use_wandb = True
    except Exception as e:
        print(f"W&B init failed ({e}), continuing without logging.")
        use_wandb = False

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    model = NeuralNetwork(args)
    best_f1 = 0.0
    test_f1 = 0.0

    for epoch in range(args.epochs):

        train_loss, grad_norm = model.train(X_train, y_train, 1, args.batch_size)
        val_acc = model.evaluate(X_val, y_val)

        # Compute test F1 after each epoch
        test_logits = model.forward(X_test)
        pred = np.argmax(test_logits, axis=1)
        true = np.argmax(y_test, axis=1)
        test_f1 = f1_score(true, pred, average="macro")

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Test F1: {test_f1:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "test_f1": test_f1,
                "grad_norm_layer1": grad_norm
            })

        if test_f1 > best_f1:
            best_f1 = test_f1
            save_model(model, args)
            print(f"  -> New best model saved (F1: {test_f1:.4f})")

    print(f"\nTraining complete. Best Test F1: {best_f1:.4f}")

    if use_wandb:
        wandb.finish()


def train_sweep():
    """Entry point for W&B hyperparameter sweeps."""
    wandb.init()
    config = wandb.config

    args = Namespace(
        dataset="mnist",
        epochs=5,
        batch_size=config.batch_size,
        loss="cross_entropy",
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        activation=config.activation,
        weight_init="xavier"
    )
    args.hidden_size = parse_hidden_size(args.hidden_size)

    X_train, y_train, X_val, y_val, _, _ = load_data("mnist")
    model = NeuralNetwork(args)

    for epoch in range(args.epochs):
        train_loss, grad_norm = model.train(X_train, y_train, 1, args.batch_size)
        val_acc = model.evaluate(X_val, y_val)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc})


if __name__ == "__main__":
    main()