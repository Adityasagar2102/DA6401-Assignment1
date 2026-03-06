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

def main():

    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument("-wp","--wandb_project", type=str,
                        required=True, help="W&B project name")

    parser.add_argument("-d","--dataset", type=str, required=True,
                        choices=["mnist", "fashion_mnist"], help="Dataset to use")

    parser.add_argument("-e", "--epochs", type=int, required=True,
                        help="Number of training epochs")

    parser.add_argument("-b", "--batch_size", type=int, required=True,
                        help="Mini-batch size")

    parser.add_argument("-l", "--loss", type=str, required=True,
                        choices=["cross_entropy", "mse"],help="Loss Function")

    parser.add_argument("-o", "--optimizer", type=str, required=True,
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        help="Optimizer type")

    parser.add_argument("-lr", "--learning_rate", type=float, required=True,
                        help="Learning rate")

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0,
                        help="L2 regularization strength")

    parser.add_argument("-nhl", "--num_layers", type=int, required=True,
                        help="Number of hidden layers")

    # parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, required=True,
    #                     help="List of hidden layer sizes")

    parser.add_argument("-sz", "--hidden_size", type=str, required=True,
                        help="List of hidden layer sizes")

    parser.add_argument("-a", "--activation", type=str, required=True,
                        choices=["relu", "sigmoid", "tanh"],help="Activation function")

    parser.add_argument("-w_i", "--weight_init", type=str, required=True,
                        choices=["random", "xavier"], help="Weight initialization method")

    args = parser.parse_args()

    # convert hidden_size string to list
    # args.hidden_size = [int(x) for x in args.hidden_size.split()]
    # print(wandb)
    raw_sz = args.hidden_size
    clean_sz = raw_sz.replace('[', '').replace(']', '').replace(',', '')
    args.hidden_size = [int(x) for x in clean_sz.split()]

    # Ensure it looks like this:
    wandb.init(project=args.wandb_project, config=vars(args))

    if args.num_layers != len(args.hidden_size):
        raise ValueError("num_layers must match lenght of hidden_size list")

    X_train, y_train, X_val, y_val, X_test,y_test = load_data(args.dataset)

    model = NeuralNetwork(args)
    best_f1 = 0.0
    test_f1 = 0.0
    # model.train(X_train,y_train, args.epochs, args.batch_size)
    # loop to check best accuracy at each epoch
    # ---------------------------------------------------
    for epoch in range(args.epochs):

        train_loss, grad_norm = model.train(X_train, y_train, 1, args.batch_size)

        val_acc = model.evaluate(X_val, y_val)

        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "test_f1": test_f1,
            "grad_norm_layer1": grad_norm
        })

        # -------- NEW CODE --------
        test_logits = model.forward(X_test)

        pred = np.argmax(test_logits, axis=1)
        true = np.argmax(y_test, axis=1)

        test_f1 = f1_score(true, pred, average="macro")

        if test_f1 > best_f1:
            best_f1 = test_f1

            save_model(model, args)

            print(f"New best model saved with F1 score: {test_f1:.4f}")
    # ----------------------------------    ------------------------


    # val_acc = model.evaluate(X_val,y_val)
    print("Training complete")
    print("Best Test F1 Score:", best_f1)

    wandb.finish()

# import wandb

def train_sweep():
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

    X_train, y_train, X_val, y_val, _, _ = load_data("mnist")

    model = NeuralNetwork(args)

    for epoch in range(args.epochs):

        train_loss, grad_norm = model.train(X_train, y_train, 1, args.batch_size)

        val_acc = model.evaluate(X_val, y_val)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_acc
        })

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------




