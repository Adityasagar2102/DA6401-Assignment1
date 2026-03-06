import argparse
import json
import numpy as np
from argparse import Namespace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("-mp", "--model_path",
                        type=str,
                        required=True,
                        help="Path to saved model weights (relative path)")

    parser.add_argument("-d", "--dataset",
                        type=str,
                        required=True,
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to evaluate on")

    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=64,
                        help="Batch size for inference")

    # parser.add_argument("-nhl", "--num_layers",
    #                     type=int,
    #                     required=True,
    #                     help="Number of hidden layers")
    #
    # parser.add_argument("-sz", "--hidden_size",
    #                     nargs="+",
    #                     type=int,
    #                     required=True,
    #                     help="List of hidden layer sizes")
    #
    # parser.add_argument("-a", "--activation",
    #                     type=str,
    #                     required=True,
    #                     choices=["relu", "sigmoid", "tanh"],
    #                     help="Activation function")

    # args = parser.parse_args()
    # if args.num_layers != len(args.hidden_size):
    #     raise ValueError("num_layers must match length of hidden_size list")


    return parser.parse_args()


def load_model(model_path):
    with open("../models/best_config.json","r") as f:
        config = json.load(f)
    args = Namespace(**config)

    model = NeuralNetwork(args)

    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)

    return model


def evaluate_model(model, X_test, y_test):
    logits = model.forward(X_test)

    predictions = np.argmax(logits, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")
    f1 = f1_score(true_labels, predictions, average="macro")

    loss = model.loss.forward(logits, y_test)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    args = parse_arguments()

    # Load test data
    _, _, _, _, X_test, y_test = load_data(args.dataset)

    # Load model
    model = load_model(args.model_path)

    # Evaluate
    results = evaluate_model(model, X_test, y_test)

    print("Evaluation Results")
    print("------------------")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

    return results


if __name__ == '__main__':
    main()
